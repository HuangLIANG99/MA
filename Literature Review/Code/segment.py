import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import s3fs

# ============================================================
# 0. 用户配置区
# ============================================================
TARGET_FILE = (
    "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_003/"
    "J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2024-10-03_142512="
    "jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS014653 _ Format01=Kreis M3-034=filesize-109888838=finished.parquet"
)

OUTPUT_DIR = "output"
CSV_SEPARATOR = ';'        # 分号作为列分隔
CSV_DECIMAL = ','          # 🟢 关键修改：用逗号作为小数点，彻底解决 Excel 数值放大问题

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. MinIO / S3 配置与数据读取
# ============================================================
def get_s3_storage_options():
    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")
    if key is None or secret is None:
        raise ValueError("未找到环境变量 MINIO_ACCESS_KEY 或 MINIO_SECRET_KEY，请先在终端设置。")
    return {
        "key": key, "secret": secret,
        "client_kwargs": {"endpoint_url": "https://iseadocker.isea.rwth-aachen.de:9000", "region_name": "us-east-1"},
        "config_kwargs": {"s3": {"addressing_style": "path"}, "signature_version": "s3v4"}
    }

def read_battery_data(target_file: str):
    storage_options = get_s3_storage_options()
    s3_path = target_file if target_file.startswith("s3://") else f"s3://{target_file}"
    print(f"📄 正在从 S3 读取原始文件: {s3_path}")
    return pd.read_parquet(s3_path, storage_options=storage_options)

# =====================================================================
# 2. 主分析程序
# =====================================================================
def main():
    df = read_battery_data(TARGET_FILE)

    # 规范化列名映射
    COL_TIME, COL_CURRENT, COL_VOLTAGE, COL_AH, COL_STATE = 'Zeit', 'Strom', 'Spannung', 'AhAkku', 'Zustand'
    
    # 强制状态列清除空格并统一大写
    df['Zustand_Clean'] = df[COL_STATE].astype(str).str.strip().str.upper()

    # 时间轴数值化处理（秒数），用于画图对齐
    if pd.api.types.is_datetime64_any_dtype(df[COL_TIME]) or pd.api.types.is_timedelta64_any_dtype(df[COL_TIME]):
        df['Time_Seconds'] = (df[COL_TIME] - df[COL_TIME].iloc[0]).dt.total_seconds()
    else:
        df['Time_Seconds'] = pd.to_numeric(df[COL_TIME], errors='coerce') - pd.to_numeric(df[COL_TIME].iloc[0], errors='coerce')

    # 基于状态切换点（change_points）定位目标片段边界
    print("🔍 正在通过状态切换点追踪放电时段边界...")
    df_temp = df[['Time_Seconds', 'Zustand_Clean']].copy()
    df_temp['prev_state'] = df_temp['Zustand_Clean'].shift()
    
    # 捕获所有状态发生切换的瞬间行
    change_points = df_temp[df_temp['Zustand_Clean'] != df_temp['prev_state']].copy().reset_index()

    # 扫描定位：从 DCH 开始，直到下一个 CHA 充电前一行的完整闭环
    valid_segments = []
    for k in range(len(change_points)):
        if change_points.loc[k, 'Zustand_Clean'] == 'DCH':
            start_df_index = change_points.loc[k, 'index']
            
            end_df_index = None
            for look_ahead in range(k + 1, len(change_points)):
                next_state = change_points.loc[look_ahead, 'Zustand_Clean']
                if next_state == 'CHA':
                    end_df_index = change_points.loc[look_ahead, 'index'] - 1
                    break
                elif next_state == 'DCH':
                    break
            
            if end_df_index is not None:
                valid_segments.append((start_df_index, end_df_index))

    total_found = len(valid_segments)
    print(f"⚡ 全周期共成功识别出【放电开始 -> 静置完毕】片段：{total_found} 个。")

    if total_found == 0:
        print("❌ 未能成功匹配到完整的放电到充电前的状态闭环。")
        return

    # 全生命周期等间距均匀抽取 5 个片段
    selected_indices = np.linspace(0, total_found - 1, 5, dtype=int) if total_found >= 5 else np.arange(total_found)
    
    # =====================================================================
    # 3. 循环抽取、独立分列导出表格与一图一存
    # =====================================================================
    summary_reports = []
    labels = ['Early Stage', 'Pre-Mid Stage', 'Mid Stage', 'Late Stage', 'End of Life']

    for rank, idx in enumerate(selected_indices):
        start_idx, end_idx = valid_segments[idx]
        
        # 从原始大 DataFrame 中精准捞取数据切片
        seg_df = df.loc[start_idx:end_idx].copy()
        
        # 相对时间轴（仅供画图对齐使用）
        seg_df['Relative_Time_s'] = seg_df['Time_Seconds'] - seg_df['Time_Seconds'].iloc[0]
        
        # 提取汇总特征指标
        duration = seg_df['Relative_Time_s'].max()
        v_start = seg_df[COL_VOLTAGE].iloc[0]
        v_end = seg_df[COL_VOLTAGE].iloc[-1]
        v_min = seg_df[COL_VOLTAGE].min()
        ah_start = seg_df[COL_AH].iloc[0]
        ah_end = seg_df[COL_AH].iloc[-1]
        ah_delta = ah_end - ah_start
        i_min = seg_df[COL_CURRENT].min()
        raw_start_time = seg_df[COL_TIME].iloc[0]

        summary_reports.append({
            "片段编号": rank + 1,
            "全周期绝对起点": str(raw_start_time),
            "总时间跨度(秒)": round(duration, 1),
            "最大放电电流(A)": round(i_min, 3),
            "初始电压(V)": round(v_start, 4),
            "最低电压(V)": round(v_min, 4),
            "结束电压(V)": round(v_end, 4),
            "安时变化量(Ah)": round(ah_delta, 4)
        })

        # 🟢 【修改点 1 & 2】剔除多余列，并使用 Excel 欧洲标准(分号分隔+逗号小数点)导出
        # 彻底移除辅助列以及 Relative_Time_s 列，确保 CSV 文件只有原始列且各自独立成列
        export_df = seg_df.drop(columns=['Zustand_Clean', 'Time_Seconds', 'Relative_Time_s'], errors='ignore')
        
        csv_name = os.path.join(OUTPUT_DIR, f"discharge_segment_{rank+1}.csv")
        # 引入 decimal=CSV_DECIMAL 修正小数点识别错误
        export_df.to_csv(csv_name, index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL, encoding='utf-8-sig')
        print(f"💾 [分列数据已成功导出] 片段 {rank+1} ({labels[rank]}) -> {csv_name}")

        # 一图一存（保持内存隔离）
        fig, axes = plt.subplots(3, 1, figsize=(11, 9))
        
        # 子图 1：电流
        axes[0].plot(seg_df['Relative_Time_s'], seg_df[COL_CURRENT], color='#d62728', linewidth=1.8)
        axes[0].set_ylabel("Current / Strom (A)", fontsize=10)
        axes[0].set_title(f"Discharge Segment {rank+1} ({labels[rank]}) - Independent Profile", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 子图 2：电压
        axes[1].plot(seg_df['Relative_Time_s'], seg_df[COL_VOLTAGE], color='#1f77b4', linewidth=1.8)
        axes[1].set_ylabel("Voltage / Spannung (V)", fontsize=10)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # 子图 3：安时相对增量
        axes[2].plot(seg_df['Relative_Time_s'], seg_df[COL_AH] - ah_start, color='#2ca02c', linewidth=1.8)
        axes[2].set_ylabel("Δ AhAkku (Ah)", fontsize=10)
        axes[2].set_xlabel("Relative Time from DCH Start (seconds)", fontsize=10)
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, f"discharge_plot_segment_{rank+1}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"🖼️  [图像已独立保存] 片段 {rank+1} ({labels[rank]}) -> {plot_path}")

    # 输出汇总报告
    summary_df = pd.DataFrame(summary_reports)
    print("\n" + "="*105 + "\n 📊 5 组放电片段数据变化汇总报告 \n" + "="*105)
    print(summary_df.to_string(index=False))
    print("="*105)

if __name__ == "__main__":
    main()
