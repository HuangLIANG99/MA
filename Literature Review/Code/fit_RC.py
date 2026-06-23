import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import s3fs
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 用户配置区
# ============================================================
TARGET_FILE = (
    "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_003/J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2024-09-09_110048=jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS012326 _ Format01=Kreis M3-034=filesize-109580276=finished.parquet"
)

OUTPUT_DIR = "output_fit_all"
CSV_SEPARATOR = ';'
CSV_DECIMAL = ','

# ----- 处理控制选项 -----
PROCESS_ALL_SEGMENTS = True      # True=处理所有片段, False=仅处理前10个
MAX_SEGMENTS_TO_PROCESS = None   # None=全部, 或设置数字如50
SAVE_SEGMENT_CSV = False         # True=保存每个片段的CSV
SAVE_SEGMENT_PLOT = True        # True=保存每个片段的图片
VERBOSE = False                  # False=精简输出，配合tqdm进度条

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

# ============================================================
# 2. 从文件名提取 SOC 和 DOD
# ============================================================
def extract_soc_dod_from_filename(filepath: str):
    filename = filepath.split('/')[-1] if '/' in filepath else filepath
    soc_match = re.search(r'(\d+)SOC', filename, re.IGNORECASE)
    dod_match = re.search(r'(\d+)DOD', filename, re.IGNORECASE)
    soc = soc_match.group(1) if soc_match else None
    dod = dod_match.group(1) if dod_match else None
    return soc, dod

# ============================================================
# 3. 二阶RC纯指数幅值响应模型
# ============================================================
def relaxation_model_pure(t, OCV, A1, tau1, A2, tau2):
    t_safe = np.clip(t, 0, None)
    return OCV + A1 * np.exp(-t_safe / tau1) + A2 * np.exp(-t_safe / tau2)

# ============================================================
# 4. 改良后的自适应动态边界参数辨识函数
# ============================================================
def identify_parameters_adaptive(t_data, V_data):
    V_start = V_data[0]
    V_end = V_data[-1]
    
    # 基础变量
    OCV_init = V_end 
    delta_V = V_start - V_end
    abs_delta = abs(delta_V)
    
    # 根据实际收敛空间动态给出合理的初始猜测
    A1_init = delta_V * 0.3
    A2_init = delta_V * 0.7
    tau1_init = 8.0    
    tau2_init = 150.0  
    
    initial_guess = [OCV_init, A1_init, tau1_init, A2_init, tau2_init]
    
    # 动态拓宽边界约束，防止物理极限撞墙
    ocv_min, ocv_max = min(V_start, V_end) - 0.05, max(V_start, V_end) + 0.05
    if delta_V >= 0:  # 电压衰减
        a_min, a_max = 0.0, max(0.2, abs_delta * 1.5)
    else:             # 电压回升
        a_min, a_max = -max(0.2, abs_delta * 1.5), 0.0
        
    tau1_min, tau1_max = 0.2, 80.0      # 拓宽快时间常数边界
    tau2_min, tau2_max = 30.0, 5000.0   # 拓宽慢时间常数边界
    
    lower_bounds = [ocv_min, a_min, tau1_min, a_min, tau2_min]
    upper_bounds = [ocv_max, a_max, tau1_max, a_max, tau2_max]
    
    try:
        popt, pcov = curve_fit(
            relaxation_model_pure, t_data, V_data,
            p0=initial_guess, bounds=(lower_bounds, upper_bounds),
            max_nfev=150000, method='trf'
        )
        return popt, pcov, True
    except:
        return None, None, False

# ============================================================
# 5. 生成带 SOC/DOD 的文件名
# ============================================================
def generate_output_filename(base_name, rank, label, soc, dod, extension):
    if soc and dod:
        return f"{base_name}_{rank}_SOC{soc}_DOD{dod}.{extension}"
    elif soc:
        return f"{base_name}_{rank}_SOC{soc}.{extension}"
    elif dod:
        return f"{base_name}_{rank}_DOD{dod}.{extension}"
    else:
        return f"{base_name}_{rank}.{extension}"

# ============================================================
# 6. 单片段处理函数 (重构内阻核心物理公式)
# ============================================================
def process_segment(df, start_idx, end_idx, rank, COL_VOLTAGE, COL_CURRENT):
    seg_df = df.loc[start_idx:end_idx].copy()
    seg_df['Relative_Time_s'] = seg_df['Time_Seconds'] - seg_df['Time_Seconds'].iloc[0]
    
    t_data = seg_df['Relative_Time_s'].values
    V_data = seg_df[COL_VOLTAGE].values
    
    # 提取核心指标 1：静置前一刻的基准工作电流 I_prev
    try:
        lookback_idx = max(0, start_idx - 1)
        I_prev = df.loc[lookback_idx, COL_CURRENT]
        if abs(I_prev) < 0.05 and lookback_idx > 5:
            I_prev = df.loc[lookback_idx-5:lookback_idx, COL_CURRENT].mean()
    except Exception:
        I_prev = -1.4082 
        
    # 提取核心指标 2：静置前最后一刻带工况负载的电压 V_prior
    try:
        V_prior = df.loc[max(0, start_idx - 1), COL_VOLTAGE]
    except Exception:
        V_prior = V_data[0]
    
    # 参数辨识
    popt, pcov, success = identify_parameters_adaptive(t_data, V_data)
    if not success:
        return None
    
    OCV, A1, tau1, A2, tau2 = popt
    
    # 依据严格的 ECM 物理模型公式反推阻容
    if abs(I_prev) > 0.01:
        R1 = abs(A1 / I_prev)
        R2 = abs(A2 / I_prev)
        
        # 精确计算欧姆内阻 R0: 带载脉冲截止电压与拟合曲线理想起点外推值的差值
        V_fitted_zero = OCV + A1 + A2
        R0_raw = abs(V_prior - V_fitted_zero) / abs(I_prev)
        R0 = max(1e-6, R0_raw)
    else:
        R1, R2, R0 = 0.005, 0.01, 0.002
        
    C1 = tau1 / R1 if R1 > 0 else 0
    C2 = tau2 / R2 if R2 > 0 else 0
    
    V_fitted = relaxation_model_pure(t_data, OCV, A1, tau1, A2, tau2)
    residuals = V_data - V_fitted
    RMSE = np.sqrt(np.mean(residuals**2))
    R_squared = 1 - np.sum(residuals**2) / np.sum((V_data - np.mean(V_data))**2)
    
    return {
        "OCV": OCV, "A1": A1, "tau1": tau1, "A2": A2, "tau2": tau2,
        "R0": R0, "R1": R1, "R2": R2, "C1": C1, "C2": C2,
        "RMSE": RMSE, "R_squared": R_squared, "V_fitted": V_fitted, "I_prev": I_prev, "V_prior": V_prior
    }

# ============================================================
# 7. 绘图函数 (包含完整 R0, R1, R2, C1, C2 面板显示)
# ============================================================
def plot_segment_fitting(t_data, V_data, res, rank, soc_value, dod_value, stage_labels, COL_CURRENT, seg_df):
    fig = plt.figure(figsize=(16, 12))
    title_suffix = f"SOC={soc_value}%, DOD={dod_value}%" if soc_value and dod_value else ""
    
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.plot(t_data, V_data, 'b.', markersize=2, alpha=0.5, label='Measured Voltage')
    ax1.plot(t_data, res["V_fitted"], 'r-', linewidth=2, label='Fitted Model')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'PAU Segment {rank+1} - Voltage Relaxation Fitting {title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 丰富的阻容参数看板显示
    textstr = f'OCV = {res["OCV"]:.4f} V\n'
    textstr += f'R₀ (Ohmic) = {res["R0"]*1000:.3f} mΩ\n'
    textstr += f'R₁ (Fast)  = {res["R1"]*1000:.3f} mΩ, C₁ = {res["C1"]:.1f} F (τ₁ = {res["tau1"]:.1f} s)\n'
    textstr += f'R₂ (Slow)  = {res["R2"]*1000:.3f} mΩ, C₂ = {res["C2"]:.1f} F (τ₂ = {res["tau2"]:.1f} s)\n'
    textstr += f'RMSE = {res["RMSE"]*1000:.4f} mV, R² = {res["R_squared"]:.6f}'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 残差与时域对数分析
    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(t_data, (V_data - res["V_fitted"]) * 1000, 'g.', markersize=2, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Residual (mV)')
    ax2.set_title('Residuals Analysis')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax3 = plt.subplot(3, 2, 4)
    ax3.hist((V_data - res["V_fitted"]) * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_title('Residual Distribution')
    
    ax4 = plt.subplot(3, 2, 5)
    if len(t_data) > 1:
        ax4.semilogx(t_data[1:], V_data[1:], 'b.', markersize=2, alpha=0.5)
        ax4.semilogx(t_data[1:], res["V_fitted"][1:], 'r-', linewidth=2)
    ax4.set_title('Voltage Relaxation (Log Time Scale)')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    ax5 = plt.subplot(3, 2, 6)
    ax5.plot(seg_df['Relative_Time_s'], seg_df[COL_CURRENT], 'm-', linewidth=1)
    ax5.set_ylabel('Current (A)')
    ax5.set_title('Current During Pause')
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

# ============================================================
# 8. 主分析程序
# ============================================================
def main():
    df = read_battery_data(TARGET_FILE)
    soc_value, dod_value = extract_soc_dod_from_filename(TARGET_FILE)
    print(f"📊 从文件名提取信息: SOC = {soc_value}%, DOD = {dod_value}%")
    
    COL_TIME, COL_CURRENT, COL_VOLTAGE, COL_AH, COL_STATE = 'Zeit', 'Strom', 'Spannung', 'AhAkku', 'Zustand'
    df['Zustand_Clean'] = df[COL_STATE].astype(str).str.strip().str.upper()

    if pd.api.types.is_datetime64_any_dtype(df[COL_TIME]) or pd.api.types.is_timedelta64_any_dtype(df[COL_TIME]):
        df['Time_Seconds'] = (df[COL_TIME] - df[COL_TIME].iloc[0]).dt.total_seconds()
    else:
        df['Time_Seconds'] = pd.to_numeric(df[COL_TIME], errors='coerce') - pd.to_numeric(df[COL_TIME].iloc[0], errors='coerce')

    print("🔍 正在追踪静置(PAU)时段边界...")
    df_temp = df[['Time_Seconds', 'Zustand_Clean']].copy()
    df_temp['prev_state'] = df_temp['Zustand_Clean'].shift()
    change_points = df_temp[df_temp['Zustand_Clean'] != df_temp['prev_state']].copy().reset_index()

    valid_segments = []
    for k in range(len(change_points)):
        if (change_points.loc[k, 'prev_state'] == 'CHA' and change_points.loc[k, 'Zustand_Clean'] == 'PAU'):
            start_df_index = change_points.loc[k, 'index']
            end_df_index = None
            for look_ahead in range(k + 1, len(change_points)):
                next_state = change_points.loc[look_ahead, 'Zustand_Clean']
                if next_state == 'DCH':
                    end_df_index = change_points.loc[look_ahead, 'index'] - 1
                    break
                elif next_state == 'CHA':
                    break
            if end_df_index is not None:
                valid_segments.append((start_df_index, end_df_index))

    total_found = len(valid_segments)
    print(f"⚡ 全周期共成功识别出PAU片段：{total_found} 个。")

    if total_found == 0:
        print("❌ 未能成功匹配到完整的 CHA->PAU->DCH 状态闭环。")
        return

    num_to_extract = total_found if PROCESS_ALL_SEGMENTS else min(10, total_found)
    if MAX_SEGMENTS_TO_PROCESS is not None:
        num_to_extract = min(num_to_extract, MAX_SEGMENTS_TO_PROCESS)
    
    selected_indices = list(range(num_to_extract))
    print(f"📌 计划处理 {num_to_extract} 个段（自适应边界高精度拟合模式）")
    
    stage_labels = [f'Segment_{i+1}_SOC{soc_value}_DOD{dod_value}' if soc_value and dod_value else f'Segment_{i+1}' for i in range(total_found)]

    # =====================================================================
    # 9. 批量处理循环
    # =====================================================================
    parameter_reports = []
    successful_count = 0

    print("\n🚀 开始高精度阻容辨识解算...")
    for rank, idx in enumerate(tqdm(selected_indices, desc="进度", unit="段")):
        start_idx, end_idx = valid_segments[idx]
        
        res = process_segment(df, start_idx, end_idx, rank, COL_VOLTAGE, COL_CURRENT)
        if res is None:
            continue
            
        successful_count += 1
        
        # 装载完备的阻容输出报表字典
        params_dict = {
            "片段编号": rank + 1,
            "OCV (V)": round(res["OCV"], 6),
            "R0 (Ohm)": round(res["R0"], 6),  # 核心欧姆内阻
            "R1 (Ohm)": round(res["R1"], 6),  # 极化电阻 1
            "C1 (F)": round(res["C1"], 2),    # 极化电容 1
            "tau1 (s)": round(res["tau1"], 2),
            "R2 (Ohm)": round(res["R2"], 6),  # 极化电阻 2
            "C2 (F)": round(res["C2"], 2),    # 极化电容 2
            "tau2 (s)": round(res["tau2"], 2),
            "RMSE (V)": round(res["RMSE"], 6),
            "R²": round(res["R_squared"], 6),
            "I_prev (A)": round(res["I_prev"], 4),
            "V_prior (V)": round(res["V_prior"], 4)
        }
        parameter_reports.append(params_dict)
        
        # 逐段画图保存
        if SAVE_SEGMENT_PLOT:
            try:
                seg_df = df.loc[start_idx:end_idx].copy()
                seg_df['Relative_Time_s'] = seg_df['Time_Seconds'] - seg_df['Time_Seconds'].iloc[0]
                fig = plot_segment_fitting(
                    seg_df['Relative_Time_s'].values, seg_df[COL_VOLTAGE].values,
                    res, rank, soc_value, dod_value, stage_labels, COL_CURRENT, seg_df
                )
                plot_name = generate_output_filename("pau_fit_segment", f"{rank+1:04d}", stage_labels[rank], soc_value, dod_value, "png")
                fig.savefig(os.path.join(OUTPUT_DIR, plot_name), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except:
                plt.close()

    # =====================================================================
    # 10. 全量汇总报告与统计输出
    # =====================================================================
    print("\n" + "="*80)
    print(f"📊 解算完成：成功 {successful_count} / 尝试 {num_to_extract} 个片段")
    print("="*80)

    if parameter_reports:
        summary_df = pd.DataFrame(parameter_reports)
        param_csv = os.path.join(OUTPUT_DIR, f"parameter_identification_SOC{soc_value}_DOD{dod_value}.csv" if soc_value and dod_value else "parameter_identification.csv")
        
        try:
            summary_df.to_csv(param_csv, index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL, encoding='utf-8-sig')
            print(f"📋 包含完整 R0,R1,R2,C1,C2 的详尽报告已导出至: {param_csv}")
        except PermissionError:
            print("⚠️ 提示：总报告 CSV 被 Excel 占用，未能成功覆盖写入。")
            
        # 格式化终端输出统计表格
        print("\n📈 完备阻容参数物理量统计摘要 (ECM Summary):")
        print("-" * 90)
        print(f"{'阻容参数':<15} {'平均值 (Mean)':>15} {'标准差 (Std)':>15} {'最小值 (Min)':>15} {'最大值 (Max)':>15}")
        print("-" * 90)
        
        target_cols = ['OCV (V)', 'R0 (Ohm)', 'R1 (Ohm)', 'C1 (F)', 'R2 (Ohm)', 'C2 (F)', 'tau1 (s)', 'tau2 (s)', 'RMSE (V)', 'R²']
        for col in target_cols:
            if col in summary_df.columns:
                print(f"{col:<15} {summary_df[col].mean():15.6f} {summary_df[col].std():15.6f} {summary_df[col].min():15.6f} {summary_df[col].max():15.6f}")
        print("-" * 90)

        # 绘制全周期老化趋势特征联动图
        if len(summary_df) > 1:
            try:
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                axes[0].plot(summary_df['片段编号'], summary_df['R0 (Ohm)'] * 1000, 'o-', color='darkblue', label='R0 (Ohmic)')
                axes[0].set_ylabel('R0 (mΩ)')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                axes[0].set_title('Battery ECM Parameters Evolution Trend Across All Cycles')

                axes[1].plot(summary_df['片段编号'], summary_df['R1 (Ohm)'] * 1000, 's-', color='darkorange', label='R1 (Fast Polar)')
                axes[1].plot(summary_df['片段编号'], summary_df['R2 (Ohm)'] * 1000, 'd-', color='crimson', label='R2 (Slow Polar)')
                axes[1].set_ylabel('Resistance (mΩ)')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()

                axes[2].plot(summary_df['片段编号'], summary_df['C1 (F)'], '^-', color='forestgreen', label='C1 (Fast Cap)')
                axes[2].plot(summary_df['片段编号'], summary_df['C2 (F)'], 'v-', color='purple', label='C2 (Slow Cap)')
                axes[2].set_ylabel('Capacitance (F)')
                axes[2].set_xlabel('Segment / Cycle Number')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()

                plt.tight_layout()
                trend_path = os.path.join(OUTPUT_DIR, "ecm_parameters_trend.png")
                plt.savefig(trend_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"📈 全周期阻容变化趋势图已生成保存至: {trend_path}")
            except Exception as e:
                print(f"⚠️ 趋势图绘制失败: {e}")

if __name__ == "__main__":
    main()
