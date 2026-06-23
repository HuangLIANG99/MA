import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import os
import s3fs
import re

# ============================================================
# 0. 用户配置区
# ============================================================
TARGET_FILE = (
    "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_003/J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2024-09-09_110048=jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS012326 _ Format01=Kreis M3-034=filesize-109580276=finished.parquet"
)

OUTPUT_DIR = "output_fit"
CSV_SEPARATOR = ';'
CSV_DECIMAL = ','

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
# 3. 二阶RC等效电路模型 - 电压弛豫方程
# ============================================================
def relaxation_model_pure(t, OCV, A1, tau1, A2, tau2):
    """
    无电流耦合的二阶RC电压弛豫方程
    U(t) = OCV + A1*exp(-t/tau1) + A2*exp(-t/tau2)
    """
    t_safe = np.clip(t, 0, None)
    return OCV + A1 * np.exp(-t_safe / tau1) + A2 * np.exp(-t_safe / tau2)


def relaxation_model_for_fitting(t, OCV, R1, tau1, R2, tau2, I):
    """
    用于拟合的模型（使用时间常数而非电容值，提高数值稳定性）
    
    参数:
    t: 时间 (秒)
    OCV: 开路电压 (V)
    R1, R2: 极化电阻 (Ohm)
    tau1, tau2: 时间常数 (秒), tau = R*C
    I: 放电电流 (A)
    
    返回: 端电压 U_t(t)
    """
    t_safe = np.clip(t, 0, None)
    U_t = OCV - I * R1 * np.exp(-t_safe / tau1) - I * R2 * np.exp(-t_safe / tau2)
    return U_t

# ============================================================
# 4. 参数辨识函数
# ============================================================
def identify_parameters_adaptive(t_data, V_data):
    """
    自适应初始猜测与边界的参数辨识
    """
    V_start = V_data[0]
    V_end = V_data[-1]
    
    # 理想状态下，长时间静置后电压接近 OCV
    OCV_init = V_end
    
    # 总极化电压幅值
    delta_V = V_start - V_end  # 充电后为正，放电后为负
    A1_init = delta_V * 0.3
    A2_init = delta_V * 0.7
    
    tau1_init = 15.0   # 短时间常数
    tau2_init = 150.0  # 长时间常数
    
    initial_guess = [OCV_init, A1_init, tau1_init, A2_init, tau2_init]
    
    # 动态设定 A1, A2 的上下限
    if delta_V >= 0: # 充电后静置，A 应该为正
        a_min, a_max = 0.0, 0.5
    else:            # 放电后静置，A 应该为负
        a_min, a_max = -0.5, 0.0
        
    lower_bounds = [V_end - 0.05, a_min, 0.5,  a_min, 5.0]
    upper_bounds = [V_end + 0.05, a_max, 100.0, a_max, 2000.0]
    
    popt, pcov = curve_fit(
        relaxation_model_pure,
        t_data, 
        V_data,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=100000,
        method='trf'
    )
    return popt, pcov


def identify_parameters_least_squares(t_data, V_data, I_discharge,
                                       initial_guess=None, bounds=None):
    """
    使用 scipy.optimize.least_squares 进行参数辨识（更灵活的非线性最小二乘法）
    
    返回:
    result: least_squares结果对象
    params: 最优参数 [OCV, R1, tau1, R2, tau2]
    """
    if initial_guess is None:
        OCV_init = V_data[0] + 0.05
        R1_init = 0.01
        tau1_init = 10
        R2_init = 0.02
        tau2_init = 100
        initial_guess = [OCV_init, R1_init, tau1_init, R2_init, tau2_init]
    
    if bounds is None:
        lower_bounds = [V_data[0] - 0.1, 1e-6, 1e-3, 1e-6, 1e-3]
        upper_bounds = [V_data[0] + 0.5, 1.0, 10000, 1.0, 100000]
        bounds = (lower_bounds, upper_bounds)
    
    def residuals(params):
        """残差函数"""
        OCV, R1, tau1, R2, tau2 = params
        V_pred = relaxation_model_for_fitting(t_data, OCV, R1, tau1, R2, tau2, I_discharge)
        return V_pred - V_data
    
    # 使用least_squares（Trust Region Reflective算法）
    result = least_squares(
        residuals,
        initial_guess,
        bounds=bounds,
        method='trf',
        max_nfev=100000,
        loss='linear',  # 标准最小二乘
        verbose=0
    )
    
    return result, result.x

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

# =====================================================================
# 6. 主分析程序
# =====================================================================
def main():
    df = read_battery_data(TARGET_FILE)

    # ----- 提取 SOC 和 DOD -----
    soc_value, dod_value = extract_soc_dod_from_filename(TARGET_FILE)
    print(f"📊 从文件名提取信息: SOC = {soc_value}%, DOD = {dod_value}%")
    
    # 生成标签
    if soc_value and dod_value:
        stage_labels = [f'Segment_{i+1}_SOC{soc_value}_DOD{dod_value}' for i in range(10)]
    else:
        stage_labels = [f'Segment_{i+1}' for i in range(10)]

    # ----- 规范化列名映射 -----
    COL_TIME, COL_CURRENT, COL_VOLTAGE, COL_AH, COL_STATE = 'Zeit', 'Strom', 'Spannung', 'AhAkku', 'Zustand'
    
    # 强制状态列清除空格并统一大写
    df['Zustand_Clean'] = df[COL_STATE].astype(str).str.strip().str.upper()

    # 时间轴数值化处理
    if pd.api.types.is_datetime64_any_dtype(df[COL_TIME]) or pd.api.types.is_timedelta64_any_dtype(df[COL_TIME]):
        df['Time_Seconds'] = (df[COL_TIME] - df[COL_TIME].iloc[0]).dt.total_seconds()
    else:
        df['Time_Seconds'] = pd.to_numeric(df[COL_TIME], errors='coerce') - pd.to_numeric(df[COL_TIME].iloc[0], errors='coerce')

    # 定位PAU片段边界
    print("🔍 正在通过状态切换点追踪静置(PAU)时段边界...")
    df_temp = df[['Time_Seconds', 'Zustand_Clean']].copy()
    df_temp['prev_state'] = df_temp['Zustand_Clean'].shift()
    change_points = df_temp[df_temp['Zustand_Clean'] != df_temp['prev_state']].copy().reset_index()

    valid_segments = []
    for k in range(len(change_points)):
        if (change_points.loc[k, 'prev_state'] == 'CHA' and 
            change_points.loc[k, 'Zustand_Clean'] == 'PAU'):
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

    # 取前10个片段
    num_to_extract = min(10, total_found)
    selected_indices = list(range(num_to_extract))
    print(f"📌 将提取前 {num_to_extract} 个PAU片段进行分析和参数辨识")

    # =====================================================================
    # 7. 参数辨识与可视化
    # =====================================================================
    parameter_reports = []
    all_identification_results = []

    for rank, idx in enumerate(selected_indices):
        start_idx, end_idx = valid_segments[idx]
        
        # 1. 提取静置段数据
        seg_df = df.loc[start_idx:end_idx].copy()
        seg_df['Relative_Time_s'] = seg_df['Time_Seconds'] - seg_df['Time_Seconds'].iloc[0]
        t_data = seg_df['Relative_Time_s'].values
        V_data = seg_df[COL_VOLTAGE].values
        
        # 2. 核心修正：准确提取静置前一刻（脉冲结束前）的实际电流
        # 寻找 start_idx 之前的非零电流点
        try:
            I_prev = df.loc[start_idx - 1, COL_CURRENT]
            # 如果恰好前一个点是0，向前多搜寻几个点取平均
            if abs(I_prev) < 0.1:
                I_prev = df.loc[start_idx-5:start_idx-1, COL_CURRENT].mean()
        except Exception:
            I_prev = -1.4082  # 备用默认值（注意符号，你的图里充电一般为负数）

        print(f"   静置前一刻的基准电流 I_prev = {I_prev:.4f} A")
        
        try:
            # 3. 调用新的自适应拟合
            popt, pcov = identify_parameters_adaptive(t_data, V_data)
            OCV, A1, tau1, A2, tau2 = popt
            
            # 4. 根据物理关系反推 R 和 C (电阻恒为正)
            if abs(I_prev) > 0.01:
                R1 = abs(A1 / I_prev)
                R2 = abs(A2 / I_prev)
            else:
                R1, R2 = 0.0, 0.0
                
            C1 = tau1 / R1 if R1 > 0 else 0
            C2 = tau2 / R2 if R2 > 0 else 0
            
            # 5. 计算拟合优度
            V_fitted = relaxation_model_pure(t_data, OCV, A1, tau1, A2, tau2)
            residuals = V_data - V_fitted
            RMSE = np.sqrt(np.mean(residuals**2))
            R_squared = 1 - np.sum(residuals**2) / np.sum((V_data - np.mean(V_data))**2)
            
            # (后续的绘图代码里，记得把 relaxation_model_for_fitting 改为 relaxation_model_pure 即可)

            
            # 参数不确定性（从协方差矩阵对角线的平方根获得）
            perr = np.sqrt(np.diag(pcov))
            
            params_dict = {
                "片段编号": rank + 1,
                "OCV (V)": round(OCV, 6),
                "OCV_std (V)": round(perr[0], 6),
                "R1 (Ohm)": round(R1, 6),
                "R1_std (Ohm)": round(perr[1], 6),
                "C1 (F)": round(C1, 2),
                "tau1 (s)": round(tau1, 2),
                "tau1_std (s)": round(perr[2], 2),
                "R2 (Ohm)": round(R2, 6),
                "R2_std (Ohm)": round(perr[3], 6),
                "C2 (F)": round(C2, 2),
                "tau2 (s)": round(tau2, 2),
                "tau2_std (s)": round(perr[4], 2),
                "RMSE (V)": round(RMSE, 6),
                "R²": round(R_squared, 6)
            }
            parameter_reports.append(params_dict)
            
            # 保存详细结果
            all_identification_results.append({
                'segment': rank + 1,
                't_data': t_data,
                'V_data': V_data,
                'V_fitted': V_fitted,
                'residuals': residuals,
                'params': popt,
                'RMSE': RMSE,
                'R_squared': R_squared
            })
            
            print(f"   ✅ 参数辨识成功!")
            print(f"   OCV = {OCV:.6f} ± {perr[0]:.6f} V")
            print(f"   R1 = {R1:.6f} ± {perr[1]:.6f} Ohm, C1 = {C1:.2f} F, τ1 = {tau1:.2f} s")
            print(f"   R2 = {R2:.6f} ± {perr[3]:.6f} Ohm, C2 = {C2:.2f} F, τ2 = {tau2:.2f} s")
            print(f"   RMSE = {RMSE:.6f} V, R² = {R_squared:.6f}")
            
        except Exception as e:
            print(f"   ❌ 参数辨识失败: {str(e)}")
            continue
        
        # ----- 导出CSV数据 -----
        export_df = seg_df.drop(columns=['Zustand_Clean', 'Time_Seconds', 'Relative_Time_s'], errors='ignore')
        csv_name = generate_output_filename(
            base_name="pau_segment",
            rank=f"{rank+1:02d}",
            label=stage_labels[rank],
            soc=soc_value,
            dod=dod_value,
            extension="csv"
        )
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        export_df.to_csv(csv_path, index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL, encoding='utf-8-sig')
        print(f"💾 [数据导出] PAU片段 {rank+1} -> {csv_path}")

        # ----- 绘制拟合结果 -----
        fig = plt.figure(figsize=(16, 12))
        
        # 构建标题
        if soc_value and dod_value:
            title_suffix = f"SOC={soc_value}%, DOD={dod_value}%"
        else:
            title_suffix = ""
        
        # 子图1: 电压拟合结果
        ax1 = plt.subplot(3, 2, (1, 2))
        ax1.plot(t_data, V_data, 'b.', markersize=2, alpha=0.5, label='Measured Voltage')
        ax1.plot(t_data, V_fitted, 'r-', linewidth=2, label='Fitted Model')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Voltage (V)', fontsize=12)
        ax1.set_title(f'PAU Segment {rank+1} - Voltage Relaxation Fitting {title_suffix}', 
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 添加参数标注
        textstr = f'OCV = {OCV:.4f} V\n'
        textstr += f'R₁ = {R1*1000:.3f} mΩ, C₁ = {C1:.1f} F\n'
        textstr += f'R₂ = {R2*1000:.3f} mΩ, C₂ = {C2:.1f} F\n'
        textstr += f'τ₁ = {tau1:.1f} s, τ₂ = {tau2:.1f} s\n'
        textstr += f'RMSE = {RMSE*1000:.3f} mV, R² = {R_squared:.6f}'
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 子图2: 残差分析
        ax2 = plt.subplot(3, 2, 3)
        ax2.plot(t_data, residuals * 1000, 'g.', markersize=2, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (seconds)', fontsize=10)
        ax2.set_ylabel('Residual (mV)', fontsize=10)
        ax2.set_title('Residuals Analysis', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 子图3: 残差直方图
        ax3 = plt.subplot(3, 2, 4)
        ax3.hist(residuals * 1000, bins=50, edgecolor='black', alpha=0.7, density=True)
        ax3.set_xlabel('Residual (mV)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Residual Distribution', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # 子图4: 对数时间尺度的电压
        ax4 = plt.subplot(3, 2, 5)
        log_t = np.log10(t_data[1:] + 1e-6)  # 避免log(0)
        ax4.semilogx(t_data[1:], V_data[1:], 'b.', markersize=2, alpha=0.5, label='Measured')
        ax4.semilogx(t_data[1:], V_fitted[1:], 'r-', linewidth=2, label='Fitted')
        ax4.set_xlabel('Time (seconds) - Log Scale', fontsize=10)
        ax4.set_ylabel('Voltage (V)', fontsize=10)
        ax4.set_title('Voltage Relaxation (Log Time Scale)', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.5)
        
        # 子图5: 电流曲线
        ax5 = plt.subplot(3, 2, 6)
        ax5.plot(seg_df['Relative_Time_s'], seg_df[COL_CURRENT], 'm-', linewidth=1)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Time (seconds)', fontsize=10)
        ax5.set_ylabel('Current (A)', fontsize=10)
        ax5.set_title('Current During Pause', fontsize=12)
        ax5.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图像
        plot_name = generate_output_filename(
            base_name="pau_fitting_segment",
            rank=f"{rank+1:02d}",
            label=stage_labels[rank],
            soc=soc_value,
            dod=dod_value,
            extension="png"
        )
        plot_path = os.path.join(OUTPUT_DIR, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"🖼️  [拟合图像已保存] PAU片段 {rank+1} -> {plot_path}")

    # =====================================================================
    # 8. 输出汇总报告
    # =====================================================================
    if parameter_reports:
        summary_df = pd.DataFrame(parameter_reports)
        print("\n" + "="*120)
        print(f"📊 前 {len(parameter_reports)} 组PAU片段参数辨识汇总报告")
        print("="*120)
        print(summary_df.to_string(index=False))
        print("="*120)
        
        # 保存参数汇总报告
        if soc_value and dod_value:
            param_csv = os.path.join(OUTPUT_DIR, f"parameter_identification_SOC{soc_value}_DOD{dod_value}.csv")
        else:
            param_csv = os.path.join(OUTPUT_DIR, "parameter_identification.csv")
        summary_df.to_csv(param_csv, index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL, encoding='utf-8-sig')
        print(f"\n📋 参数辨识报告已保存: {param_csv}")
        
        # 打印统计信息
        print("\n📈 参数统计信息:")
        print("-" * 50)
        for col in ['OCV (V)', 'R1 (Ohm)', 'R2 (Ohm)', 'tau1 (s)', 'tau2 (s)', 'RMSE (V)', 'R²']:
            if col in summary_df.columns:
                mean_val = summary_df[col].mean()
                std_val = summary_df[col].std()
                print(f"  {col:15s}: {mean_val:10.6f} ± {std_val:10.6f}")
    else:
        print("\n❌ 没有成功辨识任何参数")

if __name__ == "__main__":
    main()