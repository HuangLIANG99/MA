import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
import os

# ==============================================================================
# 1. 从文件名自动提取参数
# ==============================================================================
def extract_params_from_filename(filepath):
    """
    从文件名中提取 SOH、SOC、DOD
    例如: "18650VTC6_003_dch_96.1SOH.csv" -> SOH=96.1
          "discharge_segment_3_SOC70_DOD60.csv" -> SOC=70, DOD=60
    """
    filename = os.path.basename(filepath)
    
    # 提取 SOH (支持 96.1SOH 或 96SOH)
    soh_match = re.search(r'(\d+\.?\d*)SOH', filename, re.IGNORECASE)
    soh = float(soh_match.group(1)) if soh_match else None
    
    # 提取 SOC
    soc_match = re.search(r'SOC(\d+)', filename, re.IGNORECASE)
    soc = float(soc_match.group(1)) if soc_match else None
    
    # 提取 DOD
    dod_match = re.search(r'DOD(\d+)', filename, re.IGNORECASE)
    dod = float(dod_match.group(1)) if dod_match else None
    
    return soh, soc, dod

# ==============================================================================
# 2. 配置文件（自动识别参数）
# ==============================================================================
ocv_file_path = r"H:\MA\output_dch_ocv\18650VTC6_003_dch_96.1SOH.csv"
rc_file_path = r"H:\MA\2RC.csv"
segment_file_path = r"H:\MA\output\discharge_segment_2_SOC70_DOD60.csv"

# 从文件名自动提取参数
soh_from_file, soc_from_file, dod_from_file = extract_params_from_filename(segment_file_path)

# 如果提取失败，使用默认值
if soh_from_file is None:
    soh_from_file = 96.1  # 从 OCV 文件名提取
    soh_from_ocv, _, _ = extract_params_from_filename(ocv_file_path)
    if soh_from_ocv is not None:
        soh_from_file = soh_from_ocv

if soc_from_file is None:
    soc_from_file = 70.0  # 默认值
if dod_from_file is None:
    dod_from_file = 60.0  # 默认值

print(f"\n========== 从文件名自动识别 ==========")
print(f"📄 OCV 文件: {os.path.basename(ocv_file_path)}")
print(f"📄 片段文件: {os.path.basename(segment_file_path)}")
print(f"📊 识别结果:")
print(f"   SOH = {soh_from_file:.1f}%")
print(f"   SOC (平均) = {soc_from_file:.0f}%")
print(f"   DOD = {dod_from_file:.0f}%")
print("======================================\n")

# ==============================================================================
# 3. 基本参数设置
# ==============================================================================
Q_initial = 3.0          # 标称容量 (Ah)
V_cut = 2.5              # 截止电压 (V)

# 从识别结果计算
soc_avg = soc_from_file / 100.0           # 平均 SOC
dod = dod_from_file / 100.0               # DOD
soc_start_seg = soc_avg + dod / 2.0       # 起点 SOC = 平均 + DOD/2
soc_end_seg = soc_avg - dod / 2.0         # 终点 SOC = 平均 - DOD/2
Q_est_current = Q_initial * (soh_from_file / 100.0)  # 当前容量 = 标称 × SOH

# ==============================================================================
# 4. 导入数据
# ==============================================================================
print(">>> 正在加载数据文件...")
df_ocv = pd.read_csv(ocv_file_path, sep=';', decimal=',')
df_rc = pd.read_csv(rc_file_path, sep=';', decimal=',')
df_seg = pd.read_csv(segment_file_path, sep=';', decimal=',')

# OCV 插值
ocv_soc = df_ocv.iloc[:, 1].values
if ocv_soc.max() > 1.1:
    ocv_soc = ocv_soc / 100.0
ocv_func = interp1d(ocv_soc, df_ocv.iloc[:, 2].values, kind='linear', fill_value="extrapolate")

# RC 参数插值
rc_soc = df_rc.iloc[:, 0].values
if rc_soc.max() > 1.1:
    rc_soc = rc_soc / 100.0

r0_func = interp1d(rc_soc, df_rc.iloc[:, 1].values, kind='linear', fill_value="extrapolate")
r1_func = interp1d(rc_soc, df_rc.iloc[:, 2].values, kind='linear', fill_value="extrapolate")
c1_func = interp1d(rc_soc, df_rc.iloc[:, 3].values, kind='linear', fill_value="extrapolate")
r2_func = interp1d(rc_soc, df_rc.iloc[:, 4].values, kind='linear', fill_value="extrapolate")
c2_func = interp1d(rc_soc, df_rc.iloc[:, 5].values, kind='linear', fill_value="extrapolate")

# 片段数据
parsed_time = pd.to_datetime(df_seg.iloc[:, 0], errors='coerce').ffill().bfill()
seg_time = (parsed_time - parsed_time.iloc[0]).dt.total_seconds().values
seg_voltage = df_seg.iloc[:, 1].values
seg_current = df_seg.iloc[:, 2].values
if seg_current.mean() < 0:
    seg_current = np.abs(seg_current)

# 验证片段消耗容量是否与 DOD 匹配
dt_seg = np.mean(np.diff(seg_time))
total_seg_ah = np.sum(seg_current * dt_seg) / 3600.0
dod_from_data = total_seg_ah / Q_est_current

print(f"\n========== 片段工况验证 ==========")
print(f"起点 SOC: {soc_start_seg * 100:.1f}%")
print(f"终点 SOC: {soc_end_seg * 100:.1f}%")
print(f"平均 SOC: {soc_avg * 100:.1f}%")
print(f"DOD (理论): {dod * 100:.1f}%")
print(f"DOD (从电流积分): {dod_from_data * 100:.1f}%")
print(f"当前容量估计: {Q_est_current:.3f} Ah")
print(f"理论消耗容量: {Q_est_current * dod:.4f} Ah")
print(f"实际消耗容量: {total_seg_ah:.4f} Ah")
print("==================================\n")

# ==============================================================================
# 5. 二阶 RC 仿真器
# ==============================================================================
def simulate_2rc(time_seq, current_seq, soc_init, Q_current_ah):
    N = len(time_seq)
    Vt_pred = np.zeros(N)
    soc = soc_init
    V1, V2 = 0.0, 0.0
    
    for k in range(N):
        dt = time_seq[k] - time_seq[k-1] if k > 0 else (time_seq[1] - time_seq[0] if N > 1 else 1.0)
        dt = max(dt, 0.0)
        I = current_seq[k]
        
        R0 = max(float(r0_func(soc)), 1e-4)
        R1 = max(float(r1_func(soc)), 1e-4)
        C1 = max(float(c1_func(soc)), 1.0)
        R2 = max(float(r2_func(soc)), 1e-4)
        C2 = max(float(c2_func(soc)), 1.0)
        
        Voc = ocv_func(soc)
        Vt_pred[k] = Voc - I * R0 - V1 - V2
        
        V1 = np.exp(-dt / (R1 * C1)) * V1 + R1 * (1 - np.exp(-dt / (R1 * C1))) * I
        V2 = np.exp(-dt / (R2 * C2)) * V2 + R2 * (1 - np.exp(-dt / (R2 * C2))) * I
        
        soc = soc - (I * dt) / (Q_current_ah * 3600.0)
        soc = np.clip(soc, 0.0, 1.0)
        
    return Vt_pred

# ==============================================================================
# 6. 片段仿真
# ==============================================================================
seg_vt_sim = simulate_2rc(seg_time, seg_current, soc_start_seg, Q_est_current)
rmse_seg = np.sqrt(np.mean((seg_voltage - seg_vt_sim) ** 2))
print(f"✅ 片段仿真 RMSE: {rmse_seg * 1000:.2f} mV\n")

# ==============================================================================
# 7. 全 SOC 外推
# ==============================================================================
I_full_sim = np.mean(seg_current)
sim_dt = 1.0

soc_full = 1.0
V1_f, V2_f = 0.0, 0.0
total_amp_sec = 0.0
max_steps = 200000

history_vt = []
history_soc = []

for step in range(max_steps):
    R0_f = max(float(r0_func(soc_full)), 1e-4)
    R1_f = max(float(r1_func(soc_full)), 1e-4)
    C1_f = max(float(c1_func(soc_full)), 1.0)
    R2_f = max(float(r2_func(soc_full)), 1e-4)
    C2_f = max(float(c2_func(soc_full)), 1.0)
    
    Voc_f = ocv_func(soc_full)
    Vt_f = Voc_f - I_full_sim * R0_f - V1_f - V2_f
    
    history_vt.append(Vt_f)
    history_soc.append(soc_full)
    
    if Vt_f <= V_cut:
        break
    
    total_amp_sec += I_full_sim * sim_dt
    soc_full = 1.0 - (total_amp_sec / (Q_est_current * 3600.0))
    soc_full = np.clip(soc_full, 0.0, 1.0)
    
    V1_f = np.exp(-sim_dt / (R1_f * C1_f)) * V1_f + R1_f * (1 - np.exp(-sim_dt / (R1_f * C1_f))) * I_full_sim
    V2_f = np.exp(-sim_dt / (R2_f * C2_f)) * V2_f + R2_f * (1 - np.exp(-sim_dt / (R2_f * C2_f))) * I_full_sim

Q_sim_discharged = total_amp_sec / 3600.0
soh_calculated = (Q_sim_discharged / Q_initial) * 100.0

print("========== SOH 估算报告 ==========")
print(f"🔋 标称容量: {Q_initial:.3f} Ah")
print(f"📊 文献 SOH: {soh_from_file:.1f}%")
print(f"📊 当前容量估计: {Q_est_current:.3f} Ah")
print(f"📊 外推放电容量: {Q_sim_discharged:.4f} Ah")
print(f"📊 计算 SOH: {soh_calculated:.2f}%")
print(f"📊 偏差: {soh_calculated - soh_from_file:+.2f}%")
print("==================================\n")

# ==============================================================================
# 8. 绘图
# ==============================================================================
plt.figure(figsize=(14, 5))

# 左图：片段仿真
plt.subplot(1, 2, 1)
plt.plot(seg_time - seg_time[0], seg_voltage, 'k-', label='Measured', linewidth=2)
plt.plot(seg_time - seg_time[0], seg_vt_sim, 'r--', label=f'2RC Model (RMSE={rmse_seg*1000:.1f}mV)', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Segment: {soc_start_seg*100:.0f}% → {soc_end_seg*100:.0f}% (Avg {soc_avg*100:.0f}%, DOD {dod*100:.0f}%)')
plt.grid(True)
plt.legend()

# 右图：全 SOC 放电曲线
plt.subplot(1, 2, 2)
plt.plot(np.array(history_soc) * 100, history_vt, 'b-', label='Full Discharge', linewidth=2)
plt.axhline(y=V_cut, color='r', linestyle='--', label=f'Cut-off {V_cut}V')
plt.axvline(x=soc_start_seg * 100, color='g', linestyle=':', label=f'Start {soc_start_seg*100:.0f}%')
plt.axvline(x=soc_end_seg * 100, color='orange', linestyle=':', label=f'End {soc_end_seg*100:.0f}%')
plt.xlim(105, -5)
plt.xlabel('SOC (%)')
plt.ylabel('Voltage (V)')
plt.title(f'SOH = {soh_calculated:.2f}% (Reference: {soh_from_file:.1f}%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# 9. 导出结果报告
# ==============================================================================
report = {
    '参数': ['标称容量', '文献 SOH', '当前容量估计', '片段平均 SOC', '片段 DOD', 
             '片段起点 SOC', '片段终点 SOC', '外推放电容量', '计算 SOH', 'RMSE'],
    '数值': [f'{Q_initial:.3f} Ah', f'{soh_from_file:.1f}%', f'{Q_est_current:.3f} Ah',
             f'{soc_avg*100:.1f}%', f'{dod*100:.1f}%',
             f'{soc_start_seg*100:.1f}%', f'{soc_end_seg*100:.1f}%',
             f'{Q_sim_discharged:.4f} Ah', f'{soh_calculated:.2f}%', f'{rmse_seg*1000:.2f} mV']
}
report_df = pd.DataFrame(report)
print("\n========== 结果汇总 ==========")
print(report_df.to_string(index=False))
print("================================")