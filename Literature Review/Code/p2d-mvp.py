import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.integrate import solve_ivp

# =========================
# 1. 常数
# =========================

F = 96485.3329       # 法拉第常数, C/mol
R = 8.314462618      # 气体常数, J/(mol K)
T = 298.15           # 温度, K

# =========================
# 2. 电池参数
# =========================

@dataclass
class CellParams:
    # 几何参数
    A: float = 1.0             # 电极面积, m^2
    L_n: float = 85e-6         # 负极厚度, m
    L_s: float = 25e-6         # 隔膜厚度, m
    L_p: float = 75e-6         # 正极厚度, m

    # 孔隙率
    eps_n: float = 0.30
    eps_s: float = 0.50
    eps_p: float = 0.30

    # 颗粒半径
    R_n: float = 5e-6
    R_p: float = 5e-6

    # 扩散系数
    Ds_n: float = 3e-14        # 负极固相扩散系数, m^2/s
    Ds_p: float = 1e-14        # 正极固相扩散系数, m^2/s
    De: float = 2e-10          # 液相扩散系数, m^2/s

    # 初始液相浓度
    c_e0: float = 1000.0       # mol/m^3

    # 固相最大浓度
    c_smax_n: float = 31000.0
    c_smax_p: float = 51000.0

    # 初始 SOC / stoichiometry
    theta_n0: float = 0.85
    theta_p0: float = 0.45

    # Butler-Volmer 反应速率常数，先用占位值
    k_n: float = 2e-11
    k_p: float = 2e-11

    # 膜阻抗 / 欧姆阻抗，占位值
    R_f: float = 0.001         # Ohm m^2

    # 锂离子迁移数
    t_plus: float = 0.38

    # x 方向网格数
    N: int = 60

    # =========================
# 3. OCP 函数
# =========================
# 注意：这里先用简化经验函数。
# 后面你要替换成 NCA 和 LFP 的真实 OCP 曲线。

def U_n(theta):
    """
    石墨负极 OCP，占位函数。
    theta: 负极表面 stoichiometry
    """
    theta = np.clip(theta, 1e-4, 0.9999)
    return 0.1 + 0.8 * np.exp(-10 * theta) + 0.05 * np.tanh((0.5 - theta) / 0.08)


def U_p_nca(theta):
    """
    NCA 正极 OCP，占位函数。
    后面做 NCA -> LFP 迁移时，这里要替换成 LFP OCP。
    """
    theta = np.clip(theta, 1e-4, 0.9999)
    return 4.25 - 0.9 * theta + 0.1 * np.tanh((0.5 - theta) / 0.08)

# =========================
# 4. 建立 x 方向网格
# =========================

def make_grid(p: CellParams):
    """
    将电池厚度方向分成：
    负极 | 隔膜 | 正极
    """
    L_total = p.L_n + p.L_s + p.L_p
    x = np.linspace(0, L_total, p.N)
    dx = x[1] - x[0]

    neg = x <= p.L_n
    sep = (x > p.L_n) & (x < p.L_n + p.L_s)
    pos = x >= p.L_n + p.L_s

    eps = np.where(neg, p.eps_n, np.where(sep, p.eps_s, p.eps_p))

    return x, dx, neg, sep, pos, eps


# =========================
# 5. 电流输入
# =========================

def current_profile(t, C_rate=1.0, capacity_Ah=3.0):
    """
    简单恒流放电。
    正电流表示放电。
    例如：
    1C, 3Ah 电池 -> I = 3A
    0.5C, 3Ah 电池 -> I = 1.5A
    3C, 3Ah 电池 -> I = 9A
    """
    return C_rate * capacity_Ah


# =========================
# 6. 简化 P2D 状态方程
# =========================

def rhs(t, y, p: CellParams, C_rate=1.0, capacity_Ah=3.0):
    """
    y 包含：
    y[0:N]     = 液相浓度 c_e(x,t)
    y[N]       = 负极固相平均浓度 cbar_n(t)
    y[N + 1]   = 正极固相平均浓度 cbar_p(t)

    这一版先保留液相浓度 PDE，
    固相颗粒半径方向用平均浓度 ODE 代替。
    """
    x, dx, neg, sep, pos, eps = make_grid(p)
    N = p.N

    c_e = y[:N]
    cbar_n = y[N]
    cbar_p = y[N + 1]

    I = current_profile(t, C_rate, capacity_Ah)

    # 比界面面积 a_s = 3 epsilon_s / R_p
    a_n = 3 * p.eps_n / p.R_n
    a_p = 3 * p.eps_p / p.R_p

    # 摩尔通量 j, mol/(m^2 s)
    # 负极放电时锂从固相出来，所以 j_n > 0
    # 正极放电时锂进入固相，所以 j_p < 0
    j_n = I / (F * p.A * a_n * p.L_n)
    j_p = -I / (F * p.A * a_p * p.L_p)

    # 液相扩散项：d2c/dx2
    # 零通量边界：边界处用镜像点近似
    c_left = np.r_[c_e[1], c_e[:-1]]
    c_right = np.r_[c_e[1:], c_e[-2]]
    d2c_dx2 = (c_right - 2 * c_e + c_left) / dx**2

    # 液相源项
    source = np.zeros_like(c_e)

    # 负极：锂进入电解液，液相浓度上升
    source[neg] = (1 - p.t_plus) * a_n * j_n / p.eps_n

    # 正极：锂离子被消耗，液相浓度下降
    source[pos] = (1 - p.t_plus) * a_p * j_p / p.eps_p

    dc_e_dt = p.De * d2c_dx2 + source

    # 固相平均浓度 ODE
    dcbar_n_dt = -3 * j_n / p.R_n
    dcbar_p_dt = -3 * j_p / p.R_p

    return np.r_[dc_e_dt, dcbar_n_dt, dcbar_p_dt]


# =========================
# 7. 由状态计算端电压
# =========================

def voltage_from_state(t, y, p: CellParams, C_rate=1.0, capacity_Ah=3.0):
    """
    用 OCP + Butler-Volmer 过电位 + 简单欧姆压降计算端电压。
    """
    c_e = y[:p.N]
    cbar_n = y[p.N]
    cbar_p = y[p.N + 1]

    x, dx, neg, sep, pos, eps = make_grid(p)

    I = current_profile(t, C_rate, capacity_Ah)

    a_n = 3 * p.eps_n / p.R_n
    a_p = 3 * p.eps_p / p.R_p

    j_n = I / (F * p.A * a_n * p.L_n)
    j_p = -I / (F * p.A * a_p * p.L_p)

    # 抛物线近似：由平均浓度得到表面浓度
    c_surf_n = cbar_n - p.R_n * j_n / (5 * p.Ds_n)
    c_surf_p = cbar_p - p.R_p * j_p / (5 * p.Ds_p)

    # 防止数值越界
    c_surf_n = np.clip(c_surf_n, 1.0, p.c_smax_n - 1.0)
    c_surf_p = np.clip(c_surf_p, 1.0, p.c_smax_p - 1.0)

    theta_n = c_surf_n / p.c_smax_n
    theta_p = c_surf_p / p.c_smax_p

    # 分别取负极区域和正极区域的平均液相浓度
    ce_n = max(np.mean(c_e[neg]), 1.0)
    ce_p = max(np.mean(c_e[pos]), 1.0)

    # 交换电流密度 i0，单位近似处理
    i0_n = F * p.k_n * np.sqrt(ce_n) * np.sqrt(c_surf_n) * np.sqrt(p.c_smax_n - c_surf_n)
    i0_p = F * p.k_p * np.sqrt(ce_p) * np.sqrt(c_surf_p) * np.sqrt(p.c_smax_p - c_surf_p)

    # 界面电流密度，A/m^2
    i_n = F * j_n
    i_p = F * j_p

    # Butler-Volmer 反解过电位
    eta_n = (2 * R * T / F) * np.arcsinh(i_n / (2 * i0_n + 1e-12))
    eta_p = (2 * R * T / F) * np.arcsinh(i_p / (2 * i0_p + 1e-12))

    # 端电压
    V = U_p_nca(theta_p) + eta_p - U_n(theta_n) - eta_n - I * p.R_f / p.A

    return V


# =========================
# 8. 仿真函数
# =========================

def simulate(C_rate=1.0, capacity_Ah=3.0, p=None):
    if p is None:
        p = CellParams()

    # 初始状态
    c_e_init = np.full(p.N, p.c_e0)
    cbar_n_init = p.theta_n0 * p.c_smax_n
    cbar_p_init = p.theta_p0 * p.c_smax_p

    y0 = np.r_[c_e_init, cbar_n_init, cbar_p_init]

    # 放电时间：大约 1 / C_rate 小时
    t_end = 1.05 / C_rate * 3600
    t_eval = np.linspace(0, t_end, 400)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p, C_rate, capacity_Ah),
        t_span=(0, t_end),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )

    V = np.array([
        voltage_from_state(t, sol.y[:, i], p, C_rate, capacity_Ah)
        for i, t in enumerate(sol.t)
    ])

    Ah = sol.t * current_profile(0, C_rate, capacity_Ah) / 3600

    return Ah, V, sol


# =========================
# 9. 主程序：比较 0.5C 和 3C
# =========================

if __name__ == "__main__":
    p = CellParams()

    Ah_05, V_05, sol_05 = simulate(C_rate=0.5, capacity_Ah=3.0, p=p)
    Ah_3, V_3, sol_3 = simulate(C_rate=3.0, capacity_Ah=3.0, p=p)

    plt.figure(figsize=(7, 5))
    plt.plot(Ah_05, V_05, label="0.5C discharge")
    plt.plot(Ah_3, V_3, label="3C discharge")
    plt.xlabel("Discharged capacity / Ah")
    plt.ylabel("Voltage / V")
    plt.title("Simplified P2D MVP voltage simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
