import os
import glob
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix


# ============================================================
# 0. 配置区
# ============================================================

# True: 从 MinIO/S3 读取
# False: 从本地文件夹读取
USE_S3 = True

# 可以是：
# 1. 一个具体 parquet 文件
# 2. 一个文件夹 / prefix
DATA_PATH = (
    "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_006"
)

# 只读取前几个 parquet 文件，用来节省时间
MAX_FILES = 3

# 文件名过滤。None 表示不过滤。
# 例如可以设成 "finished.parquet" 或 "Ageing"
FILE_NAME_CONTAINS = None

FILE_NAME_EXCLUDES = ("Init", "pulse")  # 大小写不敏感，同时排除这两类文件

# 电流绝对值小于该阈值时标记为静置
MIN_CURRENT_ABS = 0.05

# 如果自动判断电流方向失败：
# -1 表示原始 Strom 中放电为负
# +1 表示原始 Strom 中放电为正
DEFAULT_RAW_DISCHARGE_SIGN = -1

# 手动指定电流方向：
# None 表示自动判断
# -1 表示原始 Strom 中放电为负
# +1 表示原始 Strom 中放电为正
RAW_DISCHARGE_SIGN = None

# OCP / OCV 模式
# "fullcell_analytic": 不需要 OCP 文件，使用内置整电池近似 OCV
# "electrode_placeholder": 使用占位正负极 OCP
OCP_MODE = "fullcell_analytic"

# ODE 输出点数，越大越慢
MAX_SOLVE_POINTS = 1200

# ============================================================
# 完整文件仿真的数值稳定参数
# ============================================================

# 原始 parquet 中的电流单位固定为 A，不进行单位判断或转换。

# 允许的最大绝对电流，仅用于识别明显异常数据。
MAX_REASONABLE_CURRENT_A = 100.0

# 每个连续数值积分窗口的最大时长。
# 这只是求解器内部的连续检查点，不会把 parquet 结果拆成多个 segment。
SOLVER_WINDOW_S = 6.0 * 3600.0

# 求解器最大内部步长
SOLVER_MAX_STEP_S = 300.0

# 更适合浓度状态量级的容差
SOLVER_RTOL = 1e-5
SOLVER_ATOL_ELECTROLYTE = 1e-3
SOLVER_ATOL_SOLID = 1e-2

# BDF 出现溢出警告或失败时，自动改用 Radau 重算当前连续窗口
SOLVER_PRIMARY_METHOD = "BDF"
SOLVER_FALLBACK_METHOD = "Radau"

# 浓度物理边界与越界恢复时间常数
ELECTROLYTE_CONC_MIN = 1.0
ELECTROLYTE_CONC_MAX = 5000.0
SOLID_CONC_MARGIN = 1.0
BOUND_RESTORE_TAU_S = 5.0

# fullcell_analytic 模式下，按文件第一个实测电压估计初始 SOC
ESTIMATE_INITIAL_SOC_FROM_FIRST_VOLTAGE = True
INITIAL_SOC_MIN = 0.02
INITIAL_SOC_MAX = 0.98

OUTPUT_DIR = "p2d_from_parquet_output"


# ============================================================
# 1. 常数
# ============================================================

F = 96485.3329
R = 8.314462618
T_REF = 298.15


# ============================================================
# 2. S3 配置
# ============================================================

def get_s3_storage_options():
    """
    从环境变量读取 MinIO/S3 密钥。

    Windows PowerShell:
    $env:MINIO_ACCESS_KEY="你的access_key"
    $env:MINIO_SECRET_KEY="你的secret_key"

    Linux / macOS:
    export MINIO_ACCESS_KEY="你的access_key"
    export MINIO_SECRET_KEY="你的secret_key"
    """

    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")

    if key is None or secret is None:
        raise ValueError(
            "没有找到 MINIO_ACCESS_KEY 或 MINIO_SECRET_KEY。\n"
            "请先设置环境变量，不要把密钥写死进代码。"
        )

    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {
            "endpoint_url": "https://iseadocker.isea.rwth-aachen.de:9000",
            "region_name": "us-east-1",
        },
        "config_kwargs": {
            "s3": {"addressing_style": "path"},
            "signature_version": "s3v4",
        },
    }


def to_s3_uri(path):
    if path.startswith("s3://"):
        return path
    return "s3://" + path.lstrip("/")


# ============================================================
# 3. 批量列出 parquet 文件
# ============================================================

def list_parquet_files(
    data_path,
    use_s3=True,
    max_files=None,
    file_name_contains=None,
    file_name_excludes=None
):
    """
    返回 parquet 文件路径列表。
    """

    if use_s3:
        import s3fs

        storage_options = get_s3_storage_options()
        fs = s3fs.S3FileSystem(**storage_options)

        s3_uri = to_s3_uri(data_path)
        no_proto = s3_uri.replace("s3://", "", 1)

        if no_proto.endswith(".parquet"):
            files = [no_proto]
        else:
            pattern = no_proto.rstrip("/") + "/**/*.parquet"
            files = fs.glob(pattern)

        files = ["s3://" + f for f in files]

    else:
        if data_path.endswith(".parquet"):
            files = [data_path]
        else:
            files = glob.glob(
                os.path.join(data_path, "**", "*.parquet"),
                recursive=True
            )

    files = sorted(files)

    if file_name_contains is not None:
        contains_lower = file_name_contains.lower()
        files = [
            f for f in files
            if contains_lower in os.path.basename(f).lower()
        ]

    # 排除文件名中包含任一指定关键词的文件，大小写不敏感
    if file_name_excludes is not None:
        if isinstance(file_name_excludes, str):
            exclude_keywords = [file_name_excludes]
        else:
            exclude_keywords = list(file_name_excludes)

        exclude_keywords = [
            str(keyword).lower()
            for keyword in exclude_keywords
            if str(keyword).strip()
        ]

        files = [
            f for f in files
            if not any(
                keyword in os.path.basename(f).lower()
                for keyword in exclude_keywords
            )
        ]

    if max_files is not None:
        files = files[:max_files]

    print("\n📁 找到 parquet 文件:")
    for i, f in enumerate(files):
        print(f"   [{i}] {f}")

    if len(files) == 0:
        raise FileNotFoundError("没有找到 parquet 文件，请检查 DATA_PATH。")

    return files


# ============================================================
# 4. 读取单个 parquet 文件
# ============================================================

def read_one_parquet(file_path, use_s3=True):
    print("\n" + "=" * 80)
    print(f"📄 正在读取: {file_path}")

    if use_s3:
        storage_options = get_s3_storage_options()
        df = pd.read_parquet(
            file_path,
            storage_options=storage_options
        )
    else:
        df = pd.read_parquet(file_path)

    print(f"✅ 读取成功: 行数={len(df)}, 列数={len(df.columns)}")
    print(f"   原始列名: {df.columns.tolist()}")

    return df


# ============================================================
# 5. 数据预处理
# ============================================================

def rename_columns(df):
    df = df.copy()

    column_map = {
        "Zeit": "time",
        "Spannung": "voltage",
        "Strom": "current",
        "T1": "temperature",
        "Prozedur": "procedure",
        "Zustand": "state",
        "AhAkku": "capacity_ah",
        "Ahjo_Test_ID": "test_id",

        "time": "time",
        "voltage": "voltage",
        "current": "current",
        "temperature": "temperature",
        "capacity": "capacity_ah",
        "capacity_ah": "capacity_ah",
        "test_id": "test_id",
    }

    actual_map = {
        old: new for old, new in column_map.items()
        if old in df.columns
    }

    df = df.rename(columns=actual_map)

    required_cols = ["time", "voltage", "current"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"缺少必需列: {missing}\n"
            f"当前列名: {df.columns.tolist()}"
        )

    return df


def standardize_current_unit(df):
    """
    原始 parquet 中的电流单位固定为 A。

    本函数只负责将电流列转换为数值类型并检查有限值，
    不进行 A/mA 自动判断，也不进行任何倍率换算。
    """
    df = df.copy()
    df["current"] = pd.to_numeric(
        df["current"],
        errors="coerce"
    )

    valid_current = df["current"].dropna()

    if len(valid_current) == 0:
        raise ValueError("电流列没有有效数值。")

    current_values = valid_current.to_numpy(dtype=float)

    if not np.all(np.isfinite(current_values)):
        raise ValueError("电流列包含 NaN 或无穷大。")

    current_abs = np.abs(current_values)
    max_current_a = float(np.max(current_abs))
    q95_current_a = float(np.quantile(current_abs, 0.95))

    print("✅ 电流单位固定按 A 读取，不进行单位转换。")
    print(
        f"   电流绝对值95%分位数: {q95_current_a:.6f} A"
    )
    print(
        f"   最大绝对电流: {max_current_a:.6f} A"
    )

    if max_current_a > MAX_REASONABLE_CURRENT_A:
        raise ValueError(
            f"最大绝对电流为 {max_current_a:.3f} A，超过 "
            f"MAX_REASONABLE_CURRENT_A={MAX_REASONABLE_CURRENT_A:.3f} A。"
            "请检查原始数据是否存在异常值。"
        )

    return df


def convert_time_to_seconds(df):
    df = df.copy()

    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time_s"] = (
            df["time"] - df["time"].iloc[0]
        ).dt.total_seconds()
        return df

    if pd.api.types.is_timedelta64_any_dtype(df["time"]):
        df["time_s"] = df["time"].dt.total_seconds()
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
        return df

    if np.issubdtype(df["time"].dtype, np.number):
        df["time_s"] = df["time"].astype(float)
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
        return df

    time_parsed = pd.to_datetime(df["time"], errors="coerce")

    if time_parsed.notna().mean() > 0.8:
        df["time_s"] = (
            time_parsed - time_parsed.iloc[0]
        ).dt.total_seconds()
        return df

    time_delta = pd.to_timedelta(df["time"], errors="coerce")

    if time_delta.notna().mean() > 0.8:
        df["time_s"] = time_delta.dt.total_seconds()
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
        return df

    raise ValueError(f"time 列无法识别，类型为: {df['time'].dtype}")


def clean_data(df):
    df = df.copy()

    numeric_cols = ["voltage", "current"]

    if "temperature" in df.columns:
        numeric_cols.append("temperature")

    if "capacity_ah" in df.columns:
        numeric_cols.append("capacity_ah")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time_s", "voltage", "current"])

    if "test_id" in df.columns:
        df = df.sort_values(["test_id", "time_s"]).reset_index(drop=True)
    else:
        df = df.sort_values("time_s").reset_index(drop=True)

    return df


def infer_raw_discharge_sign(df, default_raw_discharge_sign=-1):
    """
    判断原始 current 中放电是正还是负。
    返回：
    +1: 原始 current 中放电为正
    -1: 原始 current 中放电为负
    """

    if "procedure" not in df.columns and "state" not in df.columns:
        print("⚠️ 没有 procedure/state，无法自动判断电流方向。")
        return default_raw_discharge_sign

    text_parts = []

    if "procedure" in df.columns:
        text_parts.append(df["procedure"].astype(str).str.lower())

    if "state" in df.columns:
        text_parts.append(df["state"].astype(str).str.lower())

    text = text_parts[0]

    for part in text_parts[1:]:
        text = text + " " + part

    discharge_keywords = [
        "entladen",
        "discharge",
        "discharging",
        "dch",
        "dc discharge",
    ]

    discharge_mask = np.zeros(len(df), dtype=bool)

    for kw in discharge_keywords:
        discharge_mask |= text.str.contains(kw, na=False).to_numpy()

    if discharge_mask.sum() > 10:
        median_i = df.loc[discharge_mask, "current"].median()

        if median_i > 0:
            print("✅ 自动判断：原始 current 中放电为正。")
            return +1
        else:
            print("✅ 自动判断：原始 current 中放电为负。")
            return -1

    print("⚠️ 未能从 procedure/state 可靠判断放电方向。")
    print(f"   使用默认 raw_discharge_sign = {default_raw_discharge_sign}")

    return default_raw_discharge_sign


def add_discharge_current(
    df,
    raw_discharge_sign=None,
    default_raw_discharge_sign=-1
):
    """
    添加 current_discharge。
    约定：
    current_discharge > 0 表示放电
    current_discharge < 0 表示充电
    """

    df = df.copy()

    if raw_discharge_sign is None:
        raw_discharge_sign = infer_raw_discharge_sign(
            df,
            default_raw_discharge_sign=default_raw_discharge_sign
        )
    else:
        print(f"✅ 手动设置 raw_discharge_sign = {raw_discharge_sign}")

    if raw_discharge_sign == -1:
        df["current_discharge"] = -df["current"]
    elif raw_discharge_sign == +1:
        df["current_discharge"] = df["current"]
    else:
        raise ValueError("raw_discharge_sign 只能是 -1、+1 或 None。")

    print(
        f"   current_discharge 平均值: "
        f"{df['current_discharge'].mean():.6f} A"
    )

    return df


def add_operation_label(df, min_current_abs=0.05):
    df = df.copy()

    I = df["current_discharge"].to_numpy(dtype=float)

    df["operation"] = np.select(
        [
            I > min_current_abs,
            I < -min_current_abs,
        ],
        [
            "discharge",
            "charge",
        ],
        default="rest",
    )

    print("📌 工况统计:")
    print(df["operation"].value_counts())

    return df


def preprocess_dataframe(df):
    df = rename_columns(df)
    df = standardize_current_unit(df)
    df = convert_time_to_seconds(df)
    df = clean_data(df)
    df = add_discharge_current(
        df,
        raw_discharge_sign=RAW_DISCHARGE_SIGN,
        default_raw_discharge_sign=DEFAULT_RAW_DISCHARGE_SIGN
    )
    df = add_operation_label(
        df,
        min_current_abs=MIN_CURRENT_ABS
    )
    return df


# ============================================================
# 7. 实测电流输入 I(t)
# ============================================================

@dataclass
class CurrentInput:
    time_s: np.ndarray
    current_a: np.ndarray

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.current_a = np.asarray(self.current_a, dtype=float)

        order = np.argsort(self.time_s)
        self.time_s = self.time_s[order]
        self.current_a = self.current_a[order]

        unique_t, unique_idx = np.unique(
            self.time_s,
            return_index=True
        )

        self.time_s = unique_t
        self.current_a = self.current_a[unique_idx]

    def __call__(self, t):
        return float(
            np.interp(
                t,
                self.time_s,
                self.current_a,
                left=self.current_a[0],
                right=self.current_a[-1],
            )
        )


# ============================================================
# 8. 简化 P2D 参数
# ============================================================

@dataclass
class CellParams:
    # 几何参数
    A: float = 0.20
    L_n: float = 85e-6
    L_s: float = 25e-6
    L_p: float = 75e-6

    # 孔隙率
    eps_n: float = 0.30
    eps_s: float = 0.50
    eps_p: float = 0.30

    # 颗粒半径
    R_n: float = 5e-6
    R_p: float = 5e-6

    # 扩散系数
    Ds_n: float = 3e-14
    Ds_p: float = 1e-14
    De: float = 2e-10

    # 液相初始浓度
    c_e0: float = 1000.0

    # 固相最大浓度
    c_smax_n: float = 31000.0
    c_smax_p: float = 51000.0

    # 初始 stoichiometry
    theta_n0: float = 0.85
    theta_p0: float = 0.20

    # Butler-Volmer 速率常数
    k_n: float = 2e-11
    k_p: float = 2e-11

    # 欧姆/薄膜等效阻抗，Ohm m^2
    R_f: float = 0.006

    # 迁移数
    t_plus: float = 0.38

    # x 方向网格
    N: int = 60


# ============================================================
# 9. OCP / OCV 函数
# ============================================================

def U_cell_analytic_from_soc(soc):
    """
    不需要 OCP 文件的整电池近似 OCV。

    soc 越高，电压越高。
    这是占位模型，用于把流程跑通。
    后续应替换为：
    1. 低倍率 Hyst / OCV 曲线生成的 pseudo-OCV；
    2. 文献中的 NCA/石墨 OCP；
    3. 实验测得的半电池 OCP。
    """

    soc = np.clip(soc, 1e-4, 0.9999)

    return (
        3.05
        + 1.10 * soc
        - 0.10 * np.exp(-20.0 * soc)
        + 0.05 * np.tanh((soc - 0.08) / 0.03)
        - 0.04 * np.tanh((soc - 0.92) / 0.03)
    )


def U_n_graphite_placeholder(theta):
    """
    石墨负极 OCP 占位函数。
    """

    theta = np.clip(theta, 1e-4, 0.9999)

    return (
        0.10
        + 0.80 * np.exp(-10.0 * theta)
        + 0.05 * np.tanh((0.50 - theta) / 0.08)
    )


def U_p_nca_placeholder(theta):
    """
    NCA 正极 OCP 占位函数。
    """

    theta = np.clip(theta, 1e-4, 0.9999)

    return (
        4.25
        - 0.90 * theta
        + 0.10 * np.tanh((0.50 - theta) / 0.08)
    )


# ============================================================
# 10. 简化 P2D 网格与方程
# ============================================================

def make_grid(p: CellParams):
    L_total = p.L_n + p.L_s + p.L_p

    x = np.linspace(0.0, L_total, p.N)
    dx = x[1] - x[0]

    neg = x <= p.L_n
    sep = (x > p.L_n) & (x < p.L_n + p.L_s)
    pos = x >= p.L_n + p.L_s

    eps = np.where(
        neg,
        p.eps_n,
        np.where(sep, p.eps_s, p.eps_p)
    )

    return x, dx, neg, sep, pos, eps


def apply_state_bounds(
    value,
    derivative,
    lower,
    upper,
    restore_tau_s=BOUND_RESTORE_TAU_S
):
    """
    对浓度状态施加软物理边界。

    边界内保持原方程；越界时阻止继续向外发展，并增加恢复项。
    这样可避免长时间完整文件积分时浓度发散到无穷大。
    """
    value_arr = np.asarray(value, dtype=float)
    derivative_arr = np.asarray(derivative, dtype=float).copy()

    below = value_arr < lower
    above = value_arr > upper

    derivative_arr = np.where(
        below,
        np.maximum(derivative_arr, 0.0)
        + (lower - value_arr) / restore_tau_s,
        derivative_arr
    )
    derivative_arr = np.where(
        above,
        np.minimum(derivative_arr, 0.0)
        + (upper - value_arr) / restore_tau_s,
        derivative_arr
    )

    # 位于边界上时不允许导数继续指向边界外
    derivative_arr = np.where(
        (value_arr <= lower) & (derivative_arr < 0.0),
        0.0,
        derivative_arr
    )
    derivative_arr = np.where(
        (value_arr >= upper) & (derivative_arr > 0.0),
        0.0,
        derivative_arr
    )

    if np.ndim(value) == 0:
        return float(derivative_arr)

    return derivative_arr


def rhs(t, y, p: CellParams, current_input: CurrentInput):
    x, dx, neg, sep, pos, eps = make_grid(p)

    if not np.all(np.isfinite(y)):
        raise FloatingPointError(
            f"ODE 状态在 t={t:.6f} s 出现 NaN 或无穷大。"
        )

    c_e_raw = y[:p.N]
    cbar_n = float(y[p.N])
    cbar_p = float(y[p.N + 1])

    # 计算扩散项时只使用物理范围内的浓度，防止雅可比扰动造成巨值
    c_e = np.clip(
        c_e_raw,
        ELECTROLYTE_CONC_MIN,
        ELECTROLYTE_CONC_MAX
    )

    I = float(current_input(t))
    if not np.isfinite(I):
        raise FloatingPointError(
            f"输入电流在 t={t:.6f} s 不是有限值。"
        )

    a_n = 3.0 * p.eps_n / p.R_n
    a_p = 3.0 * p.eps_p / p.R_p

    j_n = I / (F * p.A * a_n * p.L_n)
    j_p = -I / (F * p.A * a_p * p.L_p)

    c_left = np.r_[c_e[1], c_e[:-1]]
    c_right = np.r_[c_e[1:], c_e[-2]]
    d2c_dx2 = (c_right - 2.0 * c_e + c_left) / dx**2

    source = np.zeros_like(c_e)
    source[neg] = (1.0 - p.t_plus) * a_n * j_n / p.eps_n
    source[pos] = (1.0 - p.t_plus) * a_p * j_p / p.eps_p

    dc_e_dt = p.De * d2c_dx2 + source
    dc_e_dt = apply_state_bounds(
        c_e_raw,
        dc_e_dt,
        ELECTROLYTE_CONC_MIN,
        ELECTROLYTE_CONC_MAX
    )

    dcbar_n_dt = -3.0 * j_n / p.R_n
    dcbar_p_dt = -3.0 * j_p / p.R_p

    dcbar_n_dt = apply_state_bounds(
        cbar_n,
        dcbar_n_dt,
        SOLID_CONC_MARGIN,
        p.c_smax_n - SOLID_CONC_MARGIN
    )
    dcbar_p_dt = apply_state_bounds(
        cbar_p,
        dcbar_p_dt,
        SOLID_CONC_MARGIN,
        p.c_smax_p - SOLID_CONC_MARGIN
    )

    derivative = np.r_[dc_e_dt, dcbar_n_dt, dcbar_p_dt]

    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError(
            f"ODE 导数在 t={t:.6f} s 出现 NaN 或无穷大。"
        )

    return derivative


def voltage_from_state(
    t,
    y,
    p: CellParams,
    current_input: CurrentInput,
    ocp_mode="fullcell_analytic"
):
    c_e = np.clip(
        y[:p.N],
        ELECTROLYTE_CONC_MIN,
        ELECTROLYTE_CONC_MAX
    )
    cbar_n = float(np.clip(
        y[p.N],
        SOLID_CONC_MARGIN,
        p.c_smax_n - SOLID_CONC_MARGIN
    ))
    cbar_p = float(np.clip(
        y[p.N + 1],
        SOLID_CONC_MARGIN,
        p.c_smax_p - SOLID_CONC_MARGIN
    ))

    x, dx, neg, sep, pos, eps = make_grid(p)

    I = current_input(t)

    a_n = 3.0 * p.eps_n / p.R_n
    a_p = 3.0 * p.eps_p / p.R_p

    j_n = I / (F * p.A * a_n * p.L_n)
    j_p = -I / (F * p.A * a_p * p.L_p)

    c_surf_n = cbar_n - p.R_n * j_n / (5.0 * p.Ds_n)
    c_surf_p = cbar_p - p.R_p * j_p / (5.0 * p.Ds_p)

    c_surf_n = np.clip(c_surf_n, 1.0, p.c_smax_n - 1.0)
    c_surf_p = np.clip(c_surf_p, 1.0, p.c_smax_p - 1.0)

    theta_n = c_surf_n / p.c_smax_n
    theta_p = c_surf_p / p.c_smax_p

    ce_n = max(float(np.mean(c_e[neg])), 1.0)
    ce_p = max(float(np.mean(c_e[pos])), 1.0)

    i0_n = (
        F
        * p.k_n
        * np.sqrt(ce_n)
        * np.sqrt(c_surf_n)
        * np.sqrt(p.c_smax_n - c_surf_n)
    )

    i0_p = (
        F
        * p.k_p
        * np.sqrt(ce_p)
        * np.sqrt(c_surf_p)
        * np.sqrt(p.c_smax_p - c_surf_p)
    )

    i_n = F * j_n
    i_p = F * j_p

    eta_n = (2.0 * R * T_REF / F) * np.arcsinh(
        i_n / (2.0 * i0_n + 1e-12)
    )

    eta_p = (2.0 * R * T_REF / F) * np.arcsinh(
        i_p / (2.0 * i0_p + 1e-12)
    )

    if ocp_mode == "fullcell_analytic":
        # 没有 OCP 文件时使用整电池近似 OCV
        # 这里用负极平均 stoichiometry 作为有效 SOC 指标
        soc_eff = np.clip(cbar_n / p.c_smax_n, 1e-4, 0.9999)
        U_eq = U_cell_analytic_from_soc(soc_eff)

    elif ocp_mode == "electrode_placeholder":
        U_eq = U_p_nca_placeholder(theta_p) - U_n_graphite_placeholder(theta_n)

    else:
        raise ValueError(f"未知 OCP_MODE: {ocp_mode}")

    V = U_eq + eta_p - eta_n - I * p.R_f / p.A

    return float(V)


# ============================================================
# 11. 完整 parquet 文件仿真
# ============================================================

def make_t_eval(time_s, max_points=1200):
    """
    在完整时间范围内均匀抽取 ODE 输出点。
    电流仍由完整原始时间序列插值输入。
    """
    time_s = np.asarray(time_s, dtype=float)

    if len(time_s) <= max_points:
        return time_s

    idx = np.linspace(
        0,
        len(time_s) - 1,
        max_points,
        dtype=int
    )
    return time_s[np.unique(idx)]


def prepare_full_file_for_simulation(df):
    """
    将一个完整 parquet 文件整理成一次连续仿真的输入。

    保留充电、静置、放电全部数据，不提取或划分任何局部片段。
    若文件中含多个 test_id，则按出现顺序拼接成唯一连续时间轴。
    """
    data = df.copy()

    required = ["time_s", "voltage", "current_discharge"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"完整文件仿真缺少列: {missing}")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "capacity_ah" in data.columns:
        data["capacity_ah"] = pd.to_numeric(
            data["capacity_ah"],
            errors="coerce"
        )

    data = data.dropna(
        subset=["time_s", "voltage", "current_discharge"]
    ).copy()

    if len(data) < 2:
        raise ValueError("完整 parquet 文件的有效数据点少于 2 个。")

    continuous_parts = []
    next_offset_s = 0.0

    if "test_id" in data.columns:
        group_iterator = data.groupby(
            "test_id",
            sort=False,
            dropna=False
        )
    else:
        group_iterator = [("complete_file", data)]

    for test_id, group in group_iterator:
        group = group.sort_values("time_s").copy()
        group = group.drop_duplicates(
            subset="time_s",
            keep="first"
        ).reset_index(drop=True)

        if group.empty:
            continue

        local_time = group["time_s"].to_numpy(dtype=float)
        local_time = local_time - local_time[0]

        positive_dt = np.diff(local_time)
        positive_dt = positive_dt[positive_dt > 0]
        typical_dt = (
            float(np.median(positive_dt))
            if len(positive_dt) > 0
            else 1.0
        )

        group["simulation_time_s"] = local_time + next_offset_s
        group["source_test_id"] = str(test_id)
        continuous_parts.append(group)

        next_offset_s = (
            float(group["simulation_time_s"].iloc[-1])
            + max(typical_dt, 1e-6)
        )

    if not continuous_parts:
        raise ValueError("完整 parquet 文件没有可用于仿真的有效数据。")

    data = pd.concat(continuous_parts, ignore_index=True)
    data = data.sort_values("simulation_time_s").reset_index(drop=True)
    data = data.drop_duplicates(
        subset="simulation_time_s",
        keep="first"
    ).reset_index(drop=True)

    data["simulation_time_s"] = (
        data["simulation_time_s"]
        - data["simulation_time_s"].iloc[0]
    )

    if len(data) < 2 or data["simulation_time_s"].iloc[-1] <= 0:
        raise ValueError("完整 parquet 文件的时间范围无效。")

    dt = (
        data["simulation_time_s"]
        .diff()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    dt = np.maximum(dt, 0.0)

    current = data["current_discharge"].to_numpy(dtype=float)

    data["cumulative_discharge_ah"] = np.cumsum(
        np.maximum(current, 0.0) * dt / 3600.0
    )
    data["cumulative_charge_ah"] = np.cumsum(
        np.maximum(-current, 0.0) * dt / 3600.0
    )
    data["net_discharged_ah"] = np.cumsum(
        current * dt / 3600.0
    )

    return data


def estimate_initial_soc(first_voltage_v):
    """
    用内置整电池 OCV 曲线反查初始 SOC。
    """
    soc_grid = np.linspace(
        INITIAL_SOC_MIN,
        INITIAL_SOC_MAX,
        4000
    )
    voltage_grid = U_cell_analytic_from_soc(soc_grid)
    index = int(np.argmin(np.abs(voltage_grid - first_voltage_v)))
    return float(soc_grid[index])


def make_jacobian_sparsity(p: CellParams):
    """
    简化 P2D 状态方程的雅可比稀疏结构：
    电解液浓度为三对角耦合，两个固相平均浓度为独立状态。
    """
    size = p.N + 2
    sparsity = lil_matrix((size, size), dtype=int)

    for i in range(p.N):
        sparsity[i, i] = 1
        if i > 0:
            sparsity[i, i - 1] = 1
        if i < p.N - 1:
            sparsity[i, i + 1] = 1

    sparsity[p.N, p.N] = 1
    sparsity[p.N + 1, p.N + 1] = 1

    return sparsity.tocsr()


def solver_atol_vector(p: CellParams):
    return np.r_[
        np.full(p.N, SOLVER_ATOL_ELECTROLYTE, dtype=float),
        SOLVER_ATOL_SOLID,
        SOLVER_ATOL_SOLID,
    ]


def run_solver_window(
    t_start,
    t_end,
    y_start,
    t_eval,
    p,
    current_input,
    method
):
    kwargs = {
        "fun": lambda t, y: rhs(t, y, p, current_input),
        "t_span": (float(t_start), float(t_end)),
        "y0": np.asarray(y_start, dtype=float),
        "method": method,
        "t_eval": np.asarray(t_eval, dtype=float),
        "rtol": SOLVER_RTOL,
        "atol": solver_atol_vector(p),
        "max_step": SOLVER_MAX_STEP_S,
    }

    if method in {"BDF", "Radau"}:
        kwargs["jac_sparsity"] = make_jacobian_sparsity(p)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        solution = solve_ivp(**kwargs)

    severe_warnings = [
        warning
        for warning in caught
        if (
            "overflow" in str(warning.message).lower()
            or "invalid value" in str(warning.message).lower()
        )
    ]

    finite_solution = (
        solution.y.size > 0
        and np.all(np.isfinite(solution.y))
        and np.all(np.isfinite(solution.t))
    )

    return solution, severe_warnings, finite_solution


def solve_complete_trajectory(
    time_s,
    t_eval,
    y0,
    p,
    current_input
):
    """
    对整个 parquet 保持一条连续状态轨迹。

    为提高长时积分稳定性，求解器内部按固定时间窗口设置检查点，
    每个窗口都继承上一个窗口的最终状态；结果仍是一份完整连续仿真。
    """
    t_end = float(time_s[-1])

    window_edges = [0.0]
    while window_edges[-1] < t_end:
        window_edges.append(
            min(window_edges[-1] + SOLVER_WINDOW_S, t_end)
        )

    all_t = []
    all_y = []
    y_start = np.asarray(y0, dtype=float)

    for window_index in range(len(window_edges) - 1):
        t_start = float(window_edges[window_index])
        t_stop = float(window_edges[window_index + 1])

        local_eval = t_eval[
            (t_eval >= t_start) & (t_eval <= t_stop)
        ]

        # 每个窗口必须包含起点和终点，以便连续传递状态
        local_eval = np.unique(
            np.r_[t_start, local_eval, t_stop]
        )

        solution, severe_warnings, finite_solution = run_solver_window(
            t_start,
            t_stop,
            y_start,
            local_eval,
            p,
            current_input,
            SOLVER_PRIMARY_METHOD
        )

        needs_fallback = (
            not solution.success
            or not finite_solution
            or len(severe_warnings) > 0
        )

        if needs_fallback:
            print(
                f"⚠️ 数值窗口 {window_index + 1}/"
                f"{len(window_edges) - 1} 的 "
                f"{SOLVER_PRIMARY_METHOD} 不稳定，"
                f"自动改用 {SOLVER_FALLBACK_METHOD}。"
            )

            solution, fallback_warnings, finite_solution = run_solver_window(
                t_start,
                t_stop,
                y_start,
                local_eval,
                p,
                current_input,
                SOLVER_FALLBACK_METHOD
            )

            if fallback_warnings:
                print(
                    f"⚠️ {SOLVER_FALLBACK_METHOD} 仍产生 "
                    f"{len(fallback_warnings)} 条数值警告。"
                )

        if not solution.success:
            raise RuntimeError(
                f"完整文件积分在 {t_start:.3f}–{t_stop:.3f} s 失败: "
                f"{solution.message}"
            )

        if not finite_solution:
            raise FloatingPointError(
                f"完整文件积分在 {t_start:.3f}–{t_stop:.3f} s "
                "产生 NaN 或无穷大。"
            )

        y_window = solution.y.copy()

        # 输出与后续计算均保持在物理边界内
        y_window[:p.N, :] = np.clip(
            y_window[:p.N, :],
            ELECTROLYTE_CONC_MIN,
            ELECTROLYTE_CONC_MAX
        )
        y_window[p.N, :] = np.clip(
            y_window[p.N, :],
            SOLID_CONC_MARGIN,
            p.c_smax_n - SOLID_CONC_MARGIN
        )
        y_window[p.N + 1, :] = np.clip(
            y_window[p.N + 1, :],
            SOLID_CONC_MARGIN,
            p.c_smax_p - SOLID_CONC_MARGIN
        )

        if window_index > 0:
            all_t.append(solution.t[1:])
            all_y.append(y_window[:, 1:])
        else:
            all_t.append(solution.t)
            all_y.append(y_window)

        y_start = y_window[:, -1]

    full_t = np.concatenate(all_t)
    full_y = np.concatenate(all_y, axis=1)

    # 只返回原先需要的全局输出时间点
    output_indices = np.searchsorted(full_t, t_eval)
    output_indices = np.clip(output_indices, 0, len(full_t) - 1)

    # searchsorted 可能指向右侧点，选择距离更近者
    left_indices = np.maximum(output_indices - 1, 0)
    choose_left = (
        np.abs(full_t[left_indices] - t_eval)
        < np.abs(full_t[output_indices] - t_eval)
    )
    output_indices = np.where(
        choose_left,
        left_indices,
        output_indices
    )

    return full_t[output_indices], full_y[:, output_indices]



def simulate_full_file(df, p: CellParams, ocp_mode="fullcell_analytic"):
    """
    一个 parquet 文件只执行一次完整 P2D 前向仿真。
    """
    data = prepare_full_file_for_simulation(df)

    time_s = data["simulation_time_s"].to_numpy(dtype=float)
    current_a = data["current_discharge"].to_numpy(dtype=float)

    if not np.all(np.isfinite(time_s)):
        raise ValueError("完整时间轴包含 NaN 或无穷大。")
    if not np.all(np.isfinite(current_a)):
        raise ValueError("完整电流输入包含 NaN 或无穷大。")

    duration_h = float(time_s[-1] / 3600.0)
    max_current_a = float(np.max(np.abs(current_a)))
    net_ah_min = float(data["net_discharged_ah"].min())
    net_ah_max = float(data["net_discharged_ah"].max())

    print(f"   完整文件时长: {duration_h:.4f} h")
    print(f"   最大绝对电流: {max_current_a:.4f} A")
    print(
        f"   净放电容量轨迹范围: "
        f"{net_ah_min:.4f} ～ {net_ah_max:.4f} Ah"
    )

    current_input = CurrentInput(
        time_s=time_s,
        current_a=current_a
    )

    t_eval = make_t_eval(
        time_s,
        max_points=MAX_SOLVE_POINTS
    )

    if (
        ESTIMATE_INITIAL_SOC_FROM_FIRST_VOLTAGE
        and ocp_mode == "fullcell_analytic"
    ):
        initial_soc = estimate_initial_soc(
            float(data["voltage"].iloc[0])
        )
        print(
            f"   按首个实测电压估计初始 SOC: {initial_soc:.4f}"
        )
    else:
        initial_soc = float(np.clip(
            p.theta_n0,
            INITIAL_SOC_MIN,
            INITIAL_SOC_MAX
        ))

    c_e_init = np.full(p.N, p.c_e0)
    cbar_n_init = initial_soc * p.c_smax_n

    # 与默认参数关系保持一致，近似令正极化学计量比随 SOC 反向变化
    theta_p_init = np.clip(
        p.theta_p0 + (p.theta_n0 - initial_soc),
        INITIAL_SOC_MIN,
        INITIAL_SOC_MAX
    )
    cbar_p_init = theta_p_init * p.c_smax_p
    y0 = np.r_[c_e_init, cbar_n_init, cbar_p_init]

    solved_t, solved_y = solve_complete_trajectory(
        time_s,
        t_eval,
        y0,
        p,
        current_input
    )

    voltage_predicted = np.array(
        [
            voltage_from_state(
                t,
                solved_y[:, i],
                p,
                current_input,
                ocp_mode=ocp_mode
            )
            for i, t in enumerate(solved_t)
        ]
    )

    voltage_measured = np.interp(
        solved_t,
        data["simulation_time_s"],
        data["voltage"]
    )
    current_interp = np.array(
        [current_input(t) for t in solved_t]
    )

    result_dict = {
        "time_s": solved_t,
        "time_h": solved_t / 3600.0,
        "current_a": current_interp,
        "cumulative_discharge_ah": np.interp(
            solved_t,
            data["simulation_time_s"],
            data["cumulative_discharge_ah"]
        ),
        "cumulative_charge_ah": np.interp(
            solved_t,
            data["simulation_time_s"],
            data["cumulative_charge_ah"]
        ),
        "net_discharged_ah": np.interp(
            solved_t,
            data["simulation_time_s"],
            data["net_discharged_ah"]
        ),
        "voltage_measured_v": voltage_measured,
        "voltage_predicted_v": voltage_predicted,
        "voltage_error_v": voltage_predicted - voltage_measured,
        "cbar_n": solved_y[p.N, :],
        "cbar_p": solved_y[p.N + 1, :],
        "theta_n_bar": solved_y[p.N, :] / p.c_smax_n,
        "theta_p_bar": solved_y[p.N + 1, :] / p.c_smax_p,
    }

    if "capacity_ah" in data.columns:
        valid_capacity = data["capacity_ah"].notna()
        if valid_capacity.sum() >= 2:
            result_dict["recorded_capacity_ah"] = np.interp(
                solved_t,
                data.loc[valid_capacity, "simulation_time_s"],
                data.loc[valid_capacity, "capacity_ah"]
            )

    result = pd.DataFrame(result_dict)

    voltage_error = (
        result["voltage_predicted_v"]
        - result["voltage_measured_v"]
    )
    rmse = float(np.sqrt(np.mean(voltage_error ** 2)))
    mae = float(np.mean(np.abs(voltage_error)))

    return result, rmse, mae, data


# ============================================================
# 12. 完整仿真结果绘图与保存
# ============================================================

def safe_name(text):
    text = str(text)
    bad_chars = [
        "/", chr(92), ":", "*", "?",
        chr(34), "<", ">", "|", " "
    ]
    for char in bad_chars:
        text = text.replace(char, "_")
    return text[:160]


def plot_complete_simulation(result, title, save_prefix):
    """
    显示整个 parquet 文件的电压、电压误差和电流。
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 11),
        sharex=True
    )

    axes[0].plot(
        result["time_h"],
        result["voltage_measured_v"],
        label="Measured voltage",
        linewidth=1.2
    )
    axes[0].plot(
        result["time_h"],
        result["voltage_predicted_v"],
        label="P2D predicted voltage",
        linewidth=1.2
    )
    axes[0].set_ylabel("Voltage / V")
    axes[0].set_title(title)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        result["time_h"],
        result["voltage_error_v"],
        linewidth=1.0
    )
    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=0.8
    )
    axes[1].set_ylabel("Voltage error / V")
    axes[1].grid(True)

    axes[2].plot(
        result["time_h"],
        result["current_a"],
        linewidth=1.0
    )
    axes[2].axhline(
        0.0,
        linestyle="--",
        linewidth=0.8
    )
    axes[2].set_xlabel("Complete file time / h")
    axes[2].set_ylabel("Current / A")
    axes[2].grid(True)

    plt.tight_layout()

    image_path = save_prefix + "_full_p2d.png"
    fig.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )
    print(f"✅ 已保存完整仿真图像: {image_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# 13. 主程序
# ============================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parquet_files = list_parquet_files(
        DATA_PATH,
        use_s3=USE_S3,
        max_files=MAX_FILES,
        file_name_contains=FILE_NAME_CONTAINS,
        file_name_excludes=FILE_NAME_EXCLUDES
    )

    p = CellParams()
    batch_summary_rows = []

    for file_index, parquet_file in enumerate(parquet_files):

        try:
            df_raw = read_one_parquet(
                parquet_file,
                use_s3=USE_S3
            )
            df = preprocess_dataframe(df_raw)

            print()
            print("🚀 开始完整 parquet 文件仿真:")
            print(f"   文件编号: {file_index}")
            print(f"   文件名称: {os.path.basename(parquet_file)}")
            print(f"   原始有效数据点: {len(df)}")

            result, rmse, mae, prepared_data = simulate_full_file(
                df,
                p,
                ocp_mode=OCP_MODE
            )

            file_stem = os.path.splitext(
                os.path.basename(parquet_file)
            )[0]
            base = safe_name(
                f"file{file_index}_{file_stem}"
            )

            result_path = os.path.join(
                OUTPUT_DIR,
                base + "_full_p2d_result.csv"
            )
            result.to_csv(result_path, index=False)

            prepared_path = os.path.join(
                OUTPUT_DIR,
                base + "_full_prepared_input.csv"
            )
            prepared_data.to_csv(prepared_path, index=False)

            print(f"✅ 已保存完整仿真结果: {result_path}")
            print(f"✅ 已保存完整连续输入: {prepared_path}")
            print(f"   完整时长: {result['time_h'].iloc[-1]:.4f} h")
            print(f"   仿真输出点: {len(result)}")
            print(f"   RMSE = {rmse:.4f} V")
            print(f"   MAE  = {mae:.4f} V")

            save_prefix = os.path.join(
                OUTPUT_DIR,
                base
            )
            plot_complete_simulation(
                result,
                title=(
                    "Complete P2D simulation: "
                    + os.path.basename(parquet_file)
                ),
                save_prefix=save_prefix
            )

            batch_summary_rows.append(
                {
                    "file_index": file_index,
                    "file_path": parquet_file,
                    "input_points": len(prepared_data),
                    "simulation_points": len(result),
                    "duration_h": float(result["time_h"].iloc[-1]),
                    "mean_current_A": float(result["current_a"].mean()),
                    "cumulative_discharge_ah": float(
                        result["cumulative_discharge_ah"].iloc[-1]
                    ),
                    "cumulative_charge_ah": float(
                        result["cumulative_charge_ah"].iloc[-1]
                    ),
                    "voltage_start_V": float(
                        result["voltage_measured_v"].iloc[0]
                    ),
                    "voltage_end_V": float(
                        result["voltage_measured_v"].iloc[-1]
                    ),
                    "rmse_V": rmse,
                    "mae_V": mae,
                    "ocp_mode": OCP_MODE,
                    "result_path": result_path,
                }
            )

        except Exception as exc:
            print()
            print("❌ 当前完整文件处理失败:")
            print(f"   文件: {parquet_file}")
            print(f"   错误: {exc}")

            batch_summary_rows.append(
                {
                    "file_index": file_index,
                    "file_path": parquet_file,
                    "error": str(exc),
                }
            )

    batch_summary = pd.DataFrame(batch_summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "batch_full_p2d_summary.csv"
    )
    batch_summary.to_csv(summary_path, index=False)

    print()
    print("🎉 完整 parquet 文件 P2D 前向仿真完成。")
    print(f"汇总结果: {summary_path}")
