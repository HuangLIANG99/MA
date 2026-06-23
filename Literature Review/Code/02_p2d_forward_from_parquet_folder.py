import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


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


# 每个 parquet 文件最多仿真几个放电片段
MAX_DISCHARGE_SEGMENTS_PER_FILE = None

# 片段提取参数
MIN_CURRENT_ABS = 0.05
MIN_POINTS = 100
MIN_DURATION_S = 30.0

# 18650 VTC6 可以先按 3Ah 估计
NOMINAL_CAPACITY_AH = 3.0

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
        files = [
            f for f in files
            if file_name_contains in os.path.basename(f)
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
    统一电流单位为 A。
    如果电流中位数大于 50，认为原始单位可能是 mA。
    """

    df = df.copy()

    current_abs_median = df["current"].abs().median()

    if current_abs_median > 50:
        df["current"] = df["current"] / 1000.0
        print("✅ 电流单位可能是 mA，已转换为 A。")
    else:
        print("✅ 电流单位看起来已经是 A。")

    print(f"   电流绝对值中位数: {df['current'].abs().median():.6f} A")

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
# 6. 提取放电片段
# ============================================================

def extract_discharge_segments(
    df,
    min_current_abs=0.05,
    min_points=100,
    min_duration_s=30.0,
    nominal_capacity_ah=3.0
):
    """
    从一个 DataFrame 中提取所有连续放电片段。
    """

    all_segments = []
    summary_rows = []

    if "test_id" in df.columns:
        group_iterator = df.groupby("test_id", sort=False)
    else:
        group_iterator = [("unknown", df)]

    global_segment_id = 0

    for test_id, one in group_iterator:
        one = one.sort_values("time_s").reset_index(drop=True)

        if len(one) == 0:
            continue

        one["local_segment_id"] = (
            one["operation"] != one["operation"].shift(
                fill_value=one["operation"].iloc[0]
            )
        ).cumsum()

        for _, g in one.groupby("local_segment_id", sort=False):
            mode = g["operation"].iloc[0]

            if mode != "discharge":
                continue

            if len(g) < min_points:
                continue

            duration_s = g["time_s"].iloc[-1] - g["time_s"].iloc[0]

            if duration_s < min_duration_s:
                continue

            g = g.copy()
            g = g.sort_values("time_s").reset_index(drop=True)

            g["time_segment_s"] = g["time_s"] - g["time_s"].iloc[0]

            dt = g["time_segment_s"].diff().fillna(0).to_numpy()
            dt = np.maximum(dt, 0.0)

            I = g["current_discharge"].to_numpy(dtype=float)
            I_pos = np.maximum(I, 0.0)

            g["capacity_segment_ah"] = np.cumsum(I_pos * dt / 3600.0)

            g["global_segment_id"] = global_segment_id
            g["mode"] = "discharge"

            mean_current = float(np.mean(I_pos))
            capacity_ah = float(g["capacity_segment_ah"].iloc[-1])
            c_rate = mean_current / nominal_capacity_ah

            summary_rows.append(
                {
                    "global_segment_id": global_segment_id,
                    "test_id": test_id,
                    "mode": "discharge",
                    "points": len(g),
                    "duration_h": duration_s / 3600.0,
                    "capacity_ah": capacity_ah,
                    "mean_current_A": mean_current,
                    "estimated_C_rate": c_rate,
                    "voltage_start_V": float(g["voltage"].iloc[0]),
                    "voltage_end_V": float(g["voltage"].iloc[-1]),
                    "voltage_min_V": float(g["voltage"].min()),
                    "voltage_max_V": float(g["voltage"].max()),
                }
            )

            all_segments.append(g)
            global_segment_id += 1

    if len(all_segments) == 0:
        return [], pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    # 默认按容量从大到小排列，优先仿真容量最大的放电段
    summary_df = summary_df.sort_values(
        "capacity_ah",
        ascending=False
    ).reset_index(drop=True)

    return all_segments, summary_df


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


def rhs(t, y, p: CellParams, current_input: CurrentInput):
    x, dx, neg, sep, pos, eps = make_grid(p)

    c_e = y[:p.N]
    cbar_n = y[p.N]
    cbar_p = y[p.N + 1]

    I = current_input(t)

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

    dcbar_n_dt = -3.0 * j_n / p.R_n
    dcbar_p_dt = -3.0 * j_p / p.R_p

    return np.r_[dc_e_dt, dcbar_n_dt, dcbar_p_dt]


def voltage_from_state(
    t,
    y,
    p: CellParams,
    current_input: CurrentInput,
    ocp_mode="fullcell_analytic"
):
    c_e = y[:p.N]
    cbar_n = y[p.N]
    cbar_p = y[p.N + 1]

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
# 11. 仿真辅助函数
# ============================================================

def make_t_eval(time_s, max_points=1200):
    time_s = np.asarray(time_s, dtype=float)

    if len(time_s) <= max_points:
        return time_s

    idx = np.linspace(
        0,
        len(time_s) - 1,
        max_points,
        dtype=int
    )

    idx = np.unique(idx)

    return time_s[idx]


def prepare_segment_for_simulation(seg):
    seg = seg.copy()

    seg["time_s"] = seg["time_segment_s"].astype(float)
    seg["time_s"] = seg["time_s"] - seg["time_s"].iloc[0]

    if "capacity_segment_ah" in seg.columns:
        seg["capacity_ah"] = seg["capacity_segment_ah"].astype(float)
    else:
        dt = seg["time_s"].diff().fillna(0).to_numpy()
        dt = np.maximum(dt, 0.0)
        I = np.maximum(
            seg["current_discharge"].to_numpy(dtype=float),
            0.0
        )
        seg["capacity_ah"] = np.cumsum(I * dt / 3600.0)

    keep_cols = [
        "time_s",
        "voltage",
        "current_discharge",
        "capacity_ah",
        "global_segment_id",
    ]

    seg = seg[keep_cols].copy()

    for col in ["time_s", "voltage", "current_discharge", "capacity_ah"]:
        seg[col] = pd.to_numeric(seg[col], errors="coerce")

    seg = seg.dropna()
    seg = seg.sort_values("time_s").reset_index(drop=True)
    seg = seg.drop_duplicates(subset="time_s").reset_index(drop=True)

    return seg


def simulate_one_segment(seg, p: CellParams, ocp_mode="fullcell_analytic"):
    seg = prepare_segment_for_simulation(seg)

    time_s = seg["time_s"].to_numpy(dtype=float)
    current_a = seg["current_discharge"].to_numpy(dtype=float)

    current_input = CurrentInput(
        time_s=time_s,
        current_a=current_a
    )

    t_eval = make_t_eval(
        time_s,
        max_points=MAX_SOLVE_POINTS
    )

    t_end = float(time_s[-1])

    c_e_init = np.full(p.N, p.c_e0)
    cbar_n_init = p.theta_n0 * p.c_smax_n
    cbar_p_init = p.theta_p0 * p.c_smax_p

    y0 = np.r_[c_e_init, cbar_n_init, cbar_p_init]

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p, current_input),
        t_span=(0.0, t_end),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"ODE 求解失败: {sol.message}")

    V_pred = np.array(
        [
            voltage_from_state(
                t,
                sol.y[:, i],
                p,
                current_input,
                ocp_mode=ocp_mode
            )
            for i, t in enumerate(sol.t)
        ]
    )

    V_meas = np.interp(
        sol.t,
        seg["time_s"],
        seg["voltage"]
    )

    capacity_ah = np.interp(
        sol.t,
        seg["time_s"],
        seg["capacity_ah"]
    )

    current_interp = np.array(
        [current_input(t) for t in sol.t]
    )

    result = pd.DataFrame(
        {
            "time_s": sol.t,
            "time_h": sol.t / 3600.0,
            "capacity_ah": capacity_ah,
            "current_a": current_interp,
            "voltage_measured_v": V_meas,
            "voltage_predicted_v": V_pred,
            "voltage_error_v": V_pred - V_meas,
            "cbar_n": sol.y[p.N, :],
            "cbar_p": sol.y[p.N + 1, :],
            "theta_n_bar": sol.y[p.N, :] / p.c_smax_n,
            "theta_p_bar": sol.y[p.N + 1, :] / p.c_smax_p,
        }
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (result["voltage_predicted_v"] - result["voltage_measured_v"]) ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                result["voltage_predicted_v"] - result["voltage_measured_v"]
            )
        )
    )

    return result, rmse, mae


# ============================================================
# 12. 画图与保存
# ============================================================

def safe_name(text):
    text = str(text)
    bad_chars = ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]
    for c in bad_chars:
        text = text.replace(c, "_")
    return text[:160]


def plot_result(result, title, save_prefix):
    # 创建一个包含2个子图的图形，垂直排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # ===== 第一张子图：电压-容量曲线对比 =====
    ax1.plot(
        result["capacity_ah"],
        result["voltage_measured_v"],
        label="measured Spannung",
        linewidth=1.5,
        color='blue'
    )
    
    ax1.plot(
        result["capacity_ah"],
        result["voltage_predicted_v"],
        label="simplified P2D prediction",
        linewidth=1.5,
        color='orange'
    )
    
    ax1.set_xlabel("Discharged capacity / Ah")
    ax1.set_ylabel("Voltage / V")
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()
    
    # ===== 第二张子图：电压预测误差 =====
    ax2.plot(
        result["capacity_ah"],
        result["voltage_error_v"],
        linewidth=1.2,
        color='green'
    )
    
    ax2.axhline(0.0, linestyle="--", linewidth=1.0, color='red', alpha=0.7)
    ax2.set_xlabel("Discharged capacity / Ah")
    ax2.set_ylabel("Voltage error / V")
    ax2.set_title("Voltage prediction error")
    ax2.grid(True)
    
    # 调整布局，防止标签重叠
    plt.tight_layout()
    
    # 保存完整图形
    path = save_prefix + "_combined.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存图像: {path}")
    
    # 显示图形
    plt.show()

# ============================================================
# 13. 主程序
# ============================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parquet_files = list_parquet_files(
        DATA_PATH,
        use_s3=USE_S3,
        max_files=MAX_FILES,
        file_name_contains=FILE_NAME_CONTAINS
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

            discharge_segments, discharge_summary = extract_discharge_segments(
                df,
                min_current_abs=MIN_CURRENT_ABS,
                min_points=MIN_POINTS,
                min_duration_s=MIN_DURATION_S,
                nominal_capacity_ah=NOMINAL_CAPACITY_AH
            )

            if len(discharge_segments) == 0:
                print("⚠️ 这个文件没有找到有效放电片段，跳过。")
                continue

            print("\n📋 当前文件放电片段统计:")
            print(discharge_summary.head(10))

            # 选择容量最大的若干个放电片段
            chosen_summary = discharge_summary.head(
                MAX_DISCHARGE_SEGMENTS_PER_FILE
            )

            for _, row in chosen_summary.iterrows():

                segment_id = int(row["global_segment_id"])

                seg = discharge_segments[segment_id]

                print("\n🚀 开始仿真:")
                print(f"   文件编号: {file_index}")
                print(f"   segment_id: {segment_id}")
                print(f"   容量: {row['capacity_ah']:.4f} Ah")
                print(f"   平均电流: {row['mean_current_A']:.4f} A")
                print(f"   估计倍率: {row['estimated_C_rate']:.4f} C")

                result, rmse, mae = simulate_one_segment(
                    seg,
                    p,
                    ocp_mode=OCP_MODE
                )

                base = safe_name(
                    f"file{file_index}_seg{segment_id}"
                )

                result_path = os.path.join(
                    OUTPUT_DIR,
                    base + "_p2d_result.csv"
                )

                result.to_csv(result_path, index=False)

                print(f"✅ 已保存仿真结果: {result_path}")
                print(f"   RMSE = {rmse:.4f} V")
                print(f"   MAE  = {mae:.4f} V")

                save_prefix = os.path.join(
                    OUTPUT_DIR,
                    base
                )

                plot_result(
                    result,
                    title=f"File {file_index}, Segment {segment_id}",
                    save_prefix=save_prefix
                )

                batch_summary_rows.append(
                    {
                        "file_index": file_index,
                        "file_path": parquet_file,
                        "segment_id": segment_id,
                        "capacity_ah": row["capacity_ah"],
                        "mean_current_A": row["mean_current_A"],
                        "estimated_C_rate": row["estimated_C_rate"],
                        "duration_h": row["duration_h"],
                        "voltage_start_V": row["voltage_start_V"],
                        "voltage_end_V": row["voltage_end_V"],
                        "rmse_V": rmse,
                        "mae_V": mae,
                        "ocp_mode": OCP_MODE,
                    }
                )

        except Exception as e:
            print("\n❌ 当前文件处理失败:")
            print(f"   文件: {parquet_file}")
            print(f"   错误: {e}")

            batch_summary_rows.append(
                {
                    "file_index": file_index,
                    "file_path": parquet_file,
                    "error": str(e),
                }
            )

    batch_summary = pd.DataFrame(batch_summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "batch_p2d_summary.csv"
    )

    batch_summary.to_csv(summary_path, index=False)

    print("\n🎉 批量 parquet 直接读取 + P2D 前向仿真完成。")
    print(f"汇总结果: {summary_path}")
