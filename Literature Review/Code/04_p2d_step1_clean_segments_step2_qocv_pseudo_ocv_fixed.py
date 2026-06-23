import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

# qOCV / 低倍率文件夹。第二步必须优先从这里生成 pseudo-OCV，
# 不要再把 ageing 里的多个放电片段混合成一条 OCV 曲线。
# 示例：
# QOCV_DATA_PATH = "projects/.../qOCV"
QOCV_DATA_PATH = None
QOCV_USE_S3 = USE_S3
QOCV_MAX_FILES = None
QOCV_FILE_NAME_CONTAINS = None
QOCV_FILE_NAME_EXCLUDES = ("Init", "pulse")
QOCV_MIN_CAPACITY_AH = 2.0
QOCV_SELECT_MAX_C_RATE = 0.20
QOCV_ENFORCE_MONOTONIC_OCV = True

# 图像只保存，不弹窗。
DISPLAY_PLOTS = False

# 只读取前几个 parquet 文件，用来节省时间
MAX_FILES = 3

# 文件名过滤。None 表示不过滤。
# 例如可以设成 "finished.parquet" 或 "Ageing"
FILE_NAME_CONTAINS = None

# 可为字符串或元组；大小写不敏感。
# 建议先排除 Init / pulse，因为第一步只需要干净恒流放电窗口。
FILE_NAME_EXCLUDES = ("Init", "pulse")

# 每个 parquet 文件最多仿真几个放电片段。
# None 表示所有通过清洁筛选的片段都仿真。
MAX_DISCHARGE_SEGMENTS_PER_FILE = None

# 片段提取参数
MIN_CURRENT_ABS = 0.05
MIN_POINTS = 100
MIN_DURATION_S = 30.0

# 第一阶段：干净恒流放电窗口筛选
# 这些阈值用于剔除 CV 段、混合工况段、过短片段、温漂过大的片段等。
MIN_CAPACITY_AH = 0.03
MIN_VOLTAGE_DROP_V = 0.10
CC_REL_STD_MAX = 0.03          # 恒流段相对标准差阈值
CC_ABS_STD_MAX_A = 0.05        # 恒流段绝对标准差兜底阈值
VOLTAGE_INCREASE_STEP_V = 0.002
VOLTAGE_INCREASE_FRACTION_MAX = 0.08
TEMP_DELTA_MAX_C = 3.0
VOLTAGE_MIN_ALLOWED = 2.0
VOLTAGE_MAX_ALLOWED = 4.5

# 第二阶段：pseudo-OCV 生成
# 优先使用低倍率、干净、容量较大的放电片段。如果没有低倍率片段，脚本会退而使用最低倍率片段，
# 但会在日志和文件中标注：这只是 pseudo-OCV，不是真正准平衡 OCV。
BUILD_PSEUDO_OCV = True
PSEUDO_OCV_MAX_C_RATE = QOCV_SELECT_MAX_C_RATE
PSEUDO_OCV_NUM_SEGMENTS = 3
PSEUDO_OCV_GRID_POINTS = 600
PSEUDO_OCV_IR_CORRECTION_OHM = 0.0
PSEUDO_OCV_FILENAME = "pseudo_ocv_nca_vtc6.csv"
PSEUDO_OCV_RAW_FILENAME = "pseudo_ocv_nca_vtc6_raw_points.csv"
PSEUDO_OCV_PLOT_FILENAME = "pseudo_ocv_nca_vtc6.png"

# 18650 VTC6 可以先按 3Ah 估计；之后建议从完整低倍率容量重新估计。
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
# "pseudo_ocv": 使用由低倍率/最低倍率干净放电片段生成的 pseudo-OCV 表
# "fullcell_analytic": 不需要 OCP 文件，使用内置整电池近似 OCV
# "electrode_placeholder": 使用占位正负极 OCP
OCP_MODE = "pseudo_ocv"

# 当 pseudo-OCV 还没有生成或加载失败时，是否退回 analytic OCV。
FALLBACK_TO_ANALYTIC_OCV = True

# ODE 输出点数，越大越慢
MAX_SOLVE_POINTS = 1200

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

    修正点：
    1. file_name_excludes 真正生效；
    2. contains / excludes 均大小写不敏感；
    3. 先过滤再截断 max_files，避免前几个文件全是无用文件。
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
        contains_lower = str(file_name_contains).lower()
        files = [
            f for f in files
            if contains_lower in os.path.basename(f).lower()
        ]

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

    注意：ISEA parquet 的 Strom 通常已经是 A。这里保留一个保守检查：
    只有当电流绝对值中位数明显超过电芯测试可能范围时，才提示可能是 mA 并转换。
    """

    df = df.copy()
    df["current"] = pd.to_numeric(df["current"], errors="coerce")

    current_abs = df["current"].abs().dropna()

    if len(current_abs) == 0:
        raise ValueError("电流列没有有效数值。")

    current_abs_median = float(current_abs.median())
    current_abs_q95 = float(current_abs.quantile(0.95))

    if current_abs_median > 50:
        df["current"] = df["current"] / 1000.0
        print("⚠️ 电流中位数 > 50，疑似 mA，已转换为 A。")
    else:
        print("✅ 电流单位按 A 使用。")

    print(f"   电流绝对值中位数: {df['current'].abs().median():.6f} A")
    print(f"   电流绝对值95%分位数: {current_abs_q95:.6f} A")

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
# 6. 提取并筛选干净恒流放电片段
# ============================================================

def assess_discharge_segment_quality(g, nominal_capacity_ah=3.0):
    """
    对候选放电片段做第一步质量筛选。

    目标不是证明模型正确，而是为后续参数辨识挑出：
    1. 连续放电；
    2. 近似恒流；
    3. 电压整体单调下降；
    4. 容量/时长足够；
    5. 温度漂移不过大；
    6. 电压范围合理。
    """

    reasons = []

    I = np.maximum(
        g["current_discharge"].to_numpy(dtype=float),
        0.0
    )
    V = g["voltage"].to_numpy(dtype=float)

    time_s = g["time_s"].to_numpy(dtype=float)
    duration_s = float(time_s[-1] - time_s[0])

    local_time_s = time_s - time_s[0]
    dt = np.diff(local_time_s, prepend=0.0)
    dt = np.maximum(dt, 0.0)

    capacity_ah = float(np.sum(I * dt / 3600.0))
    mean_current = float(np.mean(I))
    std_current = float(np.std(I))
    rel_std_current = std_current / max(abs(mean_current), 1e-12)
    estimated_c_rate = mean_current / nominal_capacity_ah

    voltage_start = float(V[0])
    voltage_end = float(V[-1])
    voltage_drop = voltage_start - voltage_end
    voltage_min = float(np.min(V))
    voltage_max = float(np.max(V))

    dV = np.diff(V)
    if len(dV) > 0:
        voltage_increase_fraction = float(
            np.mean(dV > VOLTAGE_INCREASE_STEP_V)
        )
    else:
        voltage_increase_fraction = 1.0

    if "temperature" in g.columns:
        temp = pd.to_numeric(g["temperature"], errors="coerce")
        if temp.notna().sum() > 0:
            temperature_mean = float(temp.mean())
            temperature_delta = float(temp.max() - temp.min())
        else:
            temperature_mean = np.nan
            temperature_delta = np.nan
    else:
        temperature_mean = np.nan
        temperature_delta = np.nan

    # 判据
    if len(g) < MIN_POINTS:
        reasons.append("too_few_points")

    if duration_s < MIN_DURATION_S:
        reasons.append("too_short_duration")

    if capacity_ah < MIN_CAPACITY_AH:
        reasons.append("too_small_capacity")

    if mean_current < MIN_CURRENT_ABS:
        reasons.append("mean_current_too_small")

    is_cc_by_relative = rel_std_current <= CC_REL_STD_MAX
    is_cc_by_absolute = std_current <= CC_ABS_STD_MAX_A

    if not (is_cc_by_relative or is_cc_by_absolute):
        reasons.append("not_constant_current")

    if voltage_drop < MIN_VOLTAGE_DROP_V:
        reasons.append("voltage_drop_too_small")

    if voltage_increase_fraction > VOLTAGE_INCREASE_FRACTION_MAX:
        reasons.append("voltage_not_monotonic_enough")

    if voltage_min < VOLTAGE_MIN_ALLOWED or voltage_max > VOLTAGE_MAX_ALLOWED:
        reasons.append("voltage_out_of_range")

    if (
        np.isfinite(temperature_delta)
        and temperature_delta > TEMP_DELTA_MAX_C
    ):
        reasons.append("temperature_drift_too_large")

    is_clean = len(reasons) == 0

    return {
        "points": int(len(g)),
        "duration_h": duration_s / 3600.0,
        "capacity_ah": capacity_ah,
        "mean_current_A": mean_current,
        "std_current_A": std_current,
        "relative_std_current": rel_std_current,
        "estimated_C_rate": estimated_c_rate,
        "voltage_start_V": voltage_start,
        "voltage_end_V": voltage_end,
        "voltage_drop_V": voltage_drop,
        "voltage_min_V": voltage_min,
        "voltage_max_V": voltage_max,
        "voltage_increase_fraction": voltage_increase_fraction,
        "temperature_mean_C": temperature_mean,
        "temperature_delta_C": temperature_delta,
        "is_clean_for_identification": bool(is_clean),
        "reject_reason": "OK" if is_clean else ";".join(reasons),
    }


def extract_discharge_segments(
    df,
    min_current_abs=0.05,
    min_points=100,
    min_duration_s=30.0,
    nominal_capacity_ah=3.0
):
    """
    从一个 DataFrame 中提取所有连续放电片段，并增加质量标签。

    返回：
    - all_segments：候选放电片段列表；
    - summary_df：每个片段的统计表和是否可用于第一阶段辨识的标签。
    """

    all_segments = []
    summary_rows = []

    if "test_id" in df.columns:
        group_iterator = df.groupby("test_id", sort=False, dropna=False)
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

            quality = assess_discharge_segment_quality(
                g,
                nominal_capacity_ah=nominal_capacity_ah
            )

            summary_row = {
                "global_segment_id": global_segment_id,
                "test_id": test_id,
                "mode": "discharge",
            }
            summary_row.update(quality)
            summary_rows.append(summary_row)

            all_segments.append(g)
            global_segment_id += 1

    if len(all_segments) == 0:
        return [], pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    # 第一优先级：干净；第二优先级：容量大；第三优先级：电流更恒定
    summary_df = summary_df.sort_values(
        ["is_clean_for_identification", "capacity_ah", "relative_std_current"],
        ascending=[False, False, True]
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
# 9. pseudo-OCV 表生成与调用
# ============================================================

PSEUDO_OCV_TABLE = None


def load_pseudo_ocv_table(csv_path):
    """
    读取 pseudo-OCV 表。要求至少包含 soc 与 ocv_v 两列。
    """
    global PSEUDO_OCV_TABLE

    if csv_path is None or not os.path.exists(csv_path):
        PSEUDO_OCV_TABLE = None
        return None

    table = pd.read_csv(csv_path)

    required = {"soc", "ocv_v"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"pseudo-OCV 文件缺少列: {missing}，当前列: {table.columns.tolist()}"
        )

    table = table[["soc", "ocv_v"]].copy()
    table["soc"] = pd.to_numeric(table["soc"], errors="coerce")
    table["ocv_v"] = pd.to_numeric(table["ocv_v"], errors="coerce")
    table = table.dropna()
    table = table.sort_values("soc").drop_duplicates("soc")

    if len(table) < 5:
        raise ValueError("pseudo-OCV 表有效点太少。")

    PSEUDO_OCV_TABLE = table
    print(f"✅ 已加载 pseudo-OCV 表: {csv_path}, 点数={len(table)}")
    return table


def U_cell_pseudo_ocv_from_soc(soc):
    """
    由 pseudo-OCV 表插值得到整电池平衡电压。
    """
    if PSEUDO_OCV_TABLE is None:
        if FALLBACK_TO_ANALYTIC_OCV:
            return U_cell_analytic_from_soc(soc)
        raise RuntimeError("尚未加载 pseudo-OCV 表。")

    soc = np.clip(soc, 0.0, 1.0)
    return np.interp(
        soc,
        PSEUDO_OCV_TABLE["soc"].to_numpy(dtype=float),
        PSEUDO_OCV_TABLE["ocv_v"].to_numpy(dtype=float),
    )


def build_pseudo_ocv_from_clean_segments(
    segment_records,
    manifest_df,
    output_dir
):
    """
    从第一步筛选出的干净放电片段构造 pseudo-OCV。

    说明：
    - 严格 OCV 需要准静态/充分静置实验；
    - 这里生成的是 pseudo-OCV，用于替换原来的 analytic 占位函数；
    - 如果没有 <= PSEUDO_OCV_MAX_C_RATE 的低倍率片段，会使用最低倍率干净片段，并在 source_note 中标明。
    """

    if not BUILD_PSEUDO_OCV:
        return None

    if manifest_df.empty:
        print("⚠️ manifest 为空，无法生成 pseudo-OCV。")
        return None

    clean = manifest_df[
        manifest_df["is_clean_for_identification"].astype(bool)
    ].copy()

    if clean.empty:
        print("⚠️ 没有通过清洁筛选的放电片段，无法生成 pseudo-OCV。")
        return None

    low_rate = clean[
        clean["estimated_C_rate"] <= PSEUDO_OCV_MAX_C_RATE
    ].copy()

    if low_rate.empty:
        candidates = clean.sort_values(
            ["estimated_C_rate", "capacity_ah"],
            ascending=[True, False]
        ).head(PSEUDO_OCV_NUM_SEGMENTS)
        source_note = (
            "lowest_rate_clean_segments_not_true_ocv"
        )
        print(
            "⚠️ 没有找到足够低倍率的干净放电片段；"
            "将使用最低倍率干净片段生成 pseudo-OCV。"
        )
    else:
        candidates = low_rate.sort_values(
            ["capacity_ah", "relative_std_current"],
            ascending=[False, True]
        ).head(PSEUDO_OCV_NUM_SEGMENTS)
        source_note = "low_rate_clean_segments"

    raw_rows = []

    for _, row in candidates.iterrows():
        key = (int(row["file_index"]), int(row["global_segment_id"]))

        if key not in segment_records:
            print(f"⚠️ 找不到片段记录: {key}，跳过。")
            continue

        seg = prepare_segment_for_simulation(segment_records[key])

        q_end = float(seg["capacity_ah"].iloc[-1])
        if q_end <= 0:
            continue

        soc = 1.0 - seg["capacity_ah"].to_numpy(dtype=float) / q_end
        current = seg["current_discharge"].to_numpy(dtype=float)
        voltage = seg["voltage"].to_numpy(dtype=float)

        # 放电端电压 = OCV - 极化。这里可选做最简单的 I*R 校正。
        ocv_like = voltage + current * PSEUDO_OCV_IR_CORRECTION_OHM

        one_raw = pd.DataFrame(
            {
                "soc": soc,
                "ocv_v": ocv_like,
                "voltage_measured_v": voltage,
                "current_a": current,
                "capacity_ah": seg["capacity_ah"].to_numpy(dtype=float),
                "file_index": int(row["file_index"]),
                "global_segment_id": int(row["global_segment_id"]),
                "estimated_C_rate": float(row["estimated_C_rate"]),
                "source_note": source_note,
            }
        )

        raw_rows.append(one_raw)

    if not raw_rows:
        print("⚠️ 没有可用于 pseudo-OCV 的原始点。")
        return None

    raw = pd.concat(raw_rows, ignore_index=True)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["soc", "ocv_v"]
    )
    raw = raw[
        (raw["soc"] >= 0.0)
        & (raw["soc"] <= 1.0)
        & (raw["ocv_v"] >= VOLTAGE_MIN_ALLOWED)
        & (raw["ocv_v"] <= VOLTAGE_MAX_ALLOWED)
    ].copy()

    if len(raw) < 10:
        print("⚠️ pseudo-OCV 原始有效点太少。")
        return None

    # 分箱取中位数，减弱测量噪声和不同片段差异。
    bins = np.linspace(0.0, 1.0, PSEUDO_OCV_GRID_POINTS + 1)
    raw["soc_bin"] = pd.cut(
        raw["soc"],
        bins=bins,
        include_lowest=True,
        labels=False
    )

    binned = (
        raw.groupby("soc_bin", dropna=True)
        .agg(
            soc=("soc", "median"),
            ocv_v=("ocv_v", "median"),
            n_points=("ocv_v", "size"),
            estimated_C_rate=("estimated_C_rate", "median")
        )
        .dropna()
        .reset_index(drop=True)
        .sort_values("soc")
    )

    binned = binned.drop_duplicates("soc")

    if len(binned) < 5:
        print("⚠️ pseudo-OCV 分箱后有效点太少。")
        return None

    soc_grid = np.linspace(0.0, 1.0, PSEUDO_OCV_GRID_POINTS)
    ocv_grid = np.interp(
        soc_grid,
        binned["soc"].to_numpy(dtype=float),
        binned["ocv_v"].to_numpy(dtype=float)
    )

    ocv_df = pd.DataFrame(
        {
            "soc": soc_grid,
            "ocv_v": ocv_grid,
            "source_note": source_note,
            "ir_correction_ohm": PSEUDO_OCV_IR_CORRECTION_OHM,
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, PSEUDO_OCV_RAW_FILENAME)
    ocv_path = os.path.join(output_dir, PSEUDO_OCV_FILENAME)
    plot_path = os.path.join(output_dir, PSEUDO_OCV_PLOT_FILENAME)

    raw.to_csv(raw_path, index=False)
    ocv_df.to_csv(ocv_path, index=False)

    print(f"✅ 已保存 pseudo-OCV 原始点: {raw_path}")
    print(f"✅ 已保存 pseudo-OCV 表: {ocv_path}")

    plt.figure(figsize=(8, 5))
    plt.scatter(raw["soc"], raw["ocv_v"], s=4, alpha=0.25, label="raw pseudo-OCV points")
    plt.plot(ocv_df["soc"], ocv_df["ocv_v"], linewidth=2.0, label="binned/interpolated pseudo-OCV")
    plt.xlabel("SOC")
    plt.ylabel("Pseudo-OCV / V")
    plt.title("Pseudo-OCV from clean discharge segments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 已保存 pseudo-OCV 图像: {plot_path}")

    load_pseudo_ocv_table(ocv_path)
    return ocv_path



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

    if ocp_mode == "pseudo_ocv":
        # 使用第二步生成的整电池 pseudo-OCV 表
        soc_eff = np.clip(cbar_n / p.c_smax_n, 1e-4, 0.9999)
        U_eq = U_cell_pseudo_ocv_from_soc(soc_eff)

    elif ocp_mode == "fullcell_analytic":
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
    
    # 默认不显示图形，避免片段多时一个窗口一个窗口弹出。
    if DISPLAY_PLOTS:
        plt.show()
    plt.close(fig)


# ============================================================
# 12.5 从 qOCV 文件夹构造 pseudo-OCV
# ============================================================

def build_pseudo_ocv_from_single_qocv_segment(seg, output_dir, source_meta):
    """
    从一个选中的 qOCV/低倍率放电片段构造单值 pseudo-OCV。
    关键修正：
    - 不再把多个不同循环/不同 SOC 窗口的放电片段硬合并；
    - 只用一个最可信的 qOCV 片段生成 OCV-SOC 曲线；
    - 避免出现多分支 OCV 曲线。
    """
    seg = prepare_segment_for_simulation(seg)

    q = seg["capacity_ah"].to_numpy(dtype=float)
    v_meas = seg["voltage"].to_numpy(dtype=float)
    current = seg["current_discharge"].to_numpy(dtype=float)

    q_end = float(q[-1])
    if q_end <= 0:
        raise ValueError("选中的 qOCV 片段容量为 0，无法构造 pseudo-OCV。")

    # 对完整低倍率放电：起点近似 SOC=1，终点近似 SOC=0。
    # 如果 qOCV 不是完整 SOC 窗口，这仍然只是 local SOC，需要在 manifest 里检查容量覆盖。
    soc = 1.0 - q / q_end

    # 简单 IR 校正：放电端电压约等于 OCV - I*R，故 OCV_like = V + I*R。
    ocv_like = v_meas + current * PSEUDO_OCV_IR_CORRECTION_OHM

    raw = pd.DataFrame(
        {
            "soc": soc,
            "ocv_v": ocv_like,
            "voltage_measured_v": v_meas,
            "current_a": current,
            "capacity_ah": q,
        }
    )
    for k, value in source_meta.items():
        raw[k] = value

    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["soc", "ocv_v"])
    raw = raw[
        (raw["soc"] >= 0.0)
        & (raw["soc"] <= 1.0)
        & (raw["ocv_v"] >= VOLTAGE_MIN_ALLOWED)
        & (raw["ocv_v"] <= VOLTAGE_MAX_ALLOWED)
    ].copy()

    if len(raw) < 10:
        raise ValueError("qOCV 原始有效点太少，无法生成 pseudo-OCV。")

    raw = raw.sort_values("soc").drop_duplicates("soc", keep="first").reset_index(drop=True)

    soc_grid = np.linspace(0.0, 1.0, PSEUDO_OCV_GRID_POINTS)
    ocv_grid = np.interp(
        soc_grid,
        raw["soc"].to_numpy(dtype=float),
        raw["ocv_v"].to_numpy(dtype=float),
    )

    # 轻微平滑，避免测量噪声进入后续参数辨识。
    ocv_series = pd.Series(ocv_grid)
    window = max(7, int(PSEUDO_OCV_GRID_POINTS * 0.015))
    if window % 2 == 0:
        window += 1
    ocv_grid = (
        ocv_series
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )

    if QOCV_ENFORCE_MONOTONIC_OCV:
        # NCA/石墨整电池 OCV 随 SOC 总体应上升。这里防止噪声造成局部反斜率。
        ocv_grid = np.maximum.accumulate(ocv_grid)

    ocv_df = pd.DataFrame(
        {
            "soc": soc_grid,
            "ocv_v": ocv_grid,
            "source_note": "single_selected_qocv_segment",
            "ir_correction_ohm": PSEUDO_OCV_IR_CORRECTION_OHM,
        }
    )
    for k, value in source_meta.items():
        ocv_df[k] = value

    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, PSEUDO_OCV_RAW_FILENAME)
    ocv_path = os.path.join(output_dir, PSEUDO_OCV_FILENAME)
    plot_path = os.path.join(output_dir, PSEUDO_OCV_PLOT_FILENAME)

    raw.to_csv(raw_path, index=False)
    ocv_df.to_csv(ocv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(raw["soc"], raw["ocv_v"], s=5, alpha=0.25, label="selected qOCV raw points")
    ax.plot(ocv_df["soc"], ocv_df["ocv_v"], linewidth=2.0, label="single-segment pseudo-OCV")
    ax.set_xlabel("SOC")
    ax.set_ylabel("Pseudo-OCV / V")
    ax.set_title("Pseudo-OCV from selected qOCV segment")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ 已保存 qOCV 原始点: {raw_path}")
    print(f"✅ 已保存 pseudo-OCV 表: {ocv_path}")
    print(f"✅ 已静默保存 pseudo-OCV 图像: {plot_path}")

    load_pseudo_ocv_table(ocv_path)
    return ocv_path


def build_pseudo_ocv_from_qocv_folder(output_dir):
    """
    从用户给定的 qOCV_DATA_PATH 中选择一个最适合的低倍率放电片段，
    并生成单值 pseudo-OCV 表。

    选择逻辑：
    1. 必须是 clean discharge；
    2. 优先 C-rate <= QOCV_SELECT_MAX_C_RATE；
    3. 优先容量覆盖 >= QOCV_MIN_CAPACITY_AH；
    4. 若没有完全满足，会退而选择最低倍率 + 最大容量的片段，但会在日志里提醒。
    """
    if QOCV_DATA_PATH is None:
        print("⚠️ QOCV_DATA_PATH=None，跳过 qOCV pseudo-OCV 生成。")
        return None

    qocv_files = list_parquet_files(
        QOCV_DATA_PATH,
        use_s3=QOCV_USE_S3,
        max_files=QOCV_MAX_FILES,
        file_name_contains=QOCV_FILE_NAME_CONTAINS,
        file_name_excludes=QOCV_FILE_NAME_EXCLUDES,
    )

    candidate_rows = []
    candidate_segments = {}

    for qfile_index, qfile in enumerate(qocv_files):
        try:
            df_raw = read_one_parquet(qfile, use_s3=QOCV_USE_S3)
            df = preprocess_dataframe(df_raw)

            segments, summary = extract_discharge_segments(
                df,
                min_current_abs=MIN_CURRENT_ABS,
                min_points=MIN_POINTS,
                min_duration_s=MIN_DURATION_S,
                nominal_capacity_ah=NOMINAL_CAPACITY_AH,
            )

            if len(segments) == 0:
                continue

            summary = summary.copy()
            summary["qocv_file_index"] = qfile_index
            summary["qocv_file_path"] = qfile

            for seg in segments:
                sid = int(seg["global_segment_id"].iloc[0])
                candidate_segments[(qfile_index, sid)] = seg

            candidate_rows.append(summary)

        except Exception as e:
            print(f"⚠️ qOCV 文件处理失败，已跳过: {qfile}")
            print(f"   错误: {e}")

    if not candidate_rows:
        print("⚠️ qOCV 文件夹中没有可用放电片段。")
        return None

    qmanifest = pd.concat(candidate_rows, ignore_index=True, sort=False)
    qmanifest_path = os.path.join(output_dir, "qocv_discharge_segment_manifest.csv")
    qmanifest.to_csv(qmanifest_path, index=False)
    print(f"✅ qOCV 片段 manifest 已保存: {qmanifest_path}")

    usable = qmanifest[
        qmanifest["is_clean_for_identification"].fillna(False).astype(bool)
    ].copy()

    if usable.empty:
        print("⚠️ qOCV 文件夹中没有通过 clean 筛选的放电片段。")
        return None

    strict = usable[
        (usable["estimated_C_rate"] <= QOCV_SELECT_MAX_C_RATE)
        & (usable["capacity_ah"] >= QOCV_MIN_CAPACITY_AH)
    ].copy()

    if strict.empty:
        print(
            "⚠️ 没有同时满足低倍率和容量覆盖的 qOCV 片段。"
            "将退而选择最低倍率、容量最大的 clean 片段；生成结果只能作为 pseudo-OCV。"
        )
        selected = (
            usable.sort_values(
                ["estimated_C_rate", "capacity_ah", "relative_std_current"],
                ascending=[True, False, True],
            )
            .iloc[0]
        )
        selection_note = "fallback_lowest_rate_clean_segment"
    else:
        selected = (
            strict.sort_values(
                ["capacity_ah", "estimated_C_rate", "relative_std_current"],
                ascending=[False, True, True],
            )
            .iloc[0]
        )
        selection_note = "strict_low_rate_high_capacity_qocv_segment"

    qfile_index = int(selected["qocv_file_index"])
    sid = int(selected["global_segment_id"])
    key = (qfile_index, sid)

    if key not in candidate_segments:
        print(f"⚠️ 找不到 qOCV 片段数据: {key}")
        return None

    source_meta = {
        "selection_note": selection_note,
        "qocv_file_index": qfile_index,
        "global_segment_id": sid,
        "qocv_file_path": selected["qocv_file_path"],
        "selected_capacity_ah": float(selected["capacity_ah"]),
        "selected_estimated_C_rate": float(selected["estimated_C_rate"]),
        "selected_relative_std_current": float(selected["relative_std_current"]),
    }

    print("\n✅ 选中的 qOCV 片段:")
    print(f"   文件: {selected['qocv_file_path']}")
    print(f"   segment_id: {sid}")
    print(f"   容量: {float(selected['capacity_ah']):.4f} Ah")
    print(f"   估计倍率: {float(selected['estimated_C_rate']):.4f} C")
    print(f"   电流相对标准差: {float(selected['relative_std_current']):.6f}")
    print(f"   选择说明: {selection_note}")

    return build_pseudo_ocv_from_single_qocv_segment(
        candidate_segments[key],
        output_dir=output_dir,
        source_meta=source_meta,
    )


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

    # 第一遍：只做数据读取、预处理、放电片段提取与质量筛选。
    # 这样可以先形成第一步所需的 clean segment manifest，再进入 pseudo-OCV 与前向仿真。
    all_manifest_rows = []
    segment_records = {}

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
                all_manifest_rows.append(
                    {
                        "file_index": file_index,
                        "file_path": parquet_file,
                        "error": "no_valid_discharge_segments",
                    }
                )
                continue

            discharge_summary = discharge_summary.copy()
            discharge_summary["file_index"] = file_index
            discharge_summary["file_path"] = parquet_file

            print("\n📋 当前文件放电片段统计（前10行，已优先显示干净片段）:")
            display_cols = [
                "global_segment_id",
                "is_clean_for_identification",
                "reject_reason",
                "capacity_ah",
                "mean_current_A",
                "relative_std_current",
                "estimated_C_rate",
                "voltage_drop_V",
                "temperature_delta_C",
            ]
            print(discharge_summary[display_cols].head(10))

            all_manifest_rows.append(discharge_summary)

            for seg in discharge_segments:
                segment_id = int(seg["global_segment_id"].iloc[0])
                segment_records[(file_index, segment_id)] = seg

        except Exception as e:
            print("\n❌ 当前文件处理失败:")
            print(f"   文件: {parquet_file}")
            print(f"   错误: {e}")

            all_manifest_rows.append(
                pd.DataFrame(
                    [
                        {
                            "file_index": file_index,
                            "file_path": parquet_file,
                            "error": str(e),
                        }
                    ]
                )
            )

    if not all_manifest_rows:
        raise RuntimeError("没有任何文件被成功处理，无法继续。")

    manifest_parts = []
    for item in all_manifest_rows:
        if isinstance(item, pd.DataFrame):
            manifest_parts.append(item)
        else:
            manifest_parts.append(pd.DataFrame([item]))

    segment_manifest = pd.concat(
        manifest_parts,
        ignore_index=True,
        sort=False
    )

    manifest_path = os.path.join(
        OUTPUT_DIR,
        "clean_discharge_segment_manifest.csv"
    )
    segment_manifest.to_csv(manifest_path, index=False)

    print("\n✅ 第一步输出：干净恒流放电片段 manifest 已保存。")
    print(f"   {manifest_path}")

    if "is_clean_for_identification" in segment_manifest.columns:
        clean_count = int(
            segment_manifest["is_clean_for_identification"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        total_count = int(len(segment_manifest))
        print(f"   候选片段数: {total_count}")
        print(f"   干净片段数: {clean_count}")

        if clean_count == 0:
            print(
                "⚠️ 没有片段通过全部清洁筛选。"
                "请先查看 reject_reason，适当放宽阈值或换数据文件。"
            )

    # 第二步：从 qOCV 文件夹构造 NCA/VTC6 pseudo-OCV 表并加载。
    # 重要：不要把 ageing 中多个不同电压平台/不同 SOC 窗口的放电片段合并成 OCV，
    # 否则会得到多分支曲线。
    pseudo_ocv_path = build_pseudo_ocv_from_qocv_folder(
        output_dir=OUTPUT_DIR
    )

    if pseudo_ocv_path is None and OCP_MODE == "pseudo_ocv":
        if FALLBACK_TO_ANALYTIC_OCV:
            print(
                "⚠️ pseudo-OCV 未生成，当前将临时退回 analytic OCV。"
                "这只适合流程调试，不适合正式参数辨识。"
            )
        else:
            raise RuntimeError(
                "OCP_MODE='pseudo_ocv'，但 pseudo-OCV 生成失败。"
            )

    # 第三步：只对干净片段做前向仿真，作为后续参数辨识前的验证输入。
    batch_summary_rows = []

    if "is_clean_for_identification" in segment_manifest.columns:
        simulation_manifest = segment_manifest[
            segment_manifest["is_clean_for_identification"]
            .fillna(False)
            .astype(bool)
        ].copy()
    else:
        simulation_manifest = pd.DataFrame()

    if simulation_manifest.empty:
        print("⚠️ 没有干净片段可仿真，本次只完成 manifest 与 pseudo-OCV 尝试。")
    else:
        simulation_manifest = simulation_manifest.sort_values(
            ["file_index", "capacity_ah"],
            ascending=[True, False]
        )

        if MAX_DISCHARGE_SEGMENTS_PER_FILE is not None:
            simulation_manifest = (
                simulation_manifest
                .groupby("file_index", group_keys=False)
                .head(MAX_DISCHARGE_SEGMENTS_PER_FILE)
            )

        for _, row in simulation_manifest.iterrows():

            file_index = int(row["file_index"])
            segment_id = int(row["global_segment_id"])
            parquet_file = row["file_path"]

            key = (file_index, segment_id)

            if key not in segment_records:
                print(f"⚠️ 找不到片段 {key} 的数据，跳过仿真。")
                continue

            seg = segment_records[key]

            try:
                print("\n🚀 开始仿真干净放电片段:")
                print(f"   文件编号: {file_index}")
                print(f"   segment_id: {segment_id}")
                print(f"   容量: {row['capacity_ah']:.4f} Ah")
                print(f"   平均电流: {row['mean_current_A']:.4f} A")
                print(f"   估计倍率: {row['estimated_C_rate']:.4f} C")
                print(f"   OCP_MODE: {OCP_MODE}")

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
                        "voltage_drop_V": row["voltage_drop_V"],
                        "relative_std_current": row["relative_std_current"],
                        "rmse_V": rmse,
                        "mae_V": mae,
                        "ocp_mode": OCP_MODE,
                        "pseudo_ocv_path": pseudo_ocv_path,
                    }
                )

            except Exception as e:
                print("\n❌ 当前片段仿真失败:")
                print(f"   文件: {parquet_file}")
                print(f"   segment_id: {segment_id}")
                print(f"   错误: {e}")

                batch_summary_rows.append(
                    {
                        "file_index": file_index,
                        "file_path": parquet_file,
                        "segment_id": segment_id,
                        "error": str(e),
                    }
                )

    batch_summary = pd.DataFrame(batch_summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "batch_p2d_summary.csv"
    )

    batch_summary.to_csv(summary_path, index=False)

    print("\n🎉 完成：第一步干净恒流片段筛选 + 第二步 qOCV pseudo-OCV 生成 + P2D 前向验证。")
    print(f"第一步 manifest: {manifest_path}")
    print(f"第二步 pseudo-OCV: {pseudo_ocv_path}")
    print(f"前向仿真汇总: {summary_path}")
