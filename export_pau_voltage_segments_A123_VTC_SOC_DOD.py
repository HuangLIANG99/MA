# -*- coding: utf-8 -*-
"""
export_pau_voltage_segments.py

功能：
    读取 Aging 循环 parquet 文件（支持本地路径 / MinIO-S3 在线路径），
    按照 "CHA -> PAU -> DCH" 状态识别 + 电流二次校验逻辑，找出每个文件中
    所有真正可用于二阶 RC 拟合的静置(PAU)片段，然后把 PAU 片段以及
    **PAU 开始前若干个采样点（默认 10 点）** 的电压、电流、温度原始数据
    导出为 CSV，不做任何 RC 参数拟合，只把原始曲线数据落盘，方便人工检查 /
    画图核对静置弛豫曲线。

    电池编号可从输入路径 / 文件名自动识别，目前支持：
        - A123，例如 METABatt_A123_APR18650M1B_007 -> A123_007
        - VTC6，例如 METABatt_Sony_Murata_18650VTC6_006 -> VTC6_006

    本脚本为自包含版本：不依赖 fit_RC_interpo-soh_VTC6.py，
    可以单独运行；其中的文件筛选 / S3 在线读取 / PAU 片段识别逻辑
    均与原脚本保持一致。

在线读取（S3 / MinIO）说明：
    当 CYCLE_FILES_DIR 配置为 "s3://..." 开头的路径，或者本地路径不存在时，
    脚本会自动改用 s3fs 通过 S3 协议在线读取。
    需要提前设置好环境变量：
        MINIO_ACCESS_KEY
        MINIO_SECRET_KEY
    连接的 endpoint 默认使用 https://iseadocker.isea.rwth-aachen.de:9000，
    如需修改，请调整 get_s3_storage_options() 中的 endpoint_url。

输出：
    默认情况下（EXPORT_ONE_CSV_PER_SEGMENT = False）：
        每个 Aging 循环文件对应导出 **一个** CSV，内含该文件里所有有效 PAU 片段的
        逐点数据，用 segment_id 区分不同片段。
    如果设置 EXPORT_ONE_CSV_PER_SEGMENT = True：
        每一个 PAU 片段单独导出一个 CSV 文件。

    输出目录结构：
        <OUTPUT_DIR>/<battery_id>/
        <battery_id>_Aging<序号>_<YYYY-MM-DD_HHMMSS>_<file_hash>_pau_voltage_segments.csv

    例如：
        A123_007_Aging001_2025-03-31_172218_1e492850_pau_voltage_segments.csv
        A123_007_Aging002_2025-04-02_091530_ab12cd34_pau_voltage_segments.csv

    Aging001 / Aging002 / ... 是同一电池按源文件时间戳升序排列后的顺序。

    单片段模式：
        <battery_id>_Aging<序号>_<YYYY-MM-DD_HHMMSS>_<file_hash>_seg<N>_pau_voltage.csv

CSV 列说明：
    battery_id                 电池编号，如 A123_007
    cycle_file                  来源的 Aging parquet 文件名
    segment_id                   该文件内第几个有效 PAU 片段（从 1 开始）
    point_index                    相对 PAU 起点的采样点编号：
                                   PAU 前置点为负数（如 -10...-1），PAU 起点为 0
    data_region                    "pre_pau" 或 "pau"
    source_row_index               原始 parquet DataFrame 中的行号
    time_s_from_seg_start          相对 PAU 起点的时间 (s)，前置点为负值
    time_s_from_file_start         相对整个循环文件起点的时间 (s)
    voltage_V                       电压 (V)
    current_A                       电流 (A)
    temperature_C                    温度 (°C)
    segment_duration_s               该片段总时长 (s)
    trimmed_by_current                该片段是否因 Zustand 切换滞后、被电流二次识别截断
    max_abs_current_A_in_seg          该片段内电流绝对值的最大值 (A)
"""

import os
import re
import hashlib
import warnings

import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 0. 用户配置区
# ============================================================
# 电池型号由文件名 / 路径自动识别，目前支持 A123 和 VTC6。
# BATTERY_NUMBER_RANGE 仅限制末尾数字编号，例如 001 到 300
BATTERY_NUMBER_RANGE = (151, 200)

# 如果只想处理指定电池，例如 [6, 12, 20]，则设置这里
# 若不为 None，会优先使用 BATTERY_NUMBERS
BATTERY_NUMBERS = None

# 循环文件所在目录/前缀。建议指向 Metabatt 根目录，让程序递归搜索 A123 / VTC。
# 可以是本地路径，也可以是 "s3://桶名/前缀" 形式的在线路径。
# 示例文件:
#   projects/j8005-metabatt/Metabatt/A123/METABatt_A123_APR18650M1B_007/
#   J8005_BMWK_METABatt=METABatt_A123_APR18650M1B_007=2025-03-31_172218=jri_Aging_APR_Cyc_35grad_70SOC_60DOD_3C=...parquet
CYCLE_FILES_DIR = "projects/j8005-metabatt/Metabatt/A123"
AGING_KEYWORD = "Aging"
CYCLE_FILE_EXTENSIONS = (".parquet",)

CSV_SEPARATOR = ";"
CSV_DECIMAL = ","

COL_TIME = "Zeit"
COL_CURRENT = "Strom"
COL_VOLTAGE = "Spannung"
COL_STATE = "Zustand"
COL_TEMPERATURE = "T1"

MIN_PAU_DURATION_S = 800

# ========== PAU 静置电流二次识别 ==========
ENABLE_PAU_CURRENT_CHECK = True
PAU_ZERO_CURRENT_TOLERANCE_A = 1e-4
# "trim"   -> 在第一个非零/无效电流点之前截断 PAU（推荐）
# "reject" -> 整个候选 PAU 直接丢弃
PAU_CURRENT_INVALID_ACTION = "trim"

# 每个 PAU 片段额外导出其开始之前的采样点数量。
# 如果 PAU 离文件开头不足这么多点，则导出实际可用的全部前置点。
PRE_PAU_POINTS = 10

# 导出结果存放的根目录
OUTPUT_DIR = "output_pau_voltage_segments"

# False（默认）：每个 Aging 文件的所有有效 PAU 片段合并导出成一个 CSV（长表，用 segment_id 区分）
# True         ：每个 PAU 片段单独导出一个 CSV 文件
EXPORT_ONE_CSV_PER_SEGMENT = False


# ============================================================
# 1. MinIO / S3 / 本地读取
# ============================================================

def get_s3_storage_options():
    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")

    if key is None or secret is None:
        raise ValueError("未找到 MINIO_ACCESS_KEY 或 MINIO_SECRET_KEY。")

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


def is_s3_path(path: str):
    return str(path).startswith("s3://")


def get_s3_filesystem():
    return s3fs.S3FileSystem(**get_s3_storage_options())


def list_files_from_folder(folder, extensions=None, keyword=None):
    """列出目录下的文件，自动支持：本地文件/本地目录/S3 在线路径。"""
    folder = str(folder)

    if os.path.isfile(folder):
        files = [folder]

    elif os.path.isdir(folder):
        files = []
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                files.append(os.path.join(root, filename))

    else:
        # 本地不存在该路径，按 S3 在线路径处理
        fs = get_s3_filesystem()
        s3_prefix = folder.replace("s3://", "").rstrip("/")
        files = [f"s3://{path}" for path in fs.find(s3_prefix)]

    if extensions is not None:
        extensions = tuple(ext.lower() for ext in extensions)
        files = [
            file for file in files
            if os.path.basename(str(file)).lower().endswith(extensions)
        ]

    if keyword is not None:
        files = [
            file for file in files
            if keyword.lower() in os.path.basename(str(file)).lower()
        ]

    return sorted(files)


def read_battery_data(target_file: str):
    """读取单个循环 parquet 文件，自动支持本地路径 / S3 在线路径。"""
    if is_s3_path(target_file):
        return pd.read_parquet(
            target_file,
            storage_options=get_s3_storage_options(),
        )

    return pd.read_parquet(target_file)


# ============================================================
# 2. 通用工具
# ============================================================

def safe_filename(text: str, max_len=160):
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(text))
    text = re.sub(r"\s+", "_", text)
    return text[:max_len].strip("._ ")


# 支持从输入路径 / 文件名自动识别的电池系列。
# 顺序很重要：如果未来增加名称有包含关系的型号，应把更具体的型号放前面。
BATTERY_ID_PATTERNS = [
    (
        "A123",
        re.compile(
            r"A123[^=/\\]*?_(\d{1,4})(?=[=/\\_]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "VTC6",
        re.compile(
            r"VTC6[^=/\\]*?_(\d{1,4})(?=[=/\\_]|$)",
            re.IGNORECASE,
        ),
    ),
]

SUPPORTED_BATTERY_PREFIXES = tuple(prefix for prefix, _ in BATTERY_ID_PATTERNS)


def extract_battery_id(filepath: str):
    """
    从文件名或完整路径中自动提取电池 ID。

    示例：
        METABatt_A123_APR18650M1B_007 -> A123_007
        METABatt_Sony_Murata_18650VTC6_006 -> VTC6_006
    """
    texts = [
        os.path.basename(str(filepath)),
        str(filepath),
    ]

    for text in texts:
        for prefix, pattern in BATTERY_ID_PATTERNS:
            match = pattern.search(text)
            if match:
                return f"{prefix}_{int(match.group(1)):03d}"

    return None


# 源 Aging 文件时间戳，例如：
# =2025-03-31_172218=
# =2024-09-09_110001=
AGING_DATETIME_PATTERN = re.compile(
    r"=(\d{4}-\d{2}-\d{2}_\d{6})=",
    re.IGNORECASE,
)


def extract_aging_datetime(filepath: str):
    """
    从 Aging parquet 文件名中提取测量/文件时间。

    例如：
        ...=2025-03-31_172218=jri_Aging_...
        -> Timestamp('2025-03-31 17:22:18')

    如果无法识别则返回 pd.NaT。
    """
    filename = os.path.basename(str(filepath))
    match = AGING_DATETIME_PATTERN.search(filename)

    if not match:
        return pd.NaT

    return pd.to_datetime(
        match.group(1),
        format="%Y-%m-%d_%H%M%S",
        errors="coerce",
    )


def get_aging_sort_key(filepath: str):
    """
    用源文件名中的 YYYY-MM-DD_HHMMSS 对 Aging 文件按时间升序排序。
    无法提取时间的文件排到最后。
    """
    aging_dt = extract_aging_datetime(filepath)

    if pd.isna(aging_dt):
        return (1, pd.Timestamp.max, str(filepath))

    return (0, aging_dt, str(filepath))


def format_aging_datetime_for_filename(filepath: str):
    """返回适合放入输出文件名的 Aging 时间字符串。"""
    aging_dt = extract_aging_datetime(filepath)

    if pd.isna(aging_dt):
        return "unknown_time"

    return aging_dt.strftime("%Y-%m-%d_%H%M%S")


# 源 Aging 文件名中的 SOC / DOD，例如：
# ..._70SOC_60DOD_...
SOC_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*SOC(?=[^A-Za-z0-9]|$)", re.IGNORECASE)
DOD_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*DOD(?=[^A-Za-z0-9]|$)", re.IGNORECASE)


def _format_level_for_filename(value_text: str):
    """把 SOC / DOD 数值整理成适合文件名的形式，例如 70.0 -> 70。"""
    value = float(value_text)

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


def extract_soc_dod_for_filename(filepath: str):
    """
    从源 Aging 文件名中提取 SOC 和 DOD，并返回用于输出文件名的字符串。

    例如：
        ..._70SOC_60DOD_...
        -> "70SOC_60DOD"

    若个别源文件无法识别，则使用 unknownSOC / unknownDOD，
    不影响原有 CSV 数据内容与导出逻辑。
    """
    filename = os.path.basename(str(filepath))

    soc_match = SOC_PATTERN.search(filename)
    dod_match = DOD_PATTERN.search(filename)

    soc_text = (
        f"{_format_level_for_filename(soc_match.group(1))}SOC"
        if soc_match
        else "unknownSOC"
    )
    dod_text = (
        f"{_format_level_for_filename(dod_match.group(1))}DOD"
        if dod_match
        else "unknownDOD"
    )

    return f"{soc_text}_{dod_text}"


def get_allowed_battery_numbers():
    """
    返回允许处理的电池数字编号。

    例如：
        BATTERY_NUMBERS = [6, 7, 20]
        -> {6, 7, 20}

    或：
        BATTERY_NUMBER_RANGE = (1, 300)
        -> {1, 2, ..., 300}

    注意：
        这里仅限制末尾数字编号，不限制 A123 / VTC6 型号。
        电池型号由 extract_battery_id() 从实际文件名 / 路径中自动识别。
    """
    if BATTERY_NUMBERS is not None:
        if isinstance(BATTERY_NUMBERS, (int, np.integer)):
            return {int(BATTERY_NUMBERS)}

        return {int(number) for number in BATTERY_NUMBERS}

    if BATTERY_NUMBER_RANGE is not None:
        start_number, end_number = BATTERY_NUMBER_RANGE
        return set(
            range(
                int(start_number),
                int(end_number) + 1,
            )
        )

    return None


def filter_files_by_allowed_batteries(
    files,
    allowed_battery_numbers,
    file_label,
):
    """
    按数字编号筛选文件。

    型号由 extract_battery_id() 自动识别，例如：
        A123_007 -> 数字编号 7
        VTC6_006 -> 数字编号 6

    BATTERY_NUMBERS / BATTERY_NUMBER_RANGE 仅对数字编号生效。
    """
    filtered_files = []
    unknown_count = 0
    out_of_range_count = 0

    for file in files:
        battery_id = extract_battery_id(file)

        if battery_id is None:
            unknown_count += 1
            continue

        try:
            battery_number = int(
                battery_id.rsplit("_", 1)[1]
            )
        except (IndexError, ValueError):
            unknown_count += 1
            continue

        if (
            allowed_battery_numbers is not None
            and battery_number not in allowed_battery_numbers
        ):
            out_of_range_count += 1
            continue

        filtered_files.append(file)

    if unknown_count:
        print(
            f"⚠️ {file_label} 中有 {unknown_count} 个文件"
            f"无法识别电池编号，已跳过。"
        )

    if out_of_range_count:
        print(
            f"ℹ️ {file_label} 中有 {out_of_range_count} 个文件"
            f"不在编号筛选范围内，已跳过。"
        )

    return sorted(filtered_files)


def group_files_by_battery(files):
    grouped = {}

    for file in files:
        battery_id = extract_battery_id(file)

        if battery_id is None:
            continue

        grouped.setdefault(battery_id, []).append(file)

    for battery_id in grouped:
        grouped[battery_id] = sorted(
            grouped[battery_id],
            key=get_aging_sort_key,
        )

    return grouped


def parse_numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


# ============================================================
# 3. 循环文件预处理
# ============================================================

def add_time_seconds(df, time_col):
    series = df[time_col]

    if pd.api.types.is_datetime64_any_dtype(series):
        time_values = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
        time_zero = time_values.dropna().iloc[0]
        df["Time_Seconds"] = (time_values - time_zero).dt.total_seconds()
        return df

    if pd.api.types.is_timedelta64_dtype(series):
        timedelta_values = pd.to_timedelta(series, errors="coerce")
        time_zero = timedelta_values.dropna().iloc[0]
        df["Time_Seconds"] = (timedelta_values - time_zero).dt.total_seconds()
        return df

    numeric_time = parse_numeric_series(series)
    min_valid = max(1, int(len(series) * 0.8))

    if numeric_time.notna().sum() >= min_valid:
        time_zero = numeric_time.dropna().iloc[0]
        df["Time_Seconds"] = numeric_time - time_zero
        return df

    timedelta_values = pd.to_timedelta(series, errors="coerce")

    if timedelta_values.notna().sum() >= min_valid:
        time_zero = timedelta_values.dropna().iloc[0]
        df["Time_Seconds"] = (timedelta_values - time_zero).dt.total_seconds()
        return df

    datetime_values = pd.to_datetime(series, errors="coerce", utc=True)

    if datetime_values.notna().sum() >= min_valid:
        time_zero = datetime_values.dropna().iloc[0]
        df["Time_Seconds"] = (datetime_values - time_zero).dt.total_seconds()
        return df

    raise ValueError(f"无法解析时间列: {time_col}")


def load_and_prepare_cycle_dataframe(cycle_file):
    df = read_battery_data(cycle_file).reset_index(drop=True)
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = [
        COL_TIME,
        COL_CURRENT,
        COL_VOLTAGE,
        COL_STATE,
        COL_TEMPERATURE,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"循环文件缺少必要列: {missing_cols}")

    df = add_time_seconds(df, COL_TIME)

    df["Zustand_Clean"] = (
        df[COL_STATE]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["_Voltage_Numeric"] = parse_numeric_series(df[COL_VOLTAGE])
    df["_Current_Numeric"] = parse_numeric_series(df[COL_CURRENT])
    df["_Temperature_Numeric"] = parse_numeric_series(df[COL_TEMPERATURE])

    return df


def find_valid_pau_segments(df):
    """
    查找用于电压曲线导出（以及原脚本中的 RC 拟合）的真实静置片段。

    第一层使用状态逻辑：
        CHA -> PAU -> ... -> DCH
    且在找到 DCH 前如果重新进入 CHA，则候选片段作废。

    第二层增加电流验证：
        最终保留的每一个 PAU 数据点都必须满足
        abs(Strom) <= PAU_ZERO_CURRENT_TOLERANCE_A。

    如果 Zustand 仍错误地标记为 PAU，但实际电流已经离开 0：
        - PAU_CURRENT_INVALID_ACTION == "trim":
          在第一个非零/无效电流点之前结束静置片段；
        - PAU_CURRENT_INVALID_ACTION == "reject":
          直接丢弃整个候选片段。

    无论是否因电流提前截断，真实静置持续时间仍必须 > MIN_PAU_DURATION_S。
    """
    df_temp = df[["Time_Seconds", "Zustand_Clean"]].copy()
    df_temp["prev_state"] = df_temp["Zustand_Clean"].shift()

    change_points = (
        df_temp[df_temp["Zustand_Clean"] != df_temp["prev_state"]]
        .copy()
        .reset_index()
    )

    valid_segments = []

    if ENABLE_PAU_CURRENT_CHECK:
        if "_Current_Numeric" not in df.columns:
            raise ValueError(
                "已启用 PAU 电流验证，但数据中不存在 _Current_Numeric 列。"
            )

        current_action = str(PAU_CURRENT_INVALID_ACTION).strip().lower()
        if current_action not in {"trim", "reject"}:
            raise ValueError(
                "PAU_CURRENT_INVALID_ACTION 只能设置为 'trim' 或 'reject'"
            )

        current_tol = float(PAU_ZERO_CURRENT_TOLERANCE_A)
        if not np.isfinite(current_tol) or current_tol < 0:
            raise ValueError(
                "PAU_ZERO_CURRENT_TOLERANCE_A 必须是 >= 0 的有限数值"
            )

    for k in range(len(change_points)):
        is_cha_to_pau = (
            change_points.loc[k, "prev_state"] == "CHA"
            and change_points.loc[k, "Zustand_Clean"] == "PAU"
        )

        if not is_cha_to_pau:
            continue

        start_df_index = int(change_points.loc[k, "index"])

        for look_ahead in range(k + 1, len(change_points)):
            next_state = change_points.loc[look_ahead, "Zustand_Clean"]

            if next_state == "DCH":
                dch_start_index = int(change_points.loc[look_ahead, "index"])
                status_end_df_index = dch_start_index - 1

                if status_end_df_index <= start_df_index:
                    break

                end_df_index = status_end_df_index
                pause_boundary_index = dch_start_index
                trimmed_by_current = False
                first_nonzero_current_idx = None

                if ENABLE_PAU_CURRENT_CHECK:
                    candidate_current = (
                        df["_Current_Numeric"]
                        .iloc[start_df_index:dch_start_index]
                        .to_numpy(dtype=float)
                    )

                    zero_current_mask = (
                        np.isfinite(candidate_current)
                        & (
                            np.abs(candidate_current)
                            <= float(PAU_ZERO_CURRENT_TOLERANCE_A)
                        )
                    )

                    bad_positions = np.flatnonzero(~zero_current_mask)

                    if bad_positions.size > 0:
                        first_bad_offset = int(bad_positions[0])
                        first_nonzero_current_idx = (
                            start_df_index + first_bad_offset
                        )

                        if current_action == "reject":
                            break

                        end_df_index = first_nonzero_current_idx - 1
                        pause_boundary_index = first_nonzero_current_idx
                        trimmed_by_current = True

                if end_df_index <= start_df_index:
                    break

                pau_duration = (
                    df["Time_Seconds"].iloc[pause_boundary_index]
                    - df["Time_Seconds"].iloc[start_df_index]
                )

                if not (
                    np.isfinite(pau_duration)
                    and pau_duration > MIN_PAU_DURATION_S
                ):
                    break

                max_abs_current = np.nan
                if ENABLE_PAU_CURRENT_CHECK:
                    final_current = (
                        df["_Current_Numeric"]
                        .iloc[start_df_index:end_df_index + 1]
                        .to_numpy(dtype=float)
                    )

                    if final_current.size == 0:
                        break

                    if not np.all(np.isfinite(final_current)):
                        break

                    max_abs_current = float(np.max(np.abs(final_current)))

                    if max_abs_current > float(PAU_ZERO_CURRENT_TOLERANCE_A):
                        break

                valid_segments.append(
                    {
                        "start_idx": start_df_index,
                        "end_idx": end_df_index,
                        "duration_s": float(pau_duration),
                        "status_end_idx": status_end_df_index,
                        "trimmed_by_current": bool(trimmed_by_current),
                        "first_nonzero_current_idx": first_nonzero_current_idx,
                        "max_abs_current_A": max_abs_current,
                    }
                )

                break

            if next_state == "CHA":
                break

    return valid_segments


# ============================================================
# 4. 核心逻辑：提取单个循环文件里所有 PAU 片段的逐点数据
# ============================================================

def build_segment_dataframe(df, battery_id, cycle_file, segment_idx, segment_meta):
    """
    把一个有效 PAU 片段整理成长表 DataFrame，并额外附带 PAU 开始前 PRE_PAU_POINTS 个点。

    约定：
        - point_index < 0 : PAU 开始前的点
        - point_index == 0: PAU 第一个点
        - point_index > 0 : PAU 内后续点
        - time_s_from_seg_start 以 PAU 真正开始时刻为 0，所以前置点时间为负值

    仅导出以下列：
        segment_id
        point_index
        data_region
        time_s_from_seg_start
        voltage_V
        current_A
        temperature_C
    """
    start_idx = int(segment_meta["start_idx"])
    end_idx = int(segment_meta["end_idx"])

    pre_points = max(0, int(PRE_PAU_POINTS))
    export_start_idx = max(0, start_idx - pre_points)

    seg_df = df.iloc[export_start_idx:end_idx + 1].copy().reset_index(drop=True)

    source_row_indices = np.arange(export_start_idx, end_idx + 1, dtype=int)
    relative_point_indices = source_row_indices - start_idx

    time_from_file_start = seg_df["Time_Seconds"].to_numpy(dtype=float)
    pau_start_time = float(df["Time_Seconds"].iloc[start_idx])
    time_from_seg_start = time_from_file_start - pau_start_time

    data_region = np.where(relative_point_indices < 0, "pre_pau", "pau")

    out = pd.DataFrame(
        {
            "segment_id": segment_idx,
            "point_index": relative_point_indices,
            "data_region": data_region,
            "time_s_from_seg_start": time_from_seg_start,
            "voltage_V": seg_df["_Voltage_Numeric"].to_numpy(dtype=float),
            "current_A": seg_df["_Current_Numeric"].to_numpy(dtype=float),
            "temperature_C": seg_df["_Temperature_Numeric"].to_numpy(dtype=float),
        }
    )

    return out


def export_segments_for_cycle_file(
    cycle_file,
    battery_id,
    base_output_dir,
    aging_seq=None,
):
    """
    处理单个 Aging 循环文件：识别 PAU 片段并导出逐点电压数据。

    aging_seq:
        该电池按源文件时间升序排列后的 Aging 顺序，从 1 开始。
        用于输出文件名中的 Aging001 / Aging002 / ...
    """
    df = load_and_prepare_cycle_dataframe(cycle_file)
    valid_segments = find_valid_pau_segments(df)

    if not valid_segments:
        print(f"⚠️ 无有效 PAU 片段，跳过: {cycle_file}")
        return 0

    battery_output_dir = os.path.join(base_output_dir, battery_id)
    os.makedirs(battery_output_dir, exist_ok=True)

    file_hash = hashlib.md5(str(cycle_file).encode("utf-8")).hexdigest()[:8]

    aging_time_text = format_aging_datetime_for_filename(cycle_file)
    soc_dod_text = extract_soc_dod_for_filename(cycle_file)

    if aging_seq is None:
        aging_seq_text = "Aging"
    else:
        aging_seq_text = f"Aging{int(aging_seq):03d}"

    # 示例：
    # A123_007_Aging001_2025-03-31_172218_1e492850
    # VTC6_006_Aging002_2024-09-09_110001_ab12cd34
    file_stub = safe_filename(
        f"{battery_id}_{aging_seq_text}_{aging_time_text}_{soc_dod_text}_{file_hash}"
    )

    all_segment_frames = []

    for i, segment_meta in enumerate(valid_segments, start=1):
        seg_frame = build_segment_dataframe(
            df, battery_id, cycle_file, i, segment_meta
        )

        if EXPORT_ONE_CSV_PER_SEGMENT:
            seg_csv_path = os.path.join(
                battery_output_dir,
                f"{file_stub}_seg{i}_pau_voltage.csv",
            )
            seg_frame.to_csv(
                seg_csv_path,
                index=False,
                sep=CSV_SEPARATOR,
                decimal=CSV_DECIMAL,
                encoding="utf-8-sig",
            )
            print(f"✅ 已导出: {seg_csv_path}")
        else:
            all_segment_frames.append(seg_frame)

    if not EXPORT_ONE_CSV_PER_SEGMENT and all_segment_frames:
        combined = pd.concat(all_segment_frames, ignore_index=True)
        combined_csv_path = os.path.join(
            battery_output_dir,
            f"{file_stub}_pau_voltage_segments.csv",
        )
        combined.to_csv(
            combined_csv_path,
            index=False,
            sep=CSV_SEPARATOR,
            decimal=CSV_DECIMAL,
            encoding="utf-8-sig",
        )
        print(f"✅ 已导出: {combined_csv_path}  (共 {len(valid_segments)} 个片段)")

    return len(valid_segments)


# ============================================================
# 5. 批量主程序
# ============================================================

def main():
    allowed_battery_numbers = get_allowed_battery_numbers()

    print(
        f"🔋 自动识别电池型号: "
        f"{', '.join(SUPPORTED_BATTERY_PREFIXES)}"
    )
    print(
        f"↩️ 每个 PAU 额外导出开始前 "
        f"{PRE_PAU_POINTS} 个采样点"
    )

    if allowed_battery_numbers is None:
        print("🔎 编号筛选: 不限制编号")
    elif BATTERY_NUMBERS is not None:
        print(
            "🔎 编号筛选: "
            + ", ".join(
                f"{number:03d}"
                for number in sorted(allowed_battery_numbers)
            )
        )
    else:
        print(
            f"🔎 编号筛选: "
            f"{min(allowed_battery_numbers):03d} -> "
            f"{max(allowed_battery_numbers):03d}"
        )

    print(
        f"📡 数据来源: {CYCLE_FILES_DIR}"
        + (
            "（在线 S3 路径）"
            if is_s3_path(CYCLE_FILES_DIR)
            else "（先按本地路径尝试，找不到再自动改用 S3 在线读取）"
        )
    )

    cycle_files_all = list_files_from_folder(
        CYCLE_FILES_DIR,
        extensions=CYCLE_FILE_EXTENSIONS,
        keyword=AGING_KEYWORD,
    )

    if not cycle_files_all:
        print(
            f"❌ 未找到文件名包含 {AGING_KEYWORD} "
            f"的循环 parquet 文件。"
        )
        return

    cycle_files = filter_files_by_allowed_batteries(
        cycle_files_all,
        allowed_battery_numbers,
        "循环文件",
    )

    if not cycle_files:
        print("❌ 电池编号筛选后没有可处理的循环文件。")
        return

    print(
        f"✅ 找到 Aging 循环文件数量: "
        f"{len(cycle_files_all)}"
    )
    print(
        f"✅ 筛选后循环文件数量: "
        f"{len(cycle_files)}"
    )

    cycle_files_by_battery = group_files_by_battery(
        cycle_files
    )

    # 关键修改：
    # 只处理实际找到 Aging 文件的电池，
    # 不再人为构造 A123_001~300、VTC6_001~300。
    battery_ids_to_process = sorted(
        cycle_files_by_battery.keys()
    )

    print(
        f"🔋 实际找到可处理电池数量: "
        f"{len(battery_ids_to_process)}"
    )

    if battery_ids_to_process:
        print(
            "🔋 实际电池: "
            + ", ".join(battery_ids_to_process)
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 输出文件夹: {OUTPUT_DIR}")
    print(
        "📝 导出模式: "
        + (
            "每个片段单独一个 CSV"
            if EXPORT_ONE_CSV_PER_SEGMENT
            else "每个 Aging 文件合并一个 CSV"
        )
    )

    total_segments_exported = 0
    total_files_with_segments = 0
    total_files_skipped = 0

    for battery_id in tqdm(
        battery_ids_to_process,
        desc="电池进度",
        unit="块",
    ):
        battery_cycle_files = cycle_files_by_battery.get(
            battery_id,
            [],
        )

        if not battery_cycle_files:
            continue

        print(
            f"\n🔋 开始处理 {battery_id}: "
            f"{len(battery_cycle_files)} 个 Aging 文件"
        )

        for aging_seq, cycle_file in enumerate(
            tqdm(
                battery_cycle_files,
                desc=f"{battery_id} 文件进度",
                unit="文件",
                leave=False,
            ),
            start=1,
        ):
            aging_time_text = format_aging_datetime_for_filename(cycle_file)

            print(
                f"   🧭 {battery_id} Aging{aging_seq:03d}: "
                f"{aging_time_text}"
            )

            try:
                segment_count = export_segments_for_cycle_file(
                    cycle_file,
                    battery_id,
                    OUTPUT_DIR,
                    aging_seq=aging_seq,
                )
            except Exception as e:
                total_files_skipped += 1
                print("\n⚠️ 处理失败，跳过文件:")
                print(f"   {cycle_file}")
                print(f"   原因: {e}")
                continue

            if segment_count > 0:
                total_files_with_segments += 1
                total_segments_exported += segment_count
            else:
                total_files_skipped += 1

    print("\n" + "=" * 100)
    print("✅ 批量导出完成")
    print(
        f"📦 成功导出的 Aging 文件数: "
        f"{total_files_with_segments}"
    )
    print(
        f"📦 累计导出的 PAU 片段数: "
        f"{total_segments_exported}"
    )
    print(
        f"⚠️ 跳过的 Aging 文件数 "
        f"(无有效片段/处理失败): "
        f"{total_files_skipped}"
    )
    print(f"📁 输出文件夹: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
