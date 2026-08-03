import os
import re
import hashlib
import warnings

import numpy as np
import pandas as pd
import s3fs
from scipy.optimize import curve_fit
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 0. 用户配置区
# ============================================================
# 例如只处理 VTC6_001 到 VTC6_040
BATTERY_NUMBER_RANGE = (1, 300)

# 如果只想处理指定电池，例如 [6, 12, 20]，则设置这里
# 若不为 None，会优先使用 BATTERY_NUMBERS
BATTERY_NUMBERS = None

CYCLE_FILES_DIR = "projects/j8005-metabatt/Metabatt/VTC"

SOH_RECORDS_DIR = "projects/j8005-metabatt/Metabatt/VTC/40_capacity_monitore"

CSV_SEPARATOR = ";"
CSV_DECIMAL = ","

AGING_KEYWORD = "Aging"
SOH_FILE_KEYWORD = "capacity"

CYCLE_FILE_EXTENSIONS = (".parquet",)
SOH_FILE_EXTENSIONS = (".csv",)
UNQUALIFIED_SEGMENTS = []  # 存储不符合 tau2 > 5*tau1 的片段信息

# ========== 排除特定工况 ==========
# 设置要排除的工况组合，格式为 (SOC, DOD)
# 例如排除 DOD20_SOC30 和 DOD20_SOC50
EXCLUDED_CONDITIONS = [
    (30, 20),  # SOC30, DOD20
    (50, 20),  # SOC50, DOD20
]
# ==================================

COL_TIME = "Zeit"
COL_CURRENT = "Strom"
COL_VOLTAGE = "Spannung"
COL_STATE = "Zustand"

MIN_PAU_DURATION_S = 800

PROCESS_ALL_SEGMENTS = True
MAX_SEGMENTS_TO_PROCESS = None

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
    folder = str(folder)

    if os.path.isfile(folder):
        files = [folder]

    elif os.path.isdir(folder):
        files = []
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                files.append(os.path.join(root, filename))

    else:
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
    if is_s3_path(target_file):
        return pd.read_parquet(
            target_file,
            storage_options=get_s3_storage_options(),
        )

    return pd.read_parquet(target_file)


def read_soh_csv(csv_path: str, header="infer"):
    read_kwargs = {
        "sep": None,
        "engine": "python",
        "header": header,
    }

    if is_s3_path(csv_path):
        read_kwargs["storage_options"] = get_s3_storage_options()

    try:
        return pd.read_csv(csv_path, **read_kwargs)
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "latin1"
        return pd.read_csv(csv_path, **read_kwargs)


# ============================================================
# 2. 通用工具
# ============================================================

def safe_filename(text: str, max_len=160):
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(text))
    text = re.sub(r"\s+", "_", text)
    return text[:max_len].strip("._ ")


def extract_battery_id(filepath: str):
    texts = [
        os.path.basename(str(filepath)),
        str(filepath),
    ]

    for text in texts:
        match = re.search(r"VTC6[_=\-\s]?(\d{1,4})", text, re.IGNORECASE)
        if match:
            return f"VTC6_{int(match.group(1)):03d}"

    return None


def get_allowed_battery_ids():
    if BATTERY_NUMBERS is not None:
        return {f"VTC6_{int(number):03d}" for number in BATTERY_NUMBERS}

    if BATTERY_NUMBER_RANGE is not None:
        start_number, end_number = BATTERY_NUMBER_RANGE
        return {
            f"VTC6_{number:03d}"
            for number in range(int(start_number), int(end_number) + 1)
        }

    return None


def should_exclude_condition(filepath: str):
    """
    检查文件是否属于要排除的工况
    
    参数:
        filepath: 文件路径
    
    返回:
        bool: True表示应该排除，False表示保留
    """
    if not EXCLUDED_CONDITIONS:
        return False
    
    filename = os.path.basename(str(filepath))
    
    # 提取SOC和DOD
    soc_match = re.search(r'(\d+)SOC', filename, re.IGNORECASE)
    dod_match = re.search(r'(\d+)DOD', filename, re.IGNORECASE)
    
    if not soc_match or not dod_match:
        # 如果无法提取SOC或DOD，默认保留
        return False
    
    soc = int(soc_match.group(1))
    dod = int(dod_match.group(1))
    
    # 检查是否在排除列表中
    for excluded_soc, excluded_dod in EXCLUDED_CONDITIONS:
        if soc == excluded_soc and dod == excluded_dod:
            return True
    
    return False


def filter_files_by_allowed_batteries(files, allowed_battery_ids, file_label):
    filtered_files = []
    unknown_count = 0
    out_of_range_count = 0
    excluded_condition_count = 0

    for file in files:
        battery_id = extract_battery_id(file)

        if battery_id is None:
            unknown_count += 1
            continue

        if allowed_battery_ids is not None and battery_id not in allowed_battery_ids:
            out_of_range_count += 1
            continue

        # 检查是否要排除特定工况
        if should_exclude_condition(file):
            excluded_condition_count += 1
            continue

        filtered_files.append(file)

    if unknown_count:
        print(f"⚠️ {file_label} 中有 {unknown_count} 个文件无法识别电池编号，已跳过。")

    if out_of_range_count:
        print(f"ℹ️ {file_label} 中有 {out_of_range_count} 个文件不在编号范围内，已跳过。")
    
    if excluded_condition_count:
        excluded_desc = ", ".join([f"SOC{soc}_DOD{dod}" for soc, dod in EXCLUDED_CONDITIONS])
        print(f"ℹ️ {file_label} 中排除了 {excluded_condition_count} 个 {excluded_desc} 工况的文件")

    return sorted(filtered_files)


def group_files_by_battery(files):
    grouped = {}

    for file in files:
        battery_id = extract_battery_id(file)

        if battery_id is None:
            continue

        grouped.setdefault(battery_id, []).append(file)

    for battery_id in grouped:
        grouped[battery_id] = sorted(grouped[battery_id])

    return grouped


def extract_soc_dod_from_filename(filepath: str):
    filename = os.path.basename(str(filepath))

    soc_match = re.search(r"(\d+)SOC", filename, re.IGNORECASE)
    dod_match = re.search(r"(\d+)DOD", filename, re.IGNORECASE)

    soc = soc_match.group(1) if soc_match else None
    dod = dod_match.group(1) if dod_match else None

    return soc, dod


def extract_test_conditions(filepath: str, return_dict=False):
    """
    从文件名中提取测试工况信息，如温度、SOC、DOD、倍率等
    示例: J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2024-10-24_074211=jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS015976 _ Format01=Kreis M3-034=filesize-34151246=finished.parquet
    提取结果: 25grad_70SOC_60DOD_05C_Cyc
    
    参数:
        filepath: 文件路径
        return_dict: 如果为True，返回字典格式，否则返回字符串
    """
    filename = os.path.basename(str(filepath))
    conditions = {}
    condition_parts = []
    
    # 提取温度 (例如: 25grad, 25°C, 25C)
    temp_match = re.search(r'(\d{2})grad', filename, re.IGNORECASE)
    if temp_match:
        temp_str = f"{temp_match.group(1)}grad"
        conditions['temperature'] = temp_str
        condition_parts.append(temp_str)
    else:
        temp_match = re.search(r'(\d{2})[°]?C', filename, re.IGNORECASE)
        if temp_match:
            temp_str = f"{temp_match.group(1)}C"
            conditions['temperature'] = temp_str
            condition_parts.append(temp_str)
    
    # 提取SOC (例如: 70SOC)
    soc_match = re.search(r'(\d+)SOC', filename, re.IGNORECASE)
    if soc_match:
        soc_str = f"{soc_match.group(1)}SOC"
        conditions['soc'] = soc_str
        condition_parts.append(soc_str)
    
    # 提取DOD (例如: 60DOD)
    dod_match = re.search(r'(\d+)DOD', filename, re.IGNORECASE)
    if dod_match:
        dod_str = f"{dod_match.group(1)}DOD"
        conditions['dod'] = dod_str
        condition_parts.append(dod_str)
    
    # # 提取倍率 (例如: 05C, 1C, 0.5C)
    # rate_match = re.search(r'(\d+[\.]?\d*)C', filename, re.IGNORECASE)
    # if rate_match:
    #     rate_value = rate_match.group(1)
    #     # 检查是否包含小数点
    #     if '.' in rate_value:
    #         # 将小数点替换为下划线，避免文件名中的特殊字符
    #         rate_str = f"{rate_value.replace('.', '')}C"
    #     else:
    #         rate_str = f"{rate_value}C"
    #     conditions['rate'] = rate_str
    #     condition_parts.append(rate_str)
    
    # # 提取循环类型 (例如: Cyc, Pulse, Dyn)
    # cycle_type_match = re.search(r'(Cyc|Pulse|Dyn)', filename, re.IGNORECASE)
    # if cycle_type_match:
    #     cycle_str = cycle_type_match.group(1)
    #     conditions['cycle_type'] = cycle_str
    #     condition_parts.append(cycle_str)
    
    if return_dict:
        return conditions if conditions else None
    
    # 如果没有提取到任何工况信息，返回None
    if not condition_parts:
        return None
    
    return "_".join(condition_parts)


def get_common_test_conditions(files):
    """
    从一组文件中提取共同的工况信息
    如果所有文件都有相同的工况，返回该工况字符串
    否则返回None
    """
    if not files:
        return None
    
    # 提取所有文件的工况信息
    all_conditions = []
    for file in files:
        cond = extract_test_conditions(file, return_dict=True)
        if cond:
            all_conditions.append(cond)
    
    if not all_conditions:
        return None
    
    # 检查是否所有文件的工况都相同
    first_cond = all_conditions[0]
    all_same = all(cond == first_cond for cond in all_conditions)
    
    if all_same:
        # 提取温度、SOC、DOD等关键信息组成文件夹名
        parts = []
        if 'temperature' in first_cond:
            parts.append(first_cond['temperature'])
        if 'soc' in first_cond:
            parts.append(first_cond['soc'])
        if 'dod' in first_cond:
            parts.append(first_cond['dod'])
        if 'rate' in first_cond:
            parts.append(first_cond['rate'])
        if 'cycle_type' in first_cond:
            parts.append(first_cond['cycle_type'])
        
        if parts:
            return "_".join(parts)
    
    # 如果工况不一致，返回通用名称
    return "mixed_conditions"


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df, candidates):
    normalized_columns = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_columns:
            return normalized_columns[key]

    for col in df.columns:
        col_key = normalize_column_name(col)

        for candidate in candidates:
            candidate_key = normalize_column_name(candidate)
            if candidate_key in col_key:
                return col

    return None


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
# 3. SOH 文件读取
# ============================================================

def guess_datetime_column(df):
    date_pattern = (
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
        r"|\d{8}"
    )

    best_col = None
    best_score = 0

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        text_values = df[col].astype(str)
        pattern_score = int(
            text_values.str.contains(date_pattern, regex=True, na=False).sum()
        )

        if pattern_score == 0:
            continue

        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        score = int(parsed.notna().sum())

        if score > best_score:
            best_score = score
            best_col = col

    min_score = max(1, int(len(df) * 0.5))

    if best_score >= min_score:
        return best_col

    return None


def guess_soh_column(df, time_col):
    columns = list(df.columns)
    time_pos = columns.index(time_col) if time_col in columns else len(columns)

    candidates = []

    for col in columns:
        if col == time_col:
            continue

        values = parse_numeric_series(df[col]).dropna()

        if values.empty:
            continue

        median_value = float(values.median())

        if median_value <= 0 or median_value > 120:
            continue

        col_pos = columns.index(col)

        if 40 <= median_value <= 120:
            soh_priority = 0
        elif 0.4 <= median_value <= 1.2:
            soh_priority = 1
        else:
            soh_priority = 2

        before_time_priority = 0 if col_pos < time_pos else 1
        distance_to_time = abs(col_pos - time_pos)

        candidates.append(
            (
                soh_priority,
                before_time_priority,
                distance_to_time,
                col,
            )
        )

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][3]


def normalize_soh_dataframe(df_soh_raw):
    time_col = find_column(
        df_soh_raw,
        [
            "CAP_start_time",
            "cap_start_time",
            "start_time",
            "Start_Time",
            "timestamp",
            "time",
            "date",
        ],
    )

    if time_col is None:
        time_col = guess_datetime_column(df_soh_raw)

    soh_col = find_column(
        df_soh_raw,
        [
            "SOH",
            "SOH (%)",
            "SOH_percent",
            "soh_percent",
        ],
    )

    if soh_col is None and time_col is not None:
        soh_col = guess_soh_column(df_soh_raw, time_col)

    if time_col is None or soh_col is None:
        return None

    checkup_time = pd.to_datetime(
        df_soh_raw[time_col],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    soh_values = parse_numeric_series(df_soh_raw[soh_col])

    df_soh = pd.DataFrame(
        {
            "checkup_time": checkup_time,
            "SOH": soh_values,
        }
    )

    df_soh = (
        df_soh
        .dropna(subset=["checkup_time", "SOH"])
        .sort_values("checkup_time")
        .reset_index(drop=True)
    )

    if df_soh.empty:
        return None

    if df_soh["SOH"].median() <= 1.5:
        df_soh["SOH"] = df_soh["SOH"] * 100

    return df_soh


def load_soh_records(soh_folder: str, allowed_battery_ids=None):
    soh_files_all = list_files_from_folder(
        soh_folder,
        extensions=SOH_FILE_EXTENSIONS,
        keyword=SOH_FILE_KEYWORD,
    )

    if not soh_files_all:
        print(f"⚠️ 未在 SOH 文件夹中找到 CSV 文件: {soh_folder}")
        return {}

    soh_files = filter_files_by_allowed_batteries(
        soh_files_all,
        allowed_battery_ids,
        "SOH文件",
    )

    soh_records_raw = {}

    for soh_file in soh_files:
        battery_id = extract_battery_id(soh_file)

        if battery_id is None:
            continue

        parsed_candidates = []

        for header in ["infer", None]:
            try:
                df_soh_raw = read_soh_csv(soh_file, header=header)
                df_soh = normalize_soh_dataframe(df_soh_raw)

                if df_soh is not None and not df_soh.empty:
                    parsed_candidates.append(df_soh)

            except Exception:
                pass

        if not parsed_candidates:
            print(f"⚠️ 跳过无法解析 SOH 文件: {soh_file}")
            continue

        df_soh = max(parsed_candidates, key=len)
        soh_records_raw.setdefault(battery_id, []).append(df_soh)

    soh_records = {}

    for battery_id, dfs in soh_records_raw.items():
        combined = (
            pd.concat(dfs, ignore_index=True)
            .drop_duplicates(subset=["checkup_time"], keep="last")
            .sort_values("checkup_time")
            .reset_index(drop=True)
        )

        soh_records[battery_id] = combined

    print(f"✅ 已读取 SOH 记录电池数量: {len(soh_records)}")

    for battery_id, df_soh in soh_records.items():
        print(
            f"   {battery_id}: {len(df_soh)} 条 checkup, "
            f"{df_soh['checkup_time'].min()} -> {df_soh['checkup_time'].max()}"
        )

    return soh_records


# ============================================================
# 4. 循环文件日期提取与 checkup 区间分配
# ============================================================

def extract_cycle_time_candidates(filepath: str, candidate_years=None):
    filename = os.path.basename(str(filepath))
    candidates = []
    seen = set()

    def add_candidate(timestamp, precision, source):
        if pd.isna(timestamp):
            return

        timestamp = pd.Timestamp(timestamp)
        key = (timestamp, precision, source)

        if key in seen:
            return

        seen.add(key)
        candidates.append(
            {
                "time": timestamp,
                "precision": precision,
                "source": source,
            }
        )

    for match in re.finditer(r"(\d{4}-\d{2}-\d{2})[_\s-](\d{6})", filename):
        try:
            add_candidate(
                pd.to_datetime(
                    f"{match.group(1)} {match.group(2)}",
                    format="%Y-%m-%d %H%M%S",
                ),
                "datetime",
                match.group(0),
            )
        except Exception:
            pass

    for match in re.finditer(r"(?<!\d)(\d{8})[_\s-](\d{6})(?!\d)", filename):
        try:
            add_candidate(
                pd.to_datetime(
                    f"{match.group(1)} {match.group(2)}",
                    format="%Y%m%d %H%M%S",
                ),
                "datetime",
                match.group(0),
            )
        except Exception:
            pass

    for match in re.finditer(r"(?<!\d)(\d{8})(\d{6})(?!\d)", filename):
        try:
            add_candidate(
                pd.to_datetime(
                    f"{match.group(1)} {match.group(2)}",
                    format="%Y%m%d %H%M%S",
                ),
                "datetime",
                match.group(0),
            )
        except Exception:
            pass

    for match in re.finditer(r"(\d{4}-\d{2}-\d{2})", filename):
        try:
            add_candidate(
                pd.to_datetime(match.group(1), format="%Y-%m-%d"),
                "date",
                match.group(1),
            )
        except Exception:
            pass

    for match in re.finditer(r"(?<!\d)(\d{8})(?!\d)", filename):
        try:
            add_candidate(
                pd.to_datetime(match.group(1), format="%Y%m%d"),
                "date",
                match.group(1),
            )
        except Exception:
            pass

    if candidate_years is None:
        return candidates

    for match in re.finditer(r"(?<!\d)(\d{4})(?!\d)", filename):
        token = match.group(1)
        month = int(token[:2])
        day = int(token[2:])

        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue

        for year in candidate_years:
            try:
                add_candidate(
                    pd.Timestamp(year=int(year), month=month, day=day),
                    "date",
                    token,
                )
            except Exception:
                pass

    for match in re.finditer(r"(?<!\d)(\d{2})[-_](\d{2})(?!\d)", filename):
        month = int(match.group(1))
        day = int(match.group(2))

        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue

        for year in candidate_years:
            try:
                add_candidate(
                    pd.Timestamp(year=int(year), month=month, day=day),
                    "date",
                    match.group(0),
                )
            except Exception:
                pass

    return candidates


def candidate_in_checkup_interval(candidate, start_time, end_time):
    cycle_time = candidate["time"]

    if candidate["precision"] == "datetime":
        return start_time <= cycle_time < end_time

    return start_time.date() <= cycle_time.date() <= end_time.date()


def build_soh_intervals(df_soh):
    intervals = []

    df_soh = (
        df_soh
        .sort_values("checkup_time")
        .reset_index(drop=True)
    )

    for idx in range(len(df_soh) - 1):
        before = df_soh.iloc[idx]
        after = df_soh.iloc[idx + 1]

        intervals.append(
            {
                "interval_index": idx + 1,
                "checkup_start_time": before["checkup_time"],
                "checkup_end_time": after["checkup_time"],
                "soh_start": float(before["SOH"]),
                "soh_end": float(after["SOH"]),
            }
        )

    return intervals


def match_cycle_file_to_soh_interval(cycle_file, battery_id, df_soh):
    if len(df_soh) < 2:
        raise ValueError(f"{battery_id} 的 SOH 记录少于 2 条，无法形成区间。")

    candidate_years = sorted(df_soh["checkup_time"].dt.year.unique())

    cycle_candidates = extract_cycle_time_candidates(
        cycle_file,
        candidate_years=candidate_years,
    )

    if not cycle_candidates:
        raise ValueError(f"无法从文件名提取日期: {os.path.basename(cycle_file)}")

    intervals = build_soh_intervals(df_soh)

    for candidate in cycle_candidates:
        for interval in intervals:
            if candidate_in_checkup_interval(
                candidate,
                interval["checkup_start_time"],
                interval["checkup_end_time"],
            ):
                matched = interval.copy()
                matched.update(
                    {
                        "battery_id": battery_id,
                        "cycle_time": candidate["time"],
                        "cycle_time_source": candidate["source"],
                    }
                )
                return matched

    candidate_text = ", ".join(
        f"{item['source']} -> {item['time']}"
        for item in cycle_candidates
    )

    raise ValueError(
        f"循环文件日期没有落入任何 checkup 区间: {os.path.basename(cycle_file)}; "
        f"候选日期: {candidate_text}"
    )


def group_cycle_files_by_soh_interval(battery_id, cycle_files, df_soh):
    interval_groups = {}
    skipped = 0

    for cycle_file in cycle_files:
        try:
            interval_info = match_cycle_file_to_soh_interval(
                cycle_file,
                battery_id,
                df_soh,
            )

            interval_key = interval_info["interval_index"]

            interval_groups.setdefault(
                interval_key,
                {
                    "interval_info": interval_info,
                    "cycle_entries": [],
                },
            )

            interval_groups[interval_key]["cycle_entries"].append(
                {
                    "cycle_file": cycle_file,
                    "cycle_time": interval_info["cycle_time"],
                    "cycle_time_source": interval_info["cycle_time_source"],
                }
            )

        except Exception as e:
            skipped += 1
            print(f"\n⚠️ {battery_id}: 跳过无法匹配 SOH 区间的循环文件")
            print(f"   文件: {cycle_file}")
            print(f"   原因: {e}")

    for group in interval_groups.values():
        group["cycle_entries"] = sorted(
            group["cycle_entries"],
            key=lambda item: (
                item["cycle_time"],
                os.path.basename(str(item["cycle_file"])),
            ),
        )

    return interval_groups, skipped


# ============================================================
# 5. 二阶 RC 模型
# ============================================================

def relaxation_model_pure(t, OCV, A1, tau1, A2, tau2):
    t_safe = np.clip(t, 0, None)
    return OCV + A1 * np.exp(-t_safe / tau1) + A2 * np.exp(-t_safe / tau2)


def identify_parameters_adaptive(t_data, V_data, segment_info=None):
    t_data = np.asarray(t_data, dtype=float)
    V_data = np.asarray(V_data, dtype=float)

    valid_mask = np.isfinite(t_data) & np.isfinite(V_data)
    t_data = t_data[valid_mask]
    V_data = V_data[valid_mask]

    if len(t_data) < 10:
        return None, None, False

    t_data = t_data - np.min(t_data)

    if np.max(t_data) <= 0:
        return None, None, False

    V_start = V_data[0]
    V_end = V_data[-1]

    OCV_init = V_end
    delta_V = V_start - V_end
    abs_delta = abs(delta_V)

    A1_init = delta_V * 0.3
    A2_init = delta_V * 0.7
    tau1_init = 8.0
    tau2_init = 150.0

    initial_guess = [OCV_init, A1_init, tau1_init, A2_init, tau2_init]

    ocv_min = min(V_start, V_end) - 0.05
    ocv_max = max(V_start, V_end) + 0.05

    if delta_V >= 0:
        a_min = 0.0
        a_max = max(0.2, abs_delta * 1.5)
    else:
        a_min = -max(0.2, abs_delta * 1.5)
        a_max = 0.0

    lower_bounds = [ocv_min, a_min, 0.2, a_min, 30.0]
    upper_bounds = [ocv_max, a_max, 80.0, a_max, 5000.0]

    # 尝试多个初始值以满足 tau2 > 5*tau1
    tau_combinations = [
        (8.0, 150.0),
        (5.0, 100.0),
        (10.0, 200.0),
        (3.0, 80.0),
        (15.0, 300.0),
        (20.0, 500.0),
        (2.0, 60.0),
        (12.0, 250.0),
    ]
    
    best_result = None
    best_ratio = 0
    all_attempts = []
    
    for tau1_cand, tau2_cand in tau_combinations:
        # 修改为5倍
        if tau2_cand <= 5 * tau1_cand:
            continue
            
        guess = [OCV_init, A1_init, tau1_cand, A2_init, tau2_cand]
        
        try:
            popt, pcov = curve_fit(
                relaxation_model_pure,
                t_data,
                V_data,
                p0=guess,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=150000,
                method="trf",
            )
            
            tau1_fit = popt[2]
            tau2_fit = popt[4]
            ratio = tau2_fit / tau1_fit
            
            all_attempts.append((tau1_fit, tau2_fit, ratio))
            
            # 检查条件 - 修改为5倍
            if tau2_fit > 5 * tau1_fit:
                return popt, pcov, True
            else:
                # 记录最佳结果（最接近满足条件的）
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = (popt, pcov, tau1_fit, tau2_fit)
                    
        except Exception:
            continue
    
    # 所有尝试都不满足条件
    if best_result is not None:
        popt, pcov, tau1_fit, tau2_fit = best_result
        ratio = tau2_fit / tau1_fit
        
        # 记录到全局列表 - 修改提示信息为5倍
        if segment_info is not None:
            unqualified_record = {
                'segment_info': segment_info,
                'tau1': tau1_fit,
                'tau2': tau2_fit,
                'ratio': ratio,
                'required_ratio': 5.0,
                'ocv': popt[0],
                'a1': popt[1],
                'a2': popt[3]
            }
            UNQUALIFIED_SEGMENTS.append(unqualified_record)
            
            # 打印警告 - 修改提示信息为5倍
            print(f"⚠️ 片段不满足 tau2 > 5*tau1: {segment_info}")
            print(f"   tau1={tau1_fit:.4f}, tau2={tau2_fit:.4f}, ratio={ratio:.4f} (需要 > 5.0)")
        
        return popt, pcov, False
    
    return None, None, False


# ============================================================
# 6. 循环文件预处理与 PAU 片段识别
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

    required_cols = [COL_TIME, COL_CURRENT, COL_VOLTAGE, COL_STATE]
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

    return df


def find_valid_pau_segments(df):
    df_temp = df[["Time_Seconds", "Zustand_Clean"]].copy()
    df_temp["prev_state"] = df_temp["Zustand_Clean"].shift()

    change_points = (
        df_temp[df_temp["Zustand_Clean"] != df_temp["prev_state"]]
        .copy()
        .reset_index()
    )

    valid_segments = []

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
                end_df_index = dch_start_index - 1

                if end_df_index <= start_df_index:
                    break

                pau_duration = (
                    df["Time_Seconds"].iloc[dch_start_index]
                    - df["Time_Seconds"].iloc[start_df_index]
                )

                if np.isfinite(pau_duration) and pau_duration > MIN_PAU_DURATION_S:
                    valid_segments.append(
                        {
                            "start_idx": start_df_index,
                            "end_idx": end_df_index,
                            "duration_s": float(pau_duration),
                        }
                    )

                break

            if next_state == "CHA":
                break

    return valid_segments


def count_valid_segments_for_cycle(cycle_file):
    df = load_and_prepare_cycle_dataframe(cycle_file)
    valid_segments = find_valid_pau_segments(df)
    return len(valid_segments)


# ============================================================
# 7. 单个 PAU 片段 RC 参数辨识
# ============================================================

def process_segment(df, start_idx, end_idx, segment_info=None):
    seg_df = df.iloc[start_idx:end_idx + 1].copy()

    if seg_df.empty:
        return None

    seg_df["Relative_Time_s"] = (
        seg_df["Time_Seconds"] - seg_df["Time_Seconds"].iloc[0]
    )

    t_data = seg_df["Relative_Time_s"].to_numpy(dtype=float)
    V_data = seg_df["_Voltage_Numeric"].to_numpy(dtype=float)

    valid_mask = np.isfinite(t_data) & np.isfinite(V_data)
    t_data = t_data[valid_mask]
    V_data = V_data[valid_mask]

    if len(t_data) < 10:
        return None

    popt, pcov, success = identify_parameters_adaptive(t_data, V_data, segment_info)

    if not success:
        return None

    OCV, A1, tau1, A2, tau2 = popt

    lookback_idx = max(0, start_idx - 1)

    current_series = df["_Current_Numeric"]
    voltage_series = df["_Voltage_Numeric"]

    I_prev = current_series.iloc[lookback_idx]

    if np.isfinite(I_prev) and abs(I_prev) < 0.05 and lookback_idx > 5:
        I_prev = current_series.iloc[lookback_idx - 5:lookback_idx + 1].mean()

    V_prior = voltage_series.iloc[lookback_idx]

    if np.isfinite(I_prev) and np.isfinite(V_prior) and abs(I_prev) > 0.01:
        R1 = abs(A1 / I_prev)
        R2 = abs(A2 / I_prev)

        V_fitted_zero = OCV + A1 + A2
        R0_raw = abs(V_prior - V_fitted_zero) / abs(I_prev)
        R0 = max(1e-6, R0_raw)

    else:
        R0 = 0.002
        R1 = 0.005
        R2 = 0.010

    return {
        "OCV": OCV,
        "R0": R0,
        "R1": R1,
        "tau1": tau1,
        "R2": R2,
        "tau2": tau2,
    }


# ============================================================
# 8. CSV 导出
# ============================================================

def build_output_csv_path(cycle_file, battery_id, cycle_time, soc_value, dod_value, base_output_dir):
    """
    构建输出CSV文件路径，包含工况信息
    示例输出: VTC6_003_20241024_074211_25grad_70SOC_60DOD_05C_Cyc_abcdefgh_rc_soh.csv
    
    参数:
        cycle_file: 循环文件路径
        battery_id: 电池ID
        cycle_time: 循环时间
        soc_value: SOC值
        dod_value: DOD值
        base_output_dir: 基础输出目录
    """
    file_hash = hashlib.md5(str(cycle_file).encode("utf-8")).hexdigest()[:8]
    
    # 提取工况信息
    test_conditions = extract_test_conditions(cycle_file)
    
    parts = [
        battery_id,
        cycle_time.strftime("%Y%m%d_%H%M%S"),
    ]
    
    # 如果有工况信息，添加到文件名中
    if test_conditions:
        parts.append(test_conditions)
    else:
        # 如果没有提取到完整的工况信息，使用SOC和DOD（如果有）
        if soc_value:
            parts.append(f"SOC{soc_value}")
        if dod_value:
            parts.append(f"DOD{dod_value}")
    
    parts.append(file_hash)
    parts.append("rc_soh")
    
    filename = safe_filename("_".join(parts)) + ".csv"
    
     # ========== 修改：创建包含工况信息的电池子文件夹 ==========
    # 构建电池子文件夹名，包含工况信息
    battery_folder_name = battery_id
    if test_conditions:
        # 从工况信息中提取关键部分（温度、SOC、DOD、倍率、循环类型）
        cond_parts = test_conditions.split('_')
        # 提取温度、SOC、DOD（这些是识别工况的关键）
        key_parts = []
        for part in cond_parts:
            if 'grad' in part or 'C' in part and not 'SOC' in part and not 'DOD' in part:
                # 温度
                key_parts.append(part)
            elif 'SOC' in part:
                key_parts.append(part)
            elif 'DOD' in part:
                key_parts.append(part)
        # 如果提取到了关键工况信息，添加到文件夹名
        if key_parts:
            battery_folder_name = f"{battery_id}_{'_'.join(key_parts)}"
        else:
            battery_folder_name = f"{battery_id}_{test_conditions}"
    
    battery_output_dir = os.path.join(base_output_dir, battery_folder_name)
    os.makedirs(battery_output_dir, exist_ok=True)
    # ==========================================================
    
    return os.path.join(battery_output_dir, filename)


def process_cycle_file_with_global_soh(cycle_entry, interval_info, soh_values_for_file, base_output_dir):
    cycle_file = cycle_entry["cycle_file"]
    battery_id = interval_info["battery_id"]
    cycle_time = cycle_entry["cycle_time"]

    soc_value, dod_value = extract_soc_dod_from_filename(cycle_file)

    df = load_and_prepare_cycle_dataframe(cycle_file)
    valid_segments = find_valid_pau_segments(df)

    if len(valid_segments) == 0:
        print(f"⚠️ 无有效 PAU 片段，跳过: {cycle_file}")
        return None

    if len(valid_segments) != len(soh_values_for_file):
        print(
            f"⚠️ 片段数量二次读取不一致: {os.path.basename(cycle_file)}; "
            f"第一次 {len(soh_values_for_file)}，第二次 {len(valid_segments)}"
        )

    usable_count = min(len(valid_segments), len(soh_values_for_file))

    num_to_extract = usable_count if PROCESS_ALL_SEGMENTS else min(10, usable_count)

    if MAX_SEGMENTS_TO_PROCESS is not None:
        num_to_extract = min(num_to_extract, MAX_SEGMENTS_TO_PROCESS)

    parameter_reports = []

    for segment_idx in tqdm(
        range(num_to_extract),
        desc="PAU片段拟合",
        unit="段",
        leave=False,
    ):
        segment = valid_segments[segment_idx]
        
        # 构建更详细的片段信息
        segment_info = f"{battery_id}_{os.path.basename(cycle_file)}_seg{segment_idx+1}"

        res = process_segment(
            df,
            segment["start_idx"],
            segment["end_idx"],
            segment_info,
        )

        if res is None:
            continue

        parameter_reports.append(
            {
                "segment_id": segment_idx + 1,
                "SOH (%)": round(float(soh_values_for_file[segment_idx]), 4),
                "OCV (V)": round(float(res["OCV"]), 6),
                "R0 (Ohm)": round(float(res["R0"]), 6),
                "R1 (Ohm)": round(float(res["R1"]), 6),
                "tau1 (s)": round(float(res["tau1"]), 2),
                "R2 (Ohm)": round(float(res["R2"]), 6),
                "tau2 (s)": round(float(res["tau2"]), 2),
            }
        )

    if not parameter_reports:
        print(f"⚠️ 当前循环文件所有片段拟合失败，未导出: {cycle_file}")
        return None

    summary_df = pd.DataFrame(parameter_reports)

    output_csv = build_output_csv_path(
        cycle_file,
        battery_id,
        cycle_time,
        soc_value,
        dod_value,
        base_output_dir,
    )

    summary_df.to_csv(
        output_csv,
        index=False,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        encoding="utf-8-sig",
    )

    print(f"✅ 已导出: {output_csv}")
    return output_csv


# ============================================================
# 9. checkup 区间级全局 SOH 插值处理
# ============================================================

def process_soh_interval_group(battery_id, interval_group, base_output_dir):
    interval_info = interval_group["interval_info"]
    cycle_entries = interval_group["cycle_entries"]

    print("\n" + "=" * 100)
    print(f"🔋 电池: {battery_id}")
    print(f"📆 Checkup 区间: {interval_info['checkup_start_time']} -> {interval_info['checkup_end_time']}")
    print(f"📉 SOH 区间: {interval_info['soh_start']:.4f}% -> {interval_info['soh_end']:.4f}%")
    print(f"📄 区间内 Aging 文件数: {len(cycle_entries)}")

    planned_entries = []
    skipped_count = 0

    print("🔎 第一次扫描：统计该 checkup 区间内全部有效 PAU 片段数量...")

    for cycle_entry in tqdm(cycle_entries, desc="统计片段", unit="文件", leave=False):
        cycle_file = cycle_entry["cycle_file"]

        try:
            segment_count = count_valid_segments_for_cycle(cycle_file)

            if segment_count <= 0:
                skipped_count += 1
                print(f"⚠️ 无有效 PAU 片段，跳过: {cycle_file}")
                continue

            planned_entry = cycle_entry.copy()
            planned_entry["segment_count"] = segment_count
            planned_entries.append(planned_entry)

        except Exception as e:
            skipped_count += 1
            print(f"\n⚠️ 统计片段失败，跳过文件:")
            print(f"   {cycle_file}")
            print(f"   原因: {e}")

    total_interval_segments = sum(
        entry["segment_count"]
        for entry in planned_entries
    )

    if total_interval_segments <= 0:
        print("⚠️ 当前 checkup 区间没有任何有效 PAU 片段，跳过该区间。")
        return 0, skipped_count

    print(f"✅ 当前 checkup 区间有效 Aging 文件数: {len(planned_entries)}")
    print(f"✅ 当前 checkup 区间有效 PAU 总片段数: {total_interval_segments}")

    global_soh_values = np.linspace(
        interval_info["soh_start"],
        interval_info["soh_end"],
        total_interval_segments,
    )

    success_count = 0
    cursor = 0

    print("🚀 第二次扫描：使用 checkup 区间全局 SOH 插值进行 RC 参数辨识...")

    for planned_entry in tqdm(planned_entries, desc="处理文件", unit="文件", leave=False):
        segment_count = planned_entry["segment_count"]
        soh_values_for_file = global_soh_values[cursor:cursor + segment_count]
        cursor += segment_count

        cycle_file = planned_entry["cycle_file"]

        print("\n" + "-" * 100)
        print(f"📄 循环文件: {os.path.basename(cycle_file)}")
        print(f"⏱️ 匹配日期: {planned_entry['cycle_time']}")
        print(f"🔎 日期来源: {planned_entry['cycle_time_source']}")
        print(f"🧩 该文件有效 PAU 片段数: {segment_count}")
        print(
            f"📉 该文件 SOH 分配: "
            f"{soh_values_for_file[0]:.4f}% -> {soh_values_for_file[-1]:.4f}%"
        )

        try:
            output_csv = process_cycle_file_with_global_soh(
                planned_entry,
                interval_info,
                soh_values_for_file,
                base_output_dir,
            )

            if output_csv is not None:
                success_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            skipped_count += 1
            print(f"\n⚠️ 处理失败，跳过文件:")
            print(f"   {cycle_file}")
            print(f"   原因: {e}")

    return success_count, skipped_count


# ============================================================
# 10. 批量主程序
# ============================================================

def main():
    allowed_battery_ids = get_allowed_battery_ids()

    if allowed_battery_ids is None:
        print("🔎 电池筛选: 处理所有可识别电池")
    else:
        print(
            f"🔎 电池筛选: {min(allowed_battery_ids)} -> {max(allowed_battery_ids)}, "
            f"共 {len(allowed_battery_ids)} 块"
        )
    
    # 显示排除的工况信息
    if EXCLUDED_CONDITIONS:
        excluded_desc = ", ".join([f"SOC{soc}_DOD{dod}" for soc, dod in EXCLUDED_CONDITIONS])
        print(f"🚫 排除工况: {excluded_desc}")

    cycle_files_all = list_files_from_folder(
        CYCLE_FILES_DIR,
        extensions=CYCLE_FILE_EXTENSIONS,
        keyword=AGING_KEYWORD,
    )

    if not cycle_files_all:
        print(f"❌ 未找到文件名包含 {AGING_KEYWORD} 的循环 parquet 文件。")
        return

    cycle_files = filter_files_by_allowed_batteries(
        cycle_files_all,
        allowed_battery_ids,
        "循环文件",
    )

    if not cycle_files:
        print("❌ 电池编号筛选后没有可处理的循环文件。")
        return

    print(f"✅ 找到 Aging 循环文件数量: {len(cycle_files_all)}")
    print(f"✅ 筛选后循环文件数量: {len(cycle_files)}")

    cycle_files_by_battery = group_files_by_battery(cycle_files)

    soh_records = load_soh_records(
        SOH_RECORDS_DIR,
        allowed_battery_ids=allowed_battery_ids,
    )

    if allowed_battery_ids is not None:
        battery_ids_to_process = sorted(allowed_battery_ids)
    else:
        battery_ids_to_process = sorted(cycle_files_by_battery.keys())

    # ========== 确定输出文件夹名称 ==========
    # 收集所有将要处理的循环文件
    all_files_to_process = []
    for battery_id in battery_ids_to_process:
        if battery_id in cycle_files_by_battery:
            all_files_to_process.extend(cycle_files_by_battery[battery_id])
    
    # 提取共同的工况信息
    common_conditions = get_common_test_conditions(all_files_to_process)
    
    # 添加排除工况信息到文件夹名
    exclusion_suffix = ""
    if EXCLUDED_CONDITIONS:
        excluded_parts = []
        for soc, dod in sorted(EXCLUDED_CONDITIONS):
            excluded_parts.append(f"noSOC{soc}DOD{dod}")
        exclusion_suffix = "_" + "_".join(excluded_parts)
    
    if common_conditions:
        # 构建包含工况信息的输出文件夹名
        if BATTERY_NUMBER_RANGE:
            output_dir_name = f"output_rc_soh_{BATTERY_NUMBER_RANGE[0]}_{BATTERY_NUMBER_RANGE[1]}_{common_conditions}{exclusion_suffix}"
        else:
            output_dir_name = f"output_rc_soh_{common_conditions}{exclusion_suffix}"
    else:
        # 如果没有工况信息，使用默认名称
        if BATTERY_NUMBER_RANGE:
            output_dir_name = f"output_rc_soh_{BATTERY_NUMBER_RANGE[0]}_{BATTERY_NUMBER_RANGE[1]}{exclusion_suffix}"
        else:
            output_dir_name = f"output_rc_soh{exclusion_suffix}"
    
    BASE_OUTPUT_DIR = output_dir_name
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"📁 输出文件夹: {BASE_OUTPUT_DIR}")
    print(f"📋 工况信息: {common_conditions if common_conditions else '混合工况（未提取到统一工况）'}")
    if EXCLUDED_CONDITIONS:
        excluded_desc = ", ".join([f"SOC{soc}_DOD{dod}" for soc, dod in EXCLUDED_CONDITIONS])
        print(f"🚫 已排除工况: {excluded_desc}")
    # ===========================================

    success_count = 0
    skipped_cycle_count = 0
    skipped_battery_no_cycle_count = 0
    skipped_battery_no_soh_count = 0
    skipped_interval_match_count = 0

    for battery_id in tqdm(battery_ids_to_process, desc="电池进度", unit="块"):
        battery_cycle_files = cycle_files_by_battery.get(battery_id, [])

        if not battery_cycle_files:
            skipped_battery_no_cycle_count += 1
            print(f"\nℹ️ {battery_id}: 没有找到 Aging 循环文件，跳过。")
            continue

        if battery_id not in soh_records:
            skipped_battery_no_soh_count += 1
            skipped_cycle_count += len(battery_cycle_files)
            print(f"\nℹ️ {battery_id}: 没有找到对应 SOH/checkup 文件，跳过该电池。")
            continue

        print(f"\n🔋 开始处理 {battery_id}: {len(battery_cycle_files)} 个 Aging 文件")

        interval_groups, skipped_match = group_cycle_files_by_soh_interval(
            battery_id,
            battery_cycle_files,
            soh_records[battery_id],
        )

        skipped_interval_match_count += skipped_match
        skipped_cycle_count += skipped_match

        if not interval_groups:
            print(f"⚠️ {battery_id}: 没有任何 Aging 文件能匹配到 checkup 区间。")
            continue

        for interval_key in sorted(interval_groups.keys()):
            interval_success, interval_skipped = process_soh_interval_group(
                battery_id,
                interval_groups[interval_key],
                BASE_OUTPUT_DIR,
            )

            success_count += interval_success
            skipped_cycle_count += interval_skipped

    print("\n" + "=" * 100)
    print("✅ 批量处理完成")
    print(f"📦 成功导出 CSV 的循环文件数: {success_count}")
    print(f"⚠️ 跳过循环文件数: {skipped_cycle_count}")
    print(f"ℹ️ 未匹配到 checkup 区间的循环文件数: {skipped_interval_match_count}")
    print(f"ℹ️ 没有循环文件的电池数: {skipped_battery_no_cycle_count}")
    print(f"ℹ️ 没有 SOH/checkup 文件的电池数: {skipped_battery_no_soh_count}")
    if EXCLUDED_CONDITIONS:
        excluded_desc = ", ".join([f"SOC{soc}_DOD{dod}" for soc, dod in EXCLUDED_CONDITIONS])
        print(f"🚫 已排除 {excluded_desc} 工况的文件")
    print(f"📁 输出文件夹: {BASE_OUTPUT_DIR}")
    
    # 报告不符合条件的片段
    print("\n" + "-" * 100)
    print("🔍 不符合 tau2 > 5*tau1 条件的片段统计:")
    print(f"总不符合条件的片段数: {len(UNQUALIFIED_SEGMENTS)}")
    
    if UNQUALIFIED_SEGMENTS:
        # 按电池分组统计
        battery_stats = {}
        for record in UNQUALIFIED_SEGMENTS:
            battery_id = record['segment_info'].split('_')[0]
            battery_stats[battery_id] = battery_stats.get(battery_id, 0) + 1
        
        print("\n按电池分组统计:")
        for battery_id, count in sorted(battery_stats.items()):
            print(f"  {battery_id}: {count} 个片段")
        
        # 创建不符合条件的片段汇总CSV
        unqualified_df = pd.DataFrame(UNQUALIFIED_SEGMENTS)
        unqualified_csv_path = os.path.join(BASE_OUTPUT_DIR, "unqualified_segments_summary.csv")
        unqualified_df.to_csv(
            unqualified_csv_path,
            index=False,
            sep=CSV_SEPARATOR,
            decimal=CSV_DECIMAL,
            encoding="utf-8-sig"
        )
        print(f"\n📄 不符合条件的片段详细列表已保存至: {unqualified_csv_path}")
        
        # 显示前10个不符合条件的片段
        print("\n前10个不符合条件的片段（按tau2/tau1比值从高到低）:")
        sorted_segments = sorted(UNQUALIFIED_SEGMENTS, key=lambda x: x['ratio'], reverse=True)
        for i, record in enumerate(sorted_segments[:10], 1):
            print(f"  {i}. {record['segment_info']}")
            print(f"     tau1={record['tau1']:.4f}, tau2={record['tau2']:.4f}, ratio={record['ratio']:.4f}")
        
        if len(UNQUALIFIED_SEGMENTS) > 10:
            print(f"  ... 还有 {len(UNQUALIFIED_SEGMENTS) - 10} 个片段，详见CSV文件")
    else:
        print("🎉 所有片段都满足 tau2 > 5*tau1 条件！")
    
    print("=" * 100)


if __name__ == "__main__":
    main()