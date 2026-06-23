"""
01_load_checkup_data_v2.py

用途：
1. 从 ISEA MinIO/S3 读取 checkup parquet 文件；
2. 统一列名：Zeit, Spannung, Strom, T1, Prozedur, Zustand, AhAkku, Ahjo_Test_ID；
3. 自动处理时间列、电流单位、电流符号；
4. 提取一个连续放电片段；
5. 去掉开头毛刺，重新积分容量；
6. 保存 one_checkup_discharge_segment.csv，供后续简化 P2D / PINN 使用。

重要：不要把 S3 access key / secret key 写死在代码里。
请先在终端设置环境变量：

Windows PowerShell:
    $env:MINIO_ACCESS_KEY="你的新access_key"
    $env:MINIO_SECRET_KEY="你的新secret_key"

Linux / macOS:
    export MINIO_ACCESS_KEY="你的新access_key"
    export MINIO_SECRET_KEY="你的新secret_key"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 配置
# ============================================================

@dataclass
class Config:
    # 这里填入 checkup parquet 的完整路径，包含 bucket 名，但不要加 s3:// 也可以
    # 例子："projects/.../checkup.parquet"
    target_file: str = "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_003/J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2025-06-01_154106=jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS037218 _ Format01=Kreis M3-034=filesize-107763566=finished.parquet"

    # 输出文件
    output_csv: str = "one_checkup_discharge_segment.csv"

    # 标称容量，用于估计 C-rate。Sony/Murata VTC6 通常可先用 3.0 Ah。
    nominal_capacity_ah: float = 3.0

    # 去掉放电段开头毛刺。你上一张图的开头有尖峰，建议先裁掉 10 s。
    trim_start_s: float = 10.0

    # 小于该电流认为是静置/噪声。低倍率 checkup 约 0.15 A，所以 0.02~0.05 A 都可以。
    min_current_abs_a: float = 0.02

    # 至少多少个点才算一个有效放电段
    min_points: int = 100

    # 如果文件里有多个 Ahjo_Test_ID，可以指定其中一个；None 表示自动在全文件里找最长放电段。
    test_id: Optional[str] = None

    # 多个候选放电段时如何选择：longest / first / last / index
    segment_choice: Literal["longest", "first", "last", "index"] = "longest"
    segment_index: int = 0

    # 电流单位：auto / A / mA
    current_unit: Literal["auto", "A", "mA"] = "auto"

    # 放电符号：auto / positive / negative
    # positive 表示原始 Strom 中放电为正；negative 表示原始 Strom 中放电为负。
    discharge_sign: Literal["auto", "positive", "negative"] = "auto"

    # MinIO endpoint
    s3_endpoint_url: str = os.getenv(
        "MINIO_ENDPOINT_URL",
        "https://iseadocker.isea.rwth-aachen.de:9000",
    )


# ============================================================
# 1. MinIO/S3 配置
# ============================================================

def get_s3_storage_options(endpoint_url: str):
    """
    从环境变量读取 S3 配置。
    不要把 access key 和 secret key 写进代码。
    """
    key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")

    if not key or not secret:
        raise ValueError(
            "没有找到 S3 密钥。请先设置环境变量 MINIO_ACCESS_KEY 和 MINIO_SECRET_KEY。\n"
            "Windows PowerShell:\n"
            "  $env:MINIO_ACCESS_KEY='你的access_key'\n"
            "  $env:MINIO_SECRET_KEY='你的secret_key'\n"
            "Linux/macOS:\n"
            "  export MINIO_ACCESS_KEY='你的access_key'\n"
            "  export MINIO_SECRET_KEY='你的secret_key'"
        )

    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "region_name": "us-east-1",
        },
        "config_kwargs": {
            "s3": {"addressing_style": "path"},
            "signature_version": "s3v4",
        },
    }


def to_s3_uri(target_file: str) -> str:
    """允许用户输入 s3://... 或不带 s3:// 的完整路径。"""
    if target_file.startswith("s3://"):
        return target_file
    return f"s3://{target_file}"


# ============================================================
# 2. 读取数据
# ============================================================

def read_battery_data(target_file: str, endpoint_url: str) -> pd.DataFrame:
    """
    使用 s3fs 从 MinIO 读取 parquet 文件。
    target_file 例子：projects/.../checkup.parquet
    """
    storage_options = get_s3_storage_options(endpoint_url)
    s3_path = to_s3_uri(target_file)

    print(f"📄 正在读取文件: {target_file.split('/')[-1]}")
    print(f"   S3 path: {s3_path}")

    df = pd.read_parquet(s3_path, storage_options=storage_options)

    print("✅ 成功读取数据")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   原始列名: {df.columns.tolist()}")

    return df


# ============================================================
# 3. 统一列名
# ============================================================

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    把原始列名转换成内部英文列名。
    你的 checkup 文件如果还有其他列名变体，可以继续加到 column_map。
    """
    df = df.copy()

    column_map = {
        # 德语列名
        "Zeit": "time",
        "Spannung": "voltage",
        "Strom": "current",
        "T1": "temperature",
        "Prozedur": "procedure",
        "Zustand": "state",
        "AhAkku": "capacity_ah",
        "Ahjo_Test_ID": "test_id",

        # 常见英文变体
        "Time": "time",
        "time": "time",
        "Voltage": "voltage",
        "voltage": "voltage",
        "Current": "current",
        "current": "current",
        "Temperature": "temperature",
        "temperature": "temperature",
        "Capacity": "capacity_ah",
        "capacity": "capacity_ah",
        "capacity_ah": "capacity_ah",
        "Test_ID": "test_id",
        "test_id": "test_id",
    }

    existing = df.columns.tolist()
    actual_map = {k: v for k, v in column_map.items() if k in existing}
    df = df.rename(columns=actual_map)

    print(f"✅ 列名映射完成: {actual_map}")

    required_cols = ["time", "voltage", "current"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少必需列: {missing}\n"
            f"当前列名: {df.columns.tolist()}\n"
            "请在 rename_columns() 的 column_map 里补充你的真实列名。"
        )

    return df


# ============================================================
# 4. 电流单位标准化
# ============================================================

def standardize_current_unit(
    df: pd.DataFrame,
    current_unit: Literal["auto", "A", "mA"] = "auto",
) -> pd.DataFrame:
    """
    统一电流单位为 A。

    如果 current_unit="auto"：
    - median(|current|) > 50 时，认为原始单位是 mA，除以 1000；
    - 否则认为已经是 A。

    你上一张图中放电电流约 0.15，明显已经是 A。
    """
    df = df.copy()
    df["current"] = pd.to_numeric(df["current"], errors="coerce")

    median_abs_current = df["current"].abs().median()

    if current_unit == "mA":
        df["current"] = df["current"] / 1000.0
        print("✅ 用户指定电流单位为 mA，已转换为 A。")
    elif current_unit == "A":
        print("✅ 用户指定电流单位为 A，不做转换。")
    else:
        if median_abs_current > 50:
            df["current"] = df["current"] / 1000.0
            print("✅ 自动检测：电流单位可能是 mA，已转换为 A。")
        else:
            print("✅ 自动检测：电流单位可能已经是 A，不做转换。")

    print(f"   |current| 中位数，转换后: {df['current'].abs().median():.6g} A")
    return df


# ============================================================
# 5. 时间列处理
# ============================================================

def _subtract_first_per_group(series: pd.Series, group_key: Optional[pd.Series]):
    """辅助函数：每个 test_id 内部从 0 开始计时。"""
    if group_key is None:
        return series - series.iloc[0]
    return series.groupby(group_key, sort=False).transform(lambda s: s - s.iloc[0])


def convert_time_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 time 列转换成从每个 test_id 开始计时的秒数 time_s。

    兼容：
    1. 数字秒；
    2. datetime；
    3. timedelta / 00:01:23.456。
    """
    df = df.copy()
    group_key = df["test_id"] if "test_id" in df.columns else None

    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        if group_key is None:
            df["time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
        else:
            df["time_s"] = df.groupby("test_id", sort=False)["time"].transform(
                lambda s: (s - s.iloc[0]).dt.total_seconds()
            )
        print("✅ time 列识别为 datetime，已转换为 time_s。")
        return df

    if pd.api.types.is_timedelta64_any_dtype(df["time"]):
        seconds = df["time"].dt.total_seconds()
        df["time_s"] = _subtract_first_per_group(seconds, group_key)
        print("✅ time 列识别为 timedelta，已转换为 time_s。")
        return df

    # 数字时间
    time_numeric = pd.to_numeric(df["time"], errors="coerce")
    if time_numeric.notna().mean() > 0.95:
        df["time_s"] = _subtract_first_per_group(time_numeric.astype(float), group_key)
        print("✅ time 列识别为数字，已转换为 time_s。")
        return df

    # datetime 字符串
    time_parsed = pd.to_datetime(df["time"], errors="coerce")
    if time_parsed.notna().mean() > 0.8:
        df["_time_parsed"] = time_parsed
        if group_key is None:
            df["time_s"] = (df["_time_parsed"] - df["_time_parsed"].iloc[0]).dt.total_seconds()
        else:
            df["time_s"] = df.groupby("test_id", sort=False)["_time_parsed"].transform(
                lambda s: (s - s.iloc[0]).dt.total_seconds()
            )
        df = df.drop(columns=["_time_parsed"])
        print("✅ time 列识别为 datetime 字符串，已转换为 time_s。")
        return df

    # timedelta 字符串
    time_delta = pd.to_timedelta(df["time"], errors="coerce")
    if time_delta.notna().mean() > 0.8:
        seconds = time_delta.dt.total_seconds()
        df["time_s"] = _subtract_first_per_group(seconds, group_key)
        print("✅ time 列识别为 timedelta 字符串，已转换为 time_s。")
        return df

    raise ValueError(f"time 列无法识别，dtype={df['time'].dtype}")


# ============================================================
# 6. 基础清洗
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    基础清洗：
    - 数值列转为 float；
    - 删除 time_s / voltage / current 缺失行；
    - 按 test_id 和 time_s 排序。
    """
    df = df.copy()

    numeric_cols = ["voltage", "current", "time_s"]
    for optional_col in ["temperature", "capacity_ah"]:
        if optional_col in df.columns:
            numeric_cols.append(optional_col)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["time_s", "voltage", "current"])
    after = len(df)
    if after < before:
        print(f"⚠️ 删除缺失关键值的行: {before - after}")

    sort_cols = ["test_id", "time_s"] if "test_id" in df.columns else ["time_s"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if "capacity_ah" not in df.columns:
        print("⚠️ 文件中没有 capacity_ah/AhAkku。后续将用 current_discharge 积分得到容量。")

    print("✅ 基础清洗完成")
    return df


# ============================================================
# 7. 判断电流符号：统一为放电为正
# ============================================================

def _detect_discharge_by_text(df: pd.DataFrame) -> Optional[str]:
    """
    尝试通过 Prozedur/Zustand 判断原始 current 的放电符号。
    返回：positive / negative / None
    """
    text_cols = []
    if "procedure" in df.columns:
        text_cols.append(df["procedure"].astype(str).str.lower())
    if "state" in df.columns:
        text_cols.append(df["state"].astype(str).str.lower())

    if not text_cols:
        return None

    text = text_cols[0]
    for extra in text_cols[1:]:
        text = text + " " + extra

    # 注意：entladen 包含 laden，所以优先只找放电关键词。
    discharge_keywords = [
        "entladen",
        "discharge",
        "discharging",
        "disch",
        "constant current discharge",
    ]

    discharge_mask = np.zeros(len(df), dtype=bool)
    for kw in discharge_keywords:
        discharge_mask |= text.str.contains(kw, na=False, regex=False).to_numpy()

    if discharge_mask.sum() <= 10:
        return None

    median_discharge_current = df.loc[discharge_mask, "current"].median()

    if median_discharge_current < 0:
        return "negative"
    return "positive"


def _detect_discharge_by_voltage_slope(
    df: pd.DataFrame,
    min_current_abs_a: float,
) -> Optional[str]:
    """
    当没有 procedure/state 时，尝试根据电压变化判断。
    放电段通常电压随时间下降。
    """
    one = df.sort_values("time_s").reset_index(drop=True).copy()
    d_v = one["voltage"].diff()

    pos_mask = one["current"] > min_current_abs_a
    neg_mask = one["current"] < -min_current_abs_a

    # 只看连续同号电流内部的电压差，避免工步切换干扰。
    pos_cont = pos_mask & pos_mask.shift(fill_value=False)
    neg_cont = neg_mask & neg_mask.shift(fill_value=False)

    pos_score = d_v[pos_cont].median() if pos_cont.sum() > 20 else np.nan
    neg_score = d_v[neg_cont].median() if neg_cont.sum() > 20 else np.nan

    # median dV < 0 的那一类更可能是放电。
    if np.isfinite(pos_score) and pos_score < 0 and not (np.isfinite(neg_score) and neg_score < 0):
        return "positive"
    if np.isfinite(neg_score) and neg_score < 0 and not (np.isfinite(pos_score) and pos_score < 0):
        return "negative"

    return None


def add_discharge_current(
    df: pd.DataFrame,
    min_current_abs_a: float = 0.02,
    discharge_sign: Literal["auto", "positive", "negative"] = "auto",
) -> pd.DataFrame:
    """
    建立统一符号约定：current_discharge > 0 表示放电。
    """
    df = df.copy()

    if discharge_sign in ["positive", "negative"]:
        detected = discharge_sign
        print(f"✅ 用户指定：原始 Strom 中放电为 {detected}。")
    else:
        detected = _detect_discharge_by_text(df)
        if detected is not None:
            print(f"✅ 通过 Prozedur/Zustand 检测到：原始 Strom 中放电为 {detected}。")
        else:
            detected = _detect_discharge_by_voltage_slope(df, min_current_abs_a)
            if detected is not None:
                print(f"✅ 通过电压下降趋势检测到：原始 Strom 中放电为 {detected}。")
            else:
                detected = "negative"
                print("⚠️ 未能自动判断放电方向，默认假设原始 Strom 中放电为 negative。")

    if detected == "negative":
        df["current_discharge"] = -df["current"]
    else:
        df["current_discharge"] = df["current"]

    print(f"   current_discharge 中位数: {df['current_discharge'].median():.6g} A")
    return df


# ============================================================
# 8. 提取连续放电片段
# ============================================================

def _capacity_from_current(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """由 A 和 s 积分得到 Ah，不再除以 1000。"""
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.maximum(dt, 0.0)
    current_a = np.asarray(current_a, dtype=float)
    current_a = np.maximum(current_a, 0.0)
    return np.cumsum(current_a * dt / 3600.0)


def _make_segment_summary(candidates: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, seg in enumerate(candidates):
        cap = _capacity_from_current(
            seg["time_s"].to_numpy(dtype=float),
            seg["current_discharge"].to_numpy(dtype=float),
        )[-1]
        rows.append({
            "idx": i,
            "test_id": str(seg["test_id"].iloc[0]) if "test_id" in seg.columns else "",
            "points": len(seg),
            "duration_h": seg["time_s"].iloc[-1] / 3600.0,
            "mean_I_A": seg["current_discharge"].mean(),
            "cap_Ah_from_I": cap,
            "V_start": seg["voltage"].iloc[0],
            "V_end": seg["voltage"].iloc[-1],
        })
    return pd.DataFrame(rows)


def extract_one_discharge_segment(
    df: pd.DataFrame,
    test_id: Optional[str] = None,
    min_current_abs_a: float = 0.02,
    min_points: int = 100,
    segment_choice: Literal["longest", "first", "last", "index"] = "longest",
    segment_index: int = 0,
) -> pd.DataFrame:
    """
    提取一个连续放电片段。

    对 checkup 文件，可能存在多个工步/片段，因此这里会先列出候选段，
    默认选择最长放电段。
    """
    df = df.copy()

    if test_id is not None and "test_id" in df.columns:
        df = df[df["test_id"].astype(str) == str(test_id)].copy()
        if len(df) == 0:
            raise ValueError(f"没有找到 test_id={test_id} 的数据。")

    group_cols = ["test_id"] if "test_id" in df.columns else [None]
    candidates = []

    if group_cols == [None]:
        grouped = [(None, df)]
    else:
        grouped = list(df.groupby("test_id", sort=False))

    for _, one in grouped:
        one = one.sort_values("time_s").reset_index(drop=True).copy()
        is_discharge = one["current_discharge"] > min_current_abs_a
        segment_id = (is_discharge != is_discharge.shift(fill_value=False)).cumsum()
        one["segment_id"] = segment_id

        for _, group in one.groupby("segment_id", sort=False):
            if (
                len(group) >= min_points
                and group["current_discharge"].mean() > min_current_abs_a
                and group["voltage"].iloc[-1] < group["voltage"].iloc[0]
            ):
                candidates.append(group.copy())

    if not candidates:
        raise ValueError(
            "没有找到有效放电片段。请检查：\n"
            "1. current_discharge 是否放电为正；\n"
            "2. min_current_abs_a 是否过高；\n"
            "3. min_points 是否过高；\n"
            "4. checkup 文件里是否确实包含放电工步。"
        )

    summary = _make_segment_summary(candidates)
    print("\n📊 候选放电片段:")
    print(summary.to_string(index=False))

    if segment_choice == "longest":
        chosen_idx = int(summary["points"].idxmax())
    elif segment_choice == "first":
        chosen_idx = 0
    elif segment_choice == "last":
        chosen_idx = len(candidates) - 1
    else:
        if segment_index < 0 or segment_index >= len(candidates):
            raise ValueError(f"segment_index 越界。可选范围: 0 到 {len(candidates)-1}")
        chosen_idx = segment_index

    seg = candidates[chosen_idx].copy()
    seg = seg.sort_values("time_s").reset_index(drop=True)
    seg["time_s"] = seg["time_s"] - seg["time_s"].iloc[0]

    print(f"\n✅ 选中放电片段 idx={chosen_idx}")
    print(f"   点数: {len(seg)}")
    print(f"   时长: {seg['time_s'].iloc[-1] / 3600:.3f} h")
    print(f"   平均放电电流: {seg['current_discharge'].mean():.6g} A")

    return seg


# ============================================================
# 9. 给 P2D/PINN 使用前的最终整理
# ============================================================

def finalize_segment_for_p2d(
    seg: pd.DataFrame,
    nominal_capacity_ah: float = 3.0,
    trim_start_s: float = 10.0,
    save_path: str = "one_checkup_discharge_segment.csv",
) -> pd.DataFrame:
    """
    最终整理：
    1. 去掉开头毛刺；
    2. 时间重新从 0 开始；
    3. 用 current_discharge 重新积分容量；
    4. AhAkku 容量也重新归零；
    5. 估计 C-rate；
    6. 保存后续 P2D 所需核心列。
    """
    seg = seg.copy().sort_values("time_s").reset_index(drop=True)

    if trim_start_s > 0:
        before = len(seg)
        seg = seg[seg["time_s"] >= trim_start_s].copy().reset_index(drop=True)
        after = len(seg)
        print(f"\n✂️ 已裁掉开头 {trim_start_s:.1f} s，删除点数: {before - after}")

    if len(seg) < 5:
        raise ValueError("裁剪后数据点太少，请减小 trim_start_s。")

    seg["time_s"] = seg["time_s"] - seg["time_s"].iloc[0]

    I = seg["current_discharge"].to_numpy(dtype=float)
    t = seg["time_s"].to_numpy(dtype=float)

    seg["capacity_from_current_ah"] = _capacity_from_current(t, I)

    if "capacity_ah" in seg.columns and seg["capacity_ah"].notna().sum() > 5:
        seg["capacity_relative_ah"] = seg["capacity_ah"] - seg["capacity_ah"].iloc[0]
        if seg["capacity_relative_ah"].iloc[-1] < 0:
            seg["capacity_relative_ah"] = -seg["capacity_relative_ah"]
    else:
        seg["capacity_relative_ah"] = seg["capacity_from_current_ah"]

    mean_current = float(np.mean(I))
    estimated_c_rate = mean_current / nominal_capacity_ah

    print("\n✅ P2D 输入片段整理完成:")
    print(f"   点数: {len(seg)}")
    print(f"   时长: {seg['time_s'].iloc[-1] / 3600:.3f} h")
    print(f"   平均电流: {mean_current:.6g} A")
    print(f"   估计 C-rate: {estimated_c_rate:.5f} C")
    print(f"   电流积分容量: {seg['capacity_from_current_ah'].iloc[-1]:.6g} Ah")
    print(f"   AhAkku/相对容量: {seg['capacity_relative_ah'].iloc[-1]:.6g} Ah")

    keep_cols = [
        "time_s",
        "voltage",
        "current_discharge",
        "capacity_relative_ah",
        "capacity_from_current_ah",
    ]

    optional_cols = ["temperature", "procedure", "state", "test_id", "segment_id"]
    for col in optional_cols:
        if col in seg.columns and col not in keep_cols:
            keep_cols.append(col)

    seg_out = seg[keep_cols].copy()
    seg_out.to_csv(save_path, index=False)

    print(f"✅ 已保存: {save_path}")
    return seg_out


# ============================================================
# 10. 画图检查
# ============================================================

def plot_segment(seg: pd.DataFrame) -> None:
    """画图检查：电压、电流、容量，以及容量-电压曲线。"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(seg["time_s"] / 3600.0, seg["voltage"], linewidth=1)
    axes[0].set_ylabel("Voltage / V")
    axes[0].set_title("Discharge Voltage")
    axes[0].grid(True)

    axes[1].plot(seg["time_s"] / 3600.0, seg["current_discharge"], linewidth=1)
    axes[1].set_ylabel("Discharge current / A")
    axes[1].set_title("Discharge Current")
    axes[1].grid(True)

    cap_col = "capacity_from_current_ah"
    axes[2].plot(seg["time_s"] / 3600.0, seg[cap_col], linewidth=1)
    axes[2].set_ylabel("Capacity / Ah")
    axes[2].set_xlabel("Time / h")
    axes[2].set_title("Discharged Capacity")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(seg["capacity_from_current_ah"], seg["voltage"], linewidth=1.5, label="from current")
    if "capacity_relative_ah" in seg.columns:
        plt.plot(seg["capacity_relative_ah"], seg["voltage"], linewidth=1.0, linestyle="--", label="from AhAkku")
    plt.xlabel("Discharged capacity / Ah")
    plt.ylabel("Voltage / V")
    plt.title("Discharge Curve: Voltage vs Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. 主流程
# ============================================================

def run(config: Config) -> pd.DataFrame:
    df = read_battery_data(config.target_file, config.s3_endpoint_url)
    df = rename_columns(df)
    df = standardize_current_unit(df, current_unit=config.current_unit)
    df = convert_time_to_seconds(df)
    df = clean_data(df)
    df = add_discharge_current(
        df,
        min_current_abs_a=config.min_current_abs_a,
        discharge_sign=config.discharge_sign,
    )

    seg = extract_one_discharge_segment(
        df,
        test_id=config.test_id,
        min_current_abs_a=config.min_current_abs_a,
        min_points=config.min_points,
        segment_choice=config.segment_choice,
        segment_index=config.segment_index,
    )

    seg_out = finalize_segment_for_p2d(
        seg,
        nominal_capacity_ah=config.nominal_capacity_ah,
        trim_start_s=config.trim_start_s,
        save_path=config.output_csv,
    )

    plot_segment(seg_out)
    return seg_out


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Load and preprocess ISEA checkup battery data for simplified P2D/PINN.")

    parser.add_argument("--target-file", type=str, default=Config.target_file, help="S3 path to checkup parquet file.")
    parser.add_argument("--output-csv", type=str, default=Config.output_csv)
    parser.add_argument("--nominal-capacity-ah", type=float, default=Config.nominal_capacity_ah)
    parser.add_argument("--trim-start-s", type=float, default=Config.trim_start_s)
    parser.add_argument("--min-current-abs-a", type=float, default=Config.min_current_abs_a)
    parser.add_argument("--min-points", type=int, default=Config.min_points)
    parser.add_argument("--test-id", type=str, default=None)
    parser.add_argument("--segment-choice", type=str, choices=["longest", "first", "last", "index"], default=Config.segment_choice)
    parser.add_argument("--segment-index", type=int, default=Config.segment_index)
    parser.add_argument("--current-unit", type=str, choices=["auto", "A", "mA"], default=Config.current_unit)
    parser.add_argument("--discharge-sign", type=str, choices=["auto", "positive", "negative"], default=Config.discharge_sign)
    parser.add_argument("--s3-endpoint-url", type=str, default=Config.s3_endpoint_url)

    args = parser.parse_args()

    return Config(
        target_file=args.target_file,
        output_csv=args.output_csv,
        nominal_capacity_ah=args.nominal_capacity_ah,
        trim_start_s=args.trim_start_s,
        min_current_abs_a=args.min_current_abs_a,
        min_points=args.min_points,
        test_id=args.test_id,
        segment_choice=args.segment_choice,
        segment_index=args.segment_index,
        current_unit=args.current_unit,
        discharge_sign=args.discharge_sign,
        s3_endpoint_url=args.s3_endpoint_url,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
