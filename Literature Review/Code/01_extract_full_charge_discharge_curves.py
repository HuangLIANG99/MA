import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 如果没有安装，请先运行：
# pip install s3fs pyarrow pandas numpy matplotlib


# ============================================================
# 0. 用户配置区
# ============================================================

TARGET_FILE = (
    "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_003/J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2025-06-01_154106=jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS037218 _ Format01=Kreis M3-034=filesize-107763566=finished.parquet"
)

OUTPUT_DIR = "full_charge_discharge_output"

# 电流阈值：小于这个值认为是静置
MIN_CURRENT_ABS = 0.05

# 片段至少多少点才保留
MIN_POINTS = 100

# 片段至少持续多少秒才保留
MIN_DURATION_S = 30.0

# 标称容量，用来估计 C-rate
NOMINAL_CAPACITY_AH = 3.0

# 如果自动判断电流方向失败，可以手动改：
# None: 自动判断
# -1: 原始 Strom 中放电为负
# +1: 原始 Strom 中放电为正
RAW_DISCHARGE_SIGN = None

# 如果无法自动判断，默认假设原始数据中放电为负
DEFAULT_RAW_DISCHARGE_SIGN = -1

# 画图时最多绘制多少个点，防止大文件绘图太慢
MAX_PLOT_POINTS = 200000


# ============================================================
# 1. MinIO / S3 配置
# ============================================================

def get_s3_storage_options():
    """
    从环境变量读取 MinIO/S3 密钥。

    Windows PowerShell 示例：
    $env:MINIO_ACCESS_KEY="你的新access_key"
    $env:MINIO_SECRET_KEY="你的新secret_key"

    Linux / macOS 示例：
    export MINIO_ACCESS_KEY="你的新access_key"
    export MINIO_SECRET_KEY="你的新secret_key"
    """

    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")

    if key is None or secret is None:
        raise ValueError(
            "没有找到环境变量 MINIO_ACCESS_KEY 或 MINIO_SECRET_KEY。\n"
            "请不要把密钥写死在代码里。请先在终端设置环境变量。"
        )

    storage_options = {
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

    return storage_options


# ============================================================
# 2. 读取 Parquet 数据
# ============================================================

def read_battery_data(target_file: str):
    """
    使用 s3fs 从 MinIO 读取 Parquet 文件。

    target_file 可以是：
    - projects/.../finished.parquet
    - s3://projects/.../finished.parquet
    """

    storage_options = get_s3_storage_options()

    if target_file.startswith("s3://"):
        s3_path = target_file
    else:
        s3_path = f"s3://{target_file}"

    print(f"📄 正在读取文件:")
    print(f"   {s3_path}")

    df = pd.read_parquet(
        s3_path,
        storage_options=storage_options
    )

    print("✅ 成功读取数据")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   原始列名: {df.columns.tolist()}")

    return df


# ============================================================
# 3. 统一列名
# ============================================================

def rename_columns(df):
    """
    把原始列名统一成代码内部使用的英文列名。
    """

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

        # 兼容可能出现的英文列名
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

    print("\n✅ 列名映射完成:")
    print(actual_map)

    required_cols = ["time", "voltage", "current"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"缺少必需列: {missing}\n"
            f"当前列名为: {df.columns.tolist()}"
        )

    return df


# ============================================================
# 4. 电流单位标准化
# ============================================================

def standardize_current_unit(df):
    """
    统一电流单位为 A。

    如果电流绝对值中位数大于 50，通常说明单位可能是 mA，
    此时除以 1000。
    """

    df = df.copy()

    current_abs_median = df["current"].abs().median()

    if current_abs_median > 50:
        df["current"] = df["current"] / 1000.0
        print("\n✅ 检测到电流单位可能是 mA，已转换为 A。")
    else:
        print("\n✅ 检测到电流单位可能已经是 A，不做转换。")

    print(f"   电流绝对值中位数: {df['current'].abs().median():.6f} A")

    return df


# ============================================================
# 5. 时间列转换
# ============================================================

def convert_time_to_seconds(df):
    """
    把 time 列转换成从文件开始计时的秒数 time_s。
    """

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


# ============================================================
# 6. 基础清洗
# ============================================================

def clean_data(df):
    """
    基础清洗：
    - 数值列转为 float
    - 删除关键缺失值
    - 按 test_id 和 time_s 排序
    """

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

    if "capacity_ah" not in df.columns:
        print("\n📊 原始文件没有 AhAkku，先从原始 current 积分生成 capacity_ah。")
        dt = df["time_s"].diff().fillna(0).to_numpy()
        dt = np.maximum(dt, 0)
        df["capacity_ah"] = np.cumsum(
            df["current"].to_numpy(dtype=float) * dt / 3600.0
        )

    return df


# ============================================================
# 7. 判断电流符号：统一为放电为正、充电为负
# ============================================================

def infer_raw_discharge_sign(df, default_raw_discharge_sign=-1):
    """
    判断原始 current 中放电是正还是负。

    返回值：
    +1: 原始 current 中放电为正
    -1: 原始 current 中放电为负
    """

    if "procedure" not in df.columns and "state" not in df.columns:
        print("\n⚠️ 没有 procedure/state，无法自动判断放电方向。")
        return default_raw_discharge_sign

    text_parts = []

    if "procedure" in df.columns:
        text_parts.append(df["procedure"].astype(str).str.lower())

    if "state" in df.columns:
        text_parts.append(df["state"].astype(str).str.lower())

    text = text_parts[0]
    for part in text_parts[1:]:
        text = text + " " + part

    # 注意：德语 entladen 包含 laden，所以这里只用放电关键词判断
    discharge_keywords = [
        "entladen",
        "discharge",
        "discharging",
        "dc discharge",
        "dch",
    ]

    discharge_mask = np.zeros(len(df), dtype=bool)

    for kw in discharge_keywords:
        discharge_mask |= text.str.contains(kw, na=False).to_numpy()

    if discharge_mask.sum() > 10:
        median_discharge_current = df.loc[discharge_mask, "current"].median()

        if median_discharge_current > 0:
            print("\n✅ 自动判断：原始 current 中放电为正。")
            return +1
        else:
            print("\n✅ 自动判断：原始 current 中放电为负。")
            return -1

    print("\n⚠️ 没有从 procedure/state 中可靠识别放电段。")
    print(f"   使用默认设置: raw_discharge_sign = {default_raw_discharge_sign}")

    return default_raw_discharge_sign


def add_discharge_current(
    df,
    raw_discharge_sign=None,
    default_raw_discharge_sign=-1
):
    """
    添加 current_discharge 列。

    约定：
    current_discharge > 0 表示放电
    current_discharge < 0 表示充电
    current_discharge ≈ 0 表示静置
    """

    df = df.copy()

    if raw_discharge_sign is None:
        raw_discharge_sign = infer_raw_discharge_sign(
            df,
            default_raw_discharge_sign=default_raw_discharge_sign
        )
    else:
        print(f"\n✅ 使用手动指定的 raw_discharge_sign = {raw_discharge_sign}")

    if raw_discharge_sign not in [-1, +1]:
        raise ValueError("raw_discharge_sign 只能是 -1、+1 或 None")

    if raw_discharge_sign == -1:
        df["current_discharge"] = -df["current"]
        print("   已转换：current_discharge = -current")
    else:
        df["current_discharge"] = df["current"]
        print("   已转换：current_discharge = current")

    return df


# ============================================================
# 8. 打印电流统计
# ============================================================

def print_current_statistics(df):
    """
    打印电流统计，用于检查单位和方向。
    """

    print("\n📌 current_discharge 统计:")
    print(df["current_discharge"].describe())

    print(
        f"\n   最小值: {df['current_discharge'].min():.6f} A"
        f"\n   最大值: {df['current_discharge'].max():.6f} A"
        f"\n   平均值: {df['current_discharge'].mean():.6f} A"
        f"\n   标准差: {df['current_discharge'].std():.6f} A"
    )


# ============================================================
# 9. 标注 charge / discharge / rest
# ============================================================

def add_operation_label(df, min_current_abs=0.05):
    """
    根据 current_discharge 标注工况。

    current_discharge > 0: discharge
    current_discharge < 0: charge
    接近 0: rest
    """

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

    print("\n📌 工况统计:")
    print(df["operation"].value_counts())

    return df


# ============================================================
# 10. 降采样辅助函数，仅用于画图
# ============================================================

def downsample_for_plot(df, max_points=200000):
    """
    只用于画图降采样，不改变原始数据。
    """

    if len(df) <= max_points:
        return df

    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


# ============================================================
# 11. 画完整文件时间序列
# ============================================================

def plot_full_file_timeseries(
    df,
    output_dir=None,
    max_plot_points=200000
):
    """
    画完整文件的：
    - Voltage vs Time
    - Current vs Time
    - AhAkku vs Time
    """

    plot_df = downsample_for_plot(df, max_points=max_plot_points)

    t_h = plot_df["time_s"] / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t_h, plot_df["voltage"], linewidth=0.8)
    axes[0].set_ylabel("Voltage / V")
    axes[0].set_title("Full File: Voltage")
    axes[0].grid(True)

    axes[1].plot(t_h, plot_df["current_discharge"], linewidth=0.8)
    axes[1].set_ylabel("Discharge-positive current / A")
    axes[1].set_title("Full File: Current")
    axes[1].grid(True)
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)

    axes[2].plot(t_h, plot_df["capacity_ah"], linewidth=0.8)
    axes[2].set_ylabel("AhAkku / Ah")
    axes[2].set_xlabel("Time / h")
    axes[2].set_title("Full File: AhAkku")
    axes[2].grid(True)

    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "full_file_timeseries.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ 已保存图像: {save_path}")

    plt.show()


# ============================================================
# 12. 提取完整文件所有充放电片段
# ============================================================

def extract_all_charge_discharge_segments(
    df,
    min_current_abs=0.05,
    min_points=100,
    min_duration_s=30.0,
    nominal_capacity_ah=3.0
):
    """
    提取完整文件中的所有连续充电/放电片段。

    输出：
    segments_df:
        所有片段的逐点数据

    summary_df:
        每个片段一行统计信息
    """

    df = df.copy()

    if "operation" not in df.columns:
        df = add_operation_label(df, min_current_abs=min_current_abs)

    all_segments = []
    summary_rows = []

    global_segment_id = 0

    if "test_id" in df.columns:
        group_iterator = df.groupby("test_id", sort=False)
    else:
        group_iterator = [("unknown", df)]

    for test_id, one in group_iterator:
        one = one.sort_values("time_s").reset_index(drop=True)

        if len(one) == 0:
            continue

        one["local_segment_id"] = (
            one["operation"] != one["operation"].shift(
                fill_value=one["operation"].iloc[0]
            )
        ).cumsum()

        for local_segment_id, g in one.groupby("local_segment_id", sort=False):
            mode = g["operation"].iloc[0]

            if mode not in ["charge", "discharge"]:
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
            dt = np.maximum(dt, 0)

            I_signed = g["current_discharge"].to_numpy(dtype=float)
            I_abs = np.abs(I_signed)

            # 片段容量统一用电流积分得到，避免 AhAkku 方向混乱
            g["capacity_segment_ah"] = np.cumsum(I_abs * dt / 3600.0)

            # 如果原始 AhAkku 存在，也保存片段内相对 AhAkku
            if "capacity_ah" in g.columns:
                cap_from_ahakku = g["capacity_ah"] - g["capacity_ah"].iloc[0]
                cap_from_ahakku = cap_from_ahakku.to_numpy(dtype=float)

                if cap_from_ahakku[-1] < 0:
                    cap_from_ahakku = -cap_from_ahakku

                g["capacity_segment_from_ahakku_ah"] = cap_from_ahakku

            else:
                g["capacity_segment_from_ahakku_ah"] = np.nan

            g["global_segment_id"] = global_segment_id
            g["mode"] = mode

            cap_ah = g["capacity_segment_ah"].iloc[-1]
            mean_abs_current = I_abs.mean()
            mean_signed_current = I_signed.mean()
            estimated_c_rate = mean_abs_current / nominal_capacity_ah

            row = {
                "global_segment_id": global_segment_id,
                "test_id": test_id,
                "mode": mode,
                "start_time_h": g["time_s"].iloc[0] / 3600.0,
                "end_time_h": g["time_s"].iloc[-1] / 3600.0,
                "duration_h": duration_s / 3600.0,
                "points": len(g),
                "capacity_ah": cap_ah,
                "capacity_from_ahakku_ah": g["capacity_segment_from_ahakku_ah"].iloc[-1],
                "mean_abs_current_A": mean_abs_current,
                "mean_signed_current_A": mean_signed_current,
                "estimated_C_rate": estimated_c_rate,
                "voltage_start_V": g["voltage"].iloc[0],
                "voltage_end_V": g["voltage"].iloc[-1],
                "voltage_min_V": g["voltage"].min(),
                "voltage_max_V": g["voltage"].max(),
            }

            if "temperature" in g.columns:
                row["temperature_mean_C"] = g["temperature"].mean()
                row["temperature_min_C"] = g["temperature"].min()
                row["temperature_max_C"] = g["temperature"].max()

            summary_rows.append(row)
            all_segments.append(g)

            global_segment_id += 1

    if len(all_segments) == 0:
        raise ValueError(
            "没有提取到有效充放电片段。\n"
            "请检查：\n"
            "1. 电流方向是否正确；\n"
            "2. MIN_CURRENT_ABS 是否太大；\n"
            "3. MIN_POINTS / MIN_DURATION_S 是否太严格。"
        )

    segments_df = pd.concat(all_segments, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    print("\n✅ 已提取完整文件中的充放电片段:")
    print(summary_df["mode"].value_counts())

    print("\n📋 前 10 个片段统计:")
    print(summary_df.head(10))

    return segments_df, summary_df


# ============================================================
# 13. 画所有放电曲线
# ============================================================

def plot_all_discharge_curves(
    segments_df,
    output_dir=None,
    max_curves=None
):
    """
    画所有放电曲线：
    Voltage vs Discharged capacity
    """

    dis = segments_df[segments_df["mode"] == "discharge"].copy()

    if len(dis) == 0:
        print("⚠️ 没有放电片段。")
        return

    segment_ids = dis["global_segment_id"].unique()

    if max_curves is not None:
        segment_ids = segment_ids[:max_curves]

    plt.figure(figsize=(8, 6))

    for sid in segment_ids:
        g = dis[dis["global_segment_id"] == sid]

        plt.plot(
            g["capacity_segment_ah"],
            g["voltage"],
            linewidth=0.8,
            alpha=0.35,
        )

    plt.xlabel("Discharged capacity / Ah")
    plt.ylabel("Voltage / V")
    plt.title("All Discharge Curves in This File")
    plt.grid(True)
    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "all_discharge_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ 已保存图像: {save_path}")

    plt.show()


# ============================================================
# 14. 画所有充电曲线
# ============================================================

def plot_all_charge_curves(
    segments_df,
    output_dir=None,
    max_curves=None
):
    """
    画所有充电曲线：
    Voltage vs Charged capacity
    """

    chg = segments_df[segments_df["mode"] == "charge"].copy()

    if len(chg) == 0:
        print("⚠️ 没有充电片段。")
        return

    segment_ids = chg["global_segment_id"].unique()

    if max_curves is not None:
        segment_ids = segment_ids[:max_curves]

    plt.figure(figsize=(8, 6))

    for sid in segment_ids:
        g = chg[chg["global_segment_id"] == sid]

        plt.plot(
            g["capacity_segment_ah"],
            g["voltage"],
            linewidth=0.8,
            alpha=0.35,
        )

    plt.xlabel("Charged capacity / Ah")
    plt.ylabel("Voltage / V")
    plt.title("All Charge Curves in This File")
    plt.grid(True)
    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "all_charge_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ 已保存图像: {save_path}")

    plt.show()


# ============================================================
# 15. 画充放电片段容量统计
# ============================================================

def plot_segment_capacity_summary(
    summary_df,
    output_dir=None
):
    """
    画每个片段的容量。
    """

    plt.figure(figsize=(10, 5))

    dis = summary_df[summary_df["mode"] == "discharge"]
    chg = summary_df[summary_df["mode"] == "charge"]

    if len(dis) > 0:
        plt.plot(
            dis["global_segment_id"],
            dis["capacity_ah"],
            marker="o",
            linestyle="-",
            label="discharge",
        )

    if len(chg) > 0:
        plt.plot(
            chg["global_segment_id"],
            chg["capacity_ah"],
            marker="o",
            linestyle="-",
            label="charge",
        )

    plt.xlabel("Segment index")
    plt.ylabel("Segment capacity / Ah")
    plt.title("Charge / Discharge Capacity of All Segments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, "segment_capacity_summary.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ 已保存图像: {save_path}")

    plt.show()


# ============================================================
# 16. 单独画某一个片段
# ============================================================

def plot_one_segment(
    segments_df,
    segment_id,
    output_dir=None
):
    """
    检查某一个充电或放电片段。
    """

    g = segments_df[
        segments_df["global_segment_id"] == segment_id
    ].copy()

    if len(g) == 0:
        print(f"⚠️ 没有找到 segment_id = {segment_id}")
        return

    mode = g["mode"].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    t_h = g["time_segment_s"] / 3600.0

    axes[0].plot(t_h, g["voltage"], linewidth=1)
    axes[0].set_ylabel("Voltage / V")
    axes[0].set_title(f"Segment {segment_id}: {mode} voltage")
    axes[0].grid(True)

    axes[1].plot(t_h, g["current_discharge"], linewidth=1)
    axes[1].set_ylabel("Current / A")
    axes[1].set_title("Current, discharge positive")
    axes[1].grid(True)
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)

    axes[2].plot(t_h, g["capacity_segment_ah"], linewidth=1)
    axes[2].set_ylabel("Capacity / Ah")
    axes[2].set_xlabel("Time / h")
    axes[2].set_title("Segment capacity from current")
    axes[2].grid(True)

    plt.tight_layout()

    if output_dir is not None:
        save_path = os.path.join(output_dir, f"segment_{segment_id}_{mode}.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ 已保存图像: {save_path}")

    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(g["capacity_segment_ah"], g["voltage"], linewidth=1.2)
    plt.xlabel("Segment capacity / Ah")
    plt.ylabel("Voltage / V")
    plt.title(f"Segment {segment_id}: Voltage vs Capacity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 17. 保存结果
# ============================================================

def save_results(df, segments_df, summary_df, output_dir):
    """
    保存完整处理结果。
    """

    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "full_file_preprocessed.csv")
    segments_path = os.path.join(output_dir, "all_charge_discharge_segments.csv")
    summary_path = os.path.join(output_dir, "charge_discharge_segment_summary.csv")

    df.to_csv(full_path, index=False)
    segments_df.to_csv(segments_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n✅ 已保存 CSV 文件:")
    print(f"   {full_path}")
    print(f"   {segments_path}")
    print(f"   {summary_path}")


# ============================================================
# 18. 主程序
# ============================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔍 启动完整文件充放电曲线提取程序...\n")

    # 1. 读取数据
    df = read_battery_data(TARGET_FILE)

    # 2. 统一列名
    df = rename_columns(df)

    # 3. 统一电流单位
    df = standardize_current_unit(df)

    # 4. 时间转换
    df = convert_time_to_seconds(df)

    # 5. 基础清洗
    df = clean_data(df)

    # 6. 统一电流符号：放电为正，充电为负
    df = add_discharge_current(
        df,
        raw_discharge_sign=RAW_DISCHARGE_SIGN,
        default_raw_discharge_sign=DEFAULT_RAW_DISCHARGE_SIGN,
    )

    # 7. 打印电流统计
    print_current_statistics(df)

    # 8. 标注 charge / discharge / rest
    df = add_operation_label(
        df,
        min_current_abs=MIN_CURRENT_ABS,
    )

    # 9. 画完整文件时间序列
    plot_full_file_timeseries(
        df,
        output_dir=OUTPUT_DIR,
        max_plot_points=MAX_PLOT_POINTS,
    )

    # 10. 提取完整文件所有充放电片段
    segments_df, summary_df = extract_all_charge_discharge_segments(
        df,
        min_current_abs=MIN_CURRENT_ABS,
        min_points=MIN_POINTS,
        min_duration_s=MIN_DURATION_S,
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
    )

    # 11. 保存 CSV
    save_results(
        df,
        segments_df,
        summary_df,
        output_dir=OUTPUT_DIR,
    )

    # 12. 画所有放电曲线
    plot_all_discharge_curves(
        segments_df,
        output_dir=OUTPUT_DIR,
        max_curves=None,
    )

    # 13. 画所有充电曲线
    plot_all_charge_curves(
        segments_df,
        output_dir=OUTPUT_DIR,
        max_curves=None,
    )

    # 14. 画每个片段容量
    plot_segment_capacity_summary(
        summary_df,
        output_dir=OUTPUT_DIR,
    )

    # 15. 可选：检查第一个有效片段
    first_segment_id = int(summary_df["global_segment_id"].iloc[0])
    plot_one_segment(
        segments_df,
        segment_id=first_segment_id,
        output_dir=OUTPUT_DIR,
    )

    print("\n🎉 完整文件充放电曲线提取完成。")
    print(f"结果文件夹: {OUTPUT_DIR}")
