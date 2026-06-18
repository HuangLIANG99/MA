import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_FOLDER = "projects/j8005-metabatt/Metabatt/VTC/30_export_qocv/METABatt_Sony_Murata_18650VTC6_003"
OUTPUT_DIR = "output_dch_ocv"


def get_s3_storage_options():
    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")

    if key is None or secret is None:
        raise ValueError("请先设置环境变量 MINIO_ACCESS_KEY 和 MINIO_SECRET_KEY")

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


def to_s3_path(path):
    return path if path.startswith("s3://") else f"s3://{path}"


def strip_s3_prefix(path):
    return path.replace("s3://", "", 1)


def list_parquet_files(folder):
    if os.path.exists(folder):
        files = []
        for root, _, names in os.walk(folder):
            for name in names:
                if name.endswith(".parquet"):
                    files.append(os.path.join(root, name))
        return files

    import s3fs
    fs = s3fs.S3FileSystem(**get_s3_storage_options())
    folder = strip_s3_prefix(folder)
    return [f for f in fs.find(folder) if f.endswith(".parquet")]


def is_dch_file(path):
    name = os.path.basename(path).lower()
    return re.search(r"(^|_)dch(_|\.|$)", name) is not None


def extract_metadata(path):
    name = os.path.basename(path)

    battery_match = re.search(r"(18650VTC6_\d+)", name)
    battery_id = battery_match.group(1) if battery_match else os.path.splitext(name)[0]

    soh_match = re.search(r"(\d+(?:\.\d+)?)SOH", name, re.IGNORECASE)
    soh = float(soh_match.group(1)) if soh_match else np.nan

    state_match = re.search(r"_(cha|dch)(?:_|\.|$)", name, re.IGNORECASE)
    state = state_match.group(1).lower() if state_match else "unknown"

    return battery_id, soh, state


def read_parquet_file(path):
    if os.path.exists(path):
        return pd.read_parquet(path)

    return pd.read_parquet(
        to_s3_path(path),
        storage_options=get_s3_storage_options()
    )


def normalize_to_soc(series):
    series = pd.to_numeric(series, errors="coerce")
    v_min = series.min()
    v_max = series.max()

    if pd.isna(v_min) or pd.isna(v_max) or v_max == v_min:
        return None

    return (series - v_min) / (v_max - v_min) * 100


def build_soc(df, state):
    df = df.copy()

    if "Voltage" not in df.columns:
        raise ValueError("缺少 Voltage 列")

    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")

    soc = None
    for col in ["Capacity_py", "Ah_throughput", "target"]:
        if col in df.columns:
            raw = normalize_to_soc(df[col])
            if raw is not None:
                soc = raw
                break

    if soc is None:
        soc = pd.Series(np.linspace(0, 100, len(df)), index=df.index)

    valid = pd.DataFrame({
        "SOC": soc,
        "Voltage": df["Voltage"]
    }).dropna()

    if len(valid) > 2:
        corr = valid["SOC"].corr(valid["Voltage"])
        if corr < 0:
            soc = 100 - soc

    if state == "dch":
        df["Final_SoC"] = soc
    else:
        df["Final_SoC"] = soc

    return df


def save_soc_voltage_table(df_plot, path, output_dir):
    battery_id, soh, state = extract_metadata(path)

    table = pd.DataFrame({
        "SOH": soh,
        "SOC": df_plot["Final_SoC"].values,
        "Voltage": df_plot["Voltage"].values,
    })

    name = f"{battery_id}_{state}_{soh:.1f}SOH.xlsx"
    table.to_excel(os.path.join(output_dir, name), index=False)


def plot_one_file(path, output_dir):
    battery_id, soh, state = extract_metadata(path)

    df = read_parquet_file(path)
    df = build_soc(df, state)

    df_plot = df.dropna(subset=["Final_SoC", "Voltage"]).copy()
    df_plot = df_plot.sort_values("Final_SoC", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(
        df_plot["Final_SoC"],
        df_plot["Voltage"],
        linewidth=2.2,
        label="qOCV Discharge curve"
    )

    ax.set_title(f"{battery_id} qOCV Discharge Curve\nSOH = {soh:.1f}%")
    ax.set_xlabel("State of Charge, SoC [%]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim(102, -2)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()

    image_name = f"{battery_id}_{state}_{soh:.1f}SOH.png"
    image_path = os.path.join(output_dir, image_name)

    plt.savefig(image_path, dpi=300)
    plt.close(fig)

    save_soc_voltage_table(df_plot, path, output_dir)

    return image_path


def batch_plot_dch_ocv(folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = list_parquet_files(folder)
    dch_files = [f for f in files if is_dch_file(f)]

    if not dch_files:
        raise ValueError("没有找到文件名中包含 dch 的 parquet 文件")

    for i, file_path in enumerate(dch_files, start=1):
        try:
            image_path = plot_one_file(file_path, output_dir)
            print(f"[{i}/{len(dch_files)}] 完成: {image_path}")
        except Exception as e:
            print(f"[{i}/{len(dch_files)}] 失败: {file_path} | {e}")


if __name__ == "__main__":
    batch_plot_dch_ocv(TARGET_FOLDER, OUTPUT_DIR)
