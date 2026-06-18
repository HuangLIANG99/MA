"""
    从环境变量读取 MinIO/S3 密钥。

    Windows PowerShell 示例：
    $env:MINIO_ACCESS_KEY="pBqUl80MTnKOVW0tClGY"
    $env:MINIO_SECRET_KEY="9GeYu4a6jaq6gY0Z60mjO5LL1cmi44v1HHOteVNA"

    Linux / macOS 示例：
    export MINIO_ACCESS_KEY="你的新access_key"
    export MINIO_SECRET_KEY="你的新secret_key"
    """


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
    "projects/j8005-metabatt/Metabatt/VTC/METABatt_Sony_Murata_18650VTC6_003/J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2024-10-03_142512=jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS014653 _ Format01=Kreis M3-034=filesize-109888838=finished.parquet"
)

OUTPUT_DIR = "output"


# ============================================================
# 1. MinIO / S3 配置
# ============================================================

def get_s3_storage_options():


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
    """

    storage_options = get_s3_storage_options()

    if target_file.startswith("s3://"):
        s3_path = target_file
    else:
        s3_path = f"s3://{target_file}"

    print(f"📄 正在读取文件: {s3_path}")

    df = pd.read_parquet(s3_path, storage_options=storage_options)

    print(f"✅ 成功读取数据，行数: {len(df)}")
    print(f"   列名: {df.columns.tolist()}")

    return df


# ============================================================
# 3. 主程序 - 读取并画图
# ============================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔍 开始读取数据...\n")

    try:
        # 读取数据
        df = read_battery_data(TARGET_FILE)

        # 找时间和电流电压列
        # 时间列可能是 Zeit 或 time
        time_col = None
        for col in ["Zeit", "time", "Time"]:
            if col in df.columns:
                time_col = col
                break

        # 电压列可能是 Spannung 或 voltage
        voltage_col = None
        for col in ["Spannung", "voltage", "Voltage"]:
            if col in df.columns:
                voltage_col = col
                break

        # 电流列可能是 Strom 或 current
        current_col = None
        for col in ["Strom", "current", "Current"]:
            if col in df.columns:
                current_col = col
                break

        if time_col is None:
            raise ValueError("找不到时间列 (Zeit/time)")
        if voltage_col is None:
            raise ValueError("找不到电压列 (Spannung/voltage)")
        if current_col is None:
            raise ValueError("找不到电流列 (Strom/current)")

        print(f"\n📊 使用列:")
        print(f"   时间: {time_col}")
        print(f"   电压: {voltage_col}")
        print(f"   电流: {current_col}")

        # 转换时间为数值（从开始计时的秒数）
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            time_seconds = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
        elif pd.api.types.is_timedelta64_any_dtype(df[time_col]):
            time_seconds = df[time_col].dt.total_seconds()
            time_seconds = time_seconds - time_seconds.iloc[0]
        else:
            # 尝试直接转换为数值
            time_seconds = pd.to_numeric(df[time_col], errors='coerce')
            time_seconds = time_seconds - time_seconds.iloc[0]

        # 转换为小时
        time_hours = time_seconds / 3600.0

        # 打印数据统计
        print(f"\n📈 数据统计:")
        print(f"   时间范围: {time_hours.min():.2f} 到 {time_hours.max():.2f} 小时")
        print(f"   电压范围: {df[voltage_col].min():.3f} 到 {df[voltage_col].max():.3f} V")
        print(f"   电流范围: {df[current_col].min():.3f} 到 {df[current_col].max():.3f} A")

        # 画图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 电压图
        ax1.plot(time_hours, df[voltage_col], linewidth=0.8, color='blue')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Voltage vs Time')
        ax1.grid(True)

        # 电流图
        ax2.plot(time_hours, df[current_col], linewidth=0.8, color='red')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Current vs Time')
        ax2.grid(True)

        plt.tight_layout()

        # # 保存图片
        # save_path = os.path.join(OUTPUT_DIR, "voltage_current_plot.png")
        # plt.savefig(save_path, dpi=300)
        # print(f"\n✅ 已保存图片: {save_path}")

        # 显示图片
        plt.show()

        print("\n🎉 完成！")

    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()