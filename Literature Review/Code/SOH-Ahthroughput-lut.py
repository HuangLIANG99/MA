import os
import re
import pandas as pd
import numpy as np
import s3fs
from scipy.interpolate import PchipInterpolator

# ============================================================
# 1. 用户配置区 (请根据实际情况修改)
# ============================================================
# S3 文件夹路径 (请指向包含你那40多个 qocv 文件的文件夹目录)
S3_FOLDER_PREFIX = "projects/j8005-metabatt/Metabatt/VTC/30_export_qocv/METABatt_Sony_Murata_18650VTC6_003/"

# Parquet 文件中“累计吞吐安时”的实际列名
AH_COL_NAME = "Ah_throughput"  # 👈 请务必替换为文件里真正的列名

# 查找表配置
LUT_STEP_AH = 50           # 查找表的 Ah 步长（例如每 50 Ah 生成一行数据）
OUTPUT_DIR = "output"
OUTPUT_FILENAME = "SOH_Ah_Lookup_Table.csv"

# Excel 区域设置优化
CSV_SEPARATOR = ';'        # 分号作为列分隔
CSV_DECIMAL = ','          # 用逗号作为小数点，防止中文/欧洲版 Excel 自动放大数值

# ============================================================
# 2. S3 存储凭证与文件系统初始化
# ============================================================
def get_s3_fs_and_options():
    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")
    if key is None or secret is None:
        raise ValueError("❌ 未找到环境变量 MINIO_ACCESS_KEY 或 MINIO_SECRET_KEY，请先在终端设置。")
    
    storage_options = {
        "key": key, "secret": secret,
        "client_kwargs": {"endpoint_url": "https://iseadocker.isea.rwth-aachen.de:9000", "region_name": "us-east-1"},
        "config_kwargs": {"s3": {"addressing_style": "path"}, "signature_version": "s3v4"}
    }
    
    # 初始化 s3fs 用于文件列表扫描
    fs = s3fs.S3FileSystem(
        key=key, secret=secret,
        client_kwargs=storage_options["client_kwargs"],
        config_kwargs=storage_options["config_kwargs"]
    )
    return fs, storage_options

# ============================================================
# 3. 核心业务逻辑
# ============================================================
def build_soh_lut_from_s3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取 S3 客户端和配置
    fs, storage_options = get_s3_fs_and_options()
    
    print(f"🔍 正在扫描 S3 目录: {S3_FOLDER_PREFIX} ...")
    # 列出目录下所有的 parquet 文件
    # s3fs.glob 会返回不带 s3:// 前缀的相对路径列表
    search_path = S3_FOLDER_PREFIX.rstrip("/") + "/*.parquet"
    all_s3_paths = fs.glob(search_path)
    
    if not all_s3_paths:
        print(f"❌ 未能在指定 S3 路径下找到任何 .parquet 文件，请检查路径。")
        return

    # 正则表达式：用于从文件名中提取 'cha'/'dis' 和 SOH 浮点数
    filename_pattern = re.compile(r"_(cha|dis)_.*_(\d+\.\d+)SOH.parquet$")
    
    file_records = []
    for path in all_s3_paths:
        filename = os.path.basename(path)
        match = filename_pattern.search(filename)
        if match:
            file_records.append({
                "s3_path": f"s3://{path}",
                "filename": filename,
                "type": match.group(1),        # 'cha' 或 'dis'
                "soh": float(match.group(2))   # SOH 浮点数值
            })
            
    df_files = pd.DataFrame(file_records)
    if df_files.empty:
        print("❌ 扫描到了文件，但没有文件名符合 '*_cha_*_XXSOH.parquet' 的点检命名规则。")
        print(f"示例文件名参考: METABatt_Sony_Murata_18650VTC6_003_qocv_cha_BM5_96.1SOH.parquet")
        return
        
    print(f"✅ 成功匹配到 {len(df_files)} 个点检(QOCV)文件。开始并行/逐个提取全局 Ah 历史数据...")

    # 逐个从 S3 读取 Parquet 的指定列
    extracted_data = []
    for idx, row in df_files.iterrows():
        try:
            # 仅下载和读取所需的 Ah 列，极大地节省网络带宽和内存
            df_p = pd.read_parquet(row["s3_path"], storage_options=storage_options, columns=[AH_COL_NAME])
            max_ah = df_p[AH_COL_NAME].max()
            
            extracted_data.append({
                "soh": row["soh"],
                "type": row["type"],
                "ah": max_ah
            })
            print(f"  [{idx+1}/{len(df_files)}] 已解析: {row['filename']} -> Max Ah: {max_ah:.2f}")
        except KeyError:
            print(f"⚠️ 警告：文件 {row['filename']} 中不存在列名 '{AH_COL_NAME}'，请检查！")
            return
        except Exception as e:
            print(f"❌ 读取 S3 文件失败: {row['filename']}, 错误原因: {e}")
            return

    df_ah = pd.DataFrame(extracted_data)
    
    # 4. 数据融合与对齐 (将相同 SOH 下的充放电取平均值)
    df_grouped = df_ah.groupby("soh")["ah"].mean().reset_index()
    # 按全局累计 Ah 从小到大排序 (对应 SOH 从 96.1% 降到 76%)
    df_grouped = df_grouped.sort_values(by="ah").reset_index(drop=True)
    
    print("\n📊 提取出的原始 [SOH  vs 全局累计Ah] 散点映射关系：")
    print(df_grouped)

    # 5. 实施 PCHIP 保形单调插值
    X_ah = df_grouped["ah"].values
    Y_soh = df_grouped["soh"].values
    
    pchip_model = PchipInterpolator(X_ah, Y_soh)
    
    # 生成标准的等间距查找表索引 (X轴)
    ah_grid = np.arange(np.floor(X_ah.min()), np.ceil(X_ah.max()), LUT_STEP_AH)
    # 预测对应的 SOH
    soh_lut_values = pchip_model(ah_grid)
    
    # 6. 构建最终 DataFrame 并按照指定的 Excel 格式导出
    lut_df = pd.DataFrame({
        "Cumulative_Ah_Index": ah_grid,
        "SOH_Output": np.round(soh_lut_values, 4)
    })
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    # 🟢 注入你指定的 separator 和 decimal 配置
    lut_df.to_csv(output_path, index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL)
    
    print(f"\n🎉 查找表（LUT）成功保存至本地: {output_path}")
    print("📋 查找表前 5 行预览（可直接用 Excel 双击正常打开）:")
    print(lut_df.head(5))

if __name__ == "__main__":
    build_soh_lut_from_s3()
