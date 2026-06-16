#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_build_ocv_from_qocv_folder.py

Standalone tool:
Read a folder of qOCV parquet/csv files and generate OCV / pseudo-OCV curves.

Recommended usage:
1. Set QOCV_DATA_PATH to your qOCV folder.
2. Set USE_S3 according to whether the folder is on MinIO/S3 or local disk.
3. Run this script.
4. Check:
   - qocv_file_manifest.csv
   - qocv_ocv_curves_manifest.csv
   - selected_reference_ocv.csv
   - selected_reference_ocv.png

Key design:
- Do NOT merge qOCV files from different SOH into one OCV curve.
- Generate one OCV curve per SOH checkpoint first.
- Then select one reference curve for P2D forward / parameter-identification work.

Selection policy:
- For a fixed reference OCV used by P2D parameter identification, choose highest SOH.
- If identifying a segment at a known SOH, choose nearest SOH.
- If charge and discharge qOCV exist at the same SOH, use their midline:
      OCV_mid = 0.5 * (V_charge + V_discharge)
  to reduce hysteresis and low-rate polarization bias.
"""

import os
import re
import glob
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. User configuration
# ============================================================

USE_S3 = False

# Local example:
# QOCV_DATA_PATH = r"D:\your\qocv_folder"
#
# S3 example:
# QOCV_DATA_PATH = "projects/j8005-metabatt/Metabatt/VTC/your_qocv_folder"
QOCV_DATA_PATH = r"./qocv_folder"

OUTPUT_DIR = "qocv_ocv_output"

# Optional filters
FILE_NAME_CONTAINS = None
FILE_NAME_EXCLUDES = ("Init", "pulse")

# If one folder contains many cells, set this, e.g. "18650VTC6_003".
# If None, all qOCV files in the folder are considered.
CELL_ID_CONTAINS = None

# Nominal capacity is only used for rough C-rate estimation.
NOMINAL_CAPACITY_AH = 3.0

# Basic qOCV quality thresholds
MIN_POINTS = 100
MIN_DURATION_S = 300.0
MIN_CAPACITY_AH = 0.5
MAX_ABS_CURRENT_A = 10.0
MAX_QOCV_C_RATE = 0.25
MAX_CURRENT_REL_STD = 0.15
VOLTAGE_MIN_ALLOWED = 2.0
VOLTAGE_MAX_ALLOWED = 4.5

# OCV curve generation
SOC_GRID_POINTS = 1001
SMOOTH_WINDOW = 31
ENFORCE_MONOTONIC = True

# Optional IR correction. Default 0.0 because the exact low-rate resistance
# should not be guessed. If you have a trusted DC resistance, set it here.
IR_CORRECTION_OHM = 0.0

# Selection mode for selected_reference_ocv.csv:
# - "highest_soh": choose the highest SOH curve as reference OCV.
# - "target_soh": choose curve nearest to TARGET_SOH.
# - "all": only generate all curves; selected reference still falls back to highest_soh.
SELECTION_MODE = "highest_soh"
TARGET_SOH = None

# If charge and discharge both exist for the same SOH, use midline if possible.
PREFER_CHARGE_DISCHARGE_MIDLINE = True

# Output plot settings
SAVE_PLOTS = True
DISPLAY_PLOTS = False


# ============================================================
# 1. S3 utilities
# ============================================================

def get_s3_storage_options():
    """
    Read MinIO/S3 keys from environment variables.

    Windows PowerShell:
    $env:MINIO_ACCESS_KEY="your_access_key"
    $env:MINIO_SECRET_KEY="your_secret_key"

    Linux / macOS:
    export MINIO_ACCESS_KEY="your_access_key"
    export MINIO_SECRET_KEY="your_secret_key"
    """

    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")

    if key is None or secret is None:
        raise ValueError(
            "MINIO_ACCESS_KEY or MINIO_SECRET_KEY is missing. "
            "Please set them as environment variables."
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
    if str(path).startswith("s3://"):
        return str(path)
    return "s3://" + str(path).lstrip("/")


# ============================================================
# 2. File discovery and reading
# ============================================================

def list_data_files(
    data_path,
    use_s3=False,
    file_name_contains=None,
    file_name_excludes=None,
    cell_id_contains=None,
):
    if data_path is None:
        raise ValueError("QOCV_DATA_PATH is None.")

    if use_s3:
        import s3fs
        storage_options = get_s3_storage_options()
        fs = s3fs.S3FileSystem(**storage_options)

        s3_uri = to_s3_uri(data_path)
        no_proto = s3_uri.replace("s3://", "", 1)

        if no_proto.lower().endswith((".parquet", ".csv")):
            files = [no_proto]
        else:
            files = []
            files.extend(fs.glob(no_proto.rstrip("/") + "/**/*.parquet"))
            files.extend(fs.glob(no_proto.rstrip("/") + "/**/*.csv"))

        files = ["s3://" + f for f in files]
    else:
        data_path = str(data_path)
        if data_path.lower().endswith((".parquet", ".csv")):
            files = [data_path]
        else:
            files = []
            files.extend(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
            files.extend(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))

    files = sorted(files)

    if file_name_contains is not None:
        token = str(file_name_contains).lower()
        files = [f for f in files if token in os.path.basename(f).lower()]

    if cell_id_contains is not None:
        token = str(cell_id_contains).lower()
        files = [f for f in files if token in os.path.basename(f).lower()]

    if file_name_excludes is not None:
        if isinstance(file_name_excludes, str):
            excludes = [file_name_excludes]
        else:
            excludes = list(file_name_excludes)

        excludes = [str(x).lower() for x in excludes if str(x).strip()]
        files = [
            f for f in files
            if not any(x in os.path.basename(f).lower() for x in excludes)
        ]

    if not files:
        raise FileNotFoundError("No parquet/csv files found. Check QOCV_DATA_PATH and filters.")

    print(f"\nFound {len(files)} qOCV files.")
    for i, f in enumerate(files[:20]):
        print(f"  [{i}] {f}")
    if len(files) > 20:
        print(f"  ... {len(files) - 20} more files")

    return files


def read_one_file(path, use_s3=False):
    suffix = str(path).lower().split("?")[0]

    if use_s3:
        storage_options = get_s3_storage_options()
        if suffix.endswith(".csv"):
            return pd.read_csv(path, storage_options=storage_options)
        return pd.read_parquet(path, storage_options=storage_options)

    if suffix.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


# ============================================================
# 3. Metadata parsing
# ============================================================

def parse_qocv_filename(path):
    """
    Parse metadata from filenames such as:
    METABatt_Sony_Murata_18650VTC6_003_qocv_cha_BM13_90.0SOH.parquet

    Returns:
    - cell_id: roughly "METABatt_Sony_Murata_18650VTC6_003"
    - direction: "charge" / "discharge" / "unknown"
    - bm_index: 13
    - soh: 90.0
    """
    name = os.path.basename(str(path))

    # Direction
    lower = name.lower()
    if re.search(r"(_|-)qocv(_|-)?cha", lower) or "_cha_" in lower or "charge" in lower:
        direction = "charge"
    elif re.search(r"(_|-)qocv(_|-)?(dis|dch|ent)", lower) or "_dis_" in lower or "_dch_" in lower or "discharge" in lower:
        direction = "discharge"
    else:
        direction = "unknown"

    # SOH
    soh = np.nan
    m_soh = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*SOH", name, flags=re.IGNORECASE)
    if m_soh:
        soh = float(m_soh.group(1))

    # BM index
    bm_index = np.nan
    m_bm = re.search(r"BM\s*([0-9]+)", name, flags=re.IGNORECASE)
    if m_bm:
        bm_index = int(m_bm.group(1))

    # Cell ID: substring before _qocv if possible
    m_cell = re.search(r"(.+?)_qocv", name, flags=re.IGNORECASE)
    if m_cell:
        cell_id = m_cell.group(1)
    else:
        cell_id = os.path.splitext(name)[0]

    return {
        "file_name": name,
        "cell_id": cell_id,
        "direction": direction,
        "bm_index": bm_index,
        "soh": soh,
    }


# ============================================================
# 4. Data preprocessing
# ============================================================

def rename_columns(df):
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

    actual = {old: new for old, new in column_map.items() if old in df.columns}
    out = df.rename(columns=actual).copy()

    missing = [c for c in ["time", "voltage", "current"] if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Current columns: {df.columns.tolist()}")

    return out


def convert_time_to_seconds(df):
    df = df.copy()

    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
        return df

    if pd.api.types.is_timedelta64_any_dtype(df["time"]):
        df["time_s"] = df["time"].dt.total_seconds()
        df["time_s"] -= df["time_s"].iloc[0]
        return df

    if np.issubdtype(df["time"].dtype, np.number):
        df["time_s"] = df["time"].astype(float)
        df["time_s"] -= df["time_s"].iloc[0]
        return df

    parsed = pd.to_datetime(df["time"], errors="coerce")
    if parsed.notna().mean() > 0.8:
        df["time_s"] = (parsed - parsed.iloc[0]).dt.total_seconds()
        return df

    td = pd.to_timedelta(df["time"], errors="coerce")
    if td.notna().mean() > 0.8:
        df["time_s"] = td.dt.total_seconds()
        df["time_s"] -= df["time_s"].iloc[0]
        return df

    raise ValueError(f"Cannot parse time column. dtype={df['time'].dtype}")


def preprocess_df(df):
    df = rename_columns(df)

    for c in ["voltage", "current", "temperature", "capacity_ah"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ISEA parquet current is normally already A.
    current_abs = df["current"].abs().dropna()
    if len(current_abs) > 0 and current_abs.median() > 50:
        df["current"] = df["current"] / 1000.0
        print("  Warning: current median > 50, converted from mA to A.")

    df = convert_time_to_seconds(df)
    df = df.dropna(subset=["time_s", "voltage", "current"])
    df = df.sort_values("time_s").drop_duplicates("time_s").reset_index(drop=True)

    return df


# ============================================================
# 5. qOCV curve extraction
# ============================================================

@dataclass
class QOCVCurve:
    file_path: str
    cell_id: str
    direction: str
    bm_index: float
    soh: float
    capacity_ah: float
    duration_h: float
    mean_abs_current_a: float
    estimated_c_rate: float
    current_rel_std: float
    voltage_min_v: float
    voltage_max_v: float
    voltage_start_v: float
    voltage_end_v: float
    is_usable: bool
    reject_reason: str
    soc: np.ndarray
    ocv_v: np.ndarray


def build_curve_from_file(path, use_s3=False):
    meta = parse_qocv_filename(path)
    df = read_one_file(path, use_s3=use_s3)
    df = preprocess_df(df)

    if len(df) < MIN_POINTS:
        raise ValueError("Too few valid points.")

    time_s = df["time_s"].to_numpy(dtype=float)
    voltage = df["voltage"].to_numpy(dtype=float)
    current = df["current"].to_numpy(dtype=float)

    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0:
        raise ValueError("Non-positive duration.")

    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.maximum(dt, 0.0)

    abs_current = np.abs(current)
    capacity_ah = float(np.sum(abs_current * dt / 3600.0))
    mean_abs_i = float(np.mean(abs_current))
    std_abs_i = float(np.std(abs_current))
    rel_std_i = std_abs_i / max(mean_abs_i, 1e-12)
    c_rate = mean_abs_i / NOMINAL_CAPACITY_AH

    direction = meta["direction"]
    if direction == "unknown":
        # Fallback: infer from voltage trend.
        if voltage[-1] > voltage[0]:
            direction = "charge"
        else:
            direction = "discharge"

    reasons = []
    if len(df) < MIN_POINTS:
        reasons.append("too_few_points")
    if duration_s < MIN_DURATION_S:
        reasons.append("too_short")
    if capacity_ah < MIN_CAPACITY_AH:
        reasons.append("capacity_too_small")
    if mean_abs_i <= 0:
        reasons.append("zero_current")
    if mean_abs_i > MAX_ABS_CURRENT_A:
        reasons.append("current_too_large")
    if c_rate > MAX_QOCV_C_RATE:
        reasons.append("c_rate_too_high")
    if rel_std_i > MAX_CURRENT_REL_STD:
        reasons.append("current_not_stable")
    if np.nanmin(voltage) < VOLTAGE_MIN_ALLOWED or np.nanmax(voltage) > VOLTAGE_MAX_ALLOWED:
        reasons.append("voltage_out_of_range")
    if not np.isfinite(meta["soh"]):
        reasons.append("soh_not_found_in_filename")

    is_usable = len(reasons) == 0

    # Build SOC from throughput.
    q = np.cumsum(abs_current * dt / 3600.0)
    q_end = float(q[-1])
    if q_end <= 0:
        raise ValueError("Zero integrated qOCV capacity.")

    if direction == "charge":
        soc = q / q_end
        # charge terminal voltage is above OCV; subtract simple IR if supplied.
        ocv_like = voltage - abs_current * IR_CORRECTION_OHM
    elif direction == "discharge":
        soc = 1.0 - q / q_end
        # discharge terminal voltage is below OCV; add simple IR if supplied.
        ocv_like = voltage + abs_current * IR_CORRECTION_OHM
    else:
        # Should not occur after fallback, but keep safe.
        soc = q / q_end
        ocv_like = voltage

    curve_df = pd.DataFrame({"soc": soc, "ocv_v": ocv_like})
    curve_df = curve_df.replace([np.inf, -np.inf], np.nan).dropna()
    curve_df = curve_df[
        (curve_df["soc"] >= 0.0)
        & (curve_df["soc"] <= 1.0)
        & (curve_df["ocv_v"] >= VOLTAGE_MIN_ALLOWED)
        & (curve_df["ocv_v"] <= VOLTAGE_MAX_ALLOWED)
    ]
    curve_df = curve_df.sort_values("soc").drop_duplicates("soc", keep="first")

    if len(curve_df) < 10:
        is_usable = False
        reasons.append("too_few_curve_points_after_cleaning")

    return QOCVCurve(
        file_path=str(path),
        cell_id=meta["cell_id"],
        direction=direction,
        bm_index=meta["bm_index"],
        soh=float(meta["soh"]) if np.isfinite(meta["soh"]) else np.nan,
        capacity_ah=capacity_ah,
        duration_h=duration_s / 3600.0,
        mean_abs_current_a=mean_abs_i,
        estimated_c_rate=c_rate,
        current_rel_std=rel_std_i,
        voltage_min_v=float(np.nanmin(voltage)),
        voltage_max_v=float(np.nanmax(voltage)),
        voltage_start_v=float(voltage[0]),
        voltage_end_v=float(voltage[-1]),
        is_usable=bool(is_usable),
        reject_reason="OK" if is_usable else ";".join(reasons),
        soc=curve_df["soc"].to_numpy(dtype=float),
        ocv_v=curve_df["ocv_v"].to_numpy(dtype=float),
    )


def smooth_and_monotonic(ocv):
    out = np.asarray(ocv, dtype=float)

    if SMOOTH_WINDOW and SMOOTH_WINDOW > 1 and len(out) >= 5:
        window = int(SMOOTH_WINDOW)
        if window % 2 == 0:
            window += 1
        window = min(window, len(out) if len(out) % 2 == 1 else len(out) - 1)
        if window >= 3:
            out = (
                pd.Series(out)
                .rolling(window=window, center=True, min_periods=1)
                .median()
                .to_numpy(dtype=float)
            )

    if ENFORCE_MONOTONIC:
        # Full-cell OCV should generally increase with SOC.
        out = np.maximum.accumulate(out)

    return out


def interpolate_curve(curve, soc_grid):
    return np.interp(soc_grid, curve.soc, curve.ocv_v)


def group_curves_by_cell_and_soh(curves):
    groups = {}
    for c in curves:
        if not c.is_usable:
            continue

        # Use rounded SOH as grouping key to avoid tiny float-string mismatch.
        soh_key = round(float(c.soh), 3)
        key = (c.cell_id, soh_key)
        groups.setdefault(key, []).append(c)

    return groups


def build_ocv_per_soh(curves, output_dir):
    """
    Build one OCV table per (cell_id, SOH).
    If both charge and discharge exist, use midline.
    Otherwise use the available direction.
    """
    os.makedirs(output_dir, exist_ok=True)
    soc_grid = np.linspace(0.0, 1.0, SOC_GRID_POINTS)

    groups = group_curves_by_cell_and_soh(curves)
    records = []
    ocv_tables = {}

    for (cell_id, soh), items in sorted(groups.items(), key=lambda x: (x[0][0], -x[0][1])):
        charges = [x for x in items if x.direction == "charge"]
        discharges = [x for x in items if x.direction == "discharge"]

        # If multiple same-direction files exist at same SOH, choose the largest capacity and lowest C-rate.
        def choose_best(xs):
            return sorted(xs, key=lambda x: (-x.capacity_ah, x.estimated_c_rate, x.current_rel_std))[0]

        charge = choose_best(charges) if charges else None
        discharge = choose_best(discharges) if discharges else None

        if PREFER_CHARGE_DISCHARGE_MIDLINE and charge is not None and discharge is not None:
            v_charge = interpolate_curve(charge, soc_grid)
            v_discharge = interpolate_curve(discharge, soc_grid)
            ocv = 0.5 * (v_charge + v_discharge)
            source_type = "charge_discharge_midline"
            source_files = f"{charge.file_path} | {discharge.file_path}"
            source_capacity_ah = min(charge.capacity_ah, discharge.capacity_ah)
            source_c_rate = max(charge.estimated_c_rate, discharge.estimated_c_rate)
        elif discharge is not None:
            ocv = interpolate_curve(discharge, soc_grid)
            source_type = "discharge_only"
            source_files = discharge.file_path
            source_capacity_ah = discharge.capacity_ah
            source_c_rate = discharge.estimated_c_rate
        elif charge is not None:
            ocv = interpolate_curve(charge, soc_grid)
            source_type = "charge_only"
            source_files = charge.file_path
            source_capacity_ah = charge.capacity_ah
            source_c_rate = charge.estimated_c_rate
        else:
            continue

        ocv = smooth_and_monotonic(ocv)

        safe_cell = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cell_id))
        out_name = f"ocv_{safe_cell}_{soh:.3f}SOH.csv"
        out_path = os.path.join(output_dir, out_name)

        table = pd.DataFrame({
            "soc": soc_grid,
            "ocv_v": ocv,
            "cell_id": cell_id,
            "soh": soh,
            "source_type": source_type,
            "source_files": source_files,
            "ir_correction_ohm": IR_CORRECTION_OHM,
        })
        table.to_csv(out_path, index=False)

        ocv_tables[(cell_id, soh)] = table

        records.append({
            "cell_id": cell_id,
            "soh": soh,
            "source_type": source_type,
            "source_files": source_files,
            "capacity_ah_for_selection": source_capacity_ah,
            "estimated_c_rate_for_selection": source_c_rate,
            "ocv_csv": out_path,
        })

        if SAVE_PLOTS:
            plot_path = out_path.replace(".csv", ".png")
            fig, ax = plt.subplots(figsize=(8, 5))
            if charge is not None:
                ax.scatter(charge.soc, charge.ocv_v, s=3, alpha=0.2, label="charge raw")
            if discharge is not None:
                ax.scatter(discharge.soc, discharge.ocv_v, s=3, alpha=0.2, label="discharge raw")
            ax.plot(table["soc"], table["ocv_v"], linewidth=2.0, label=source_type)
            ax.set_xlabel("SOC")
            ax.set_ylabel("OCV / V")
            ax.set_title(f"{cell_id}, SOH={soh:.3f}")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            if DISPLAY_PLOTS:
                plt.show()
            plt.close(fig)

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(output_dir, "qocv_ocv_curves_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    return manifest, ocv_tables


def select_reference_ocv(ocv_manifest):
    if ocv_manifest.empty:
        raise RuntimeError("No OCV curves generated.")

    m = ocv_manifest.copy()

    if SELECTION_MODE == "target_soh" and TARGET_SOH is not None:
        m["soh_distance"] = np.abs(m["soh"].astype(float) - float(TARGET_SOH))
        selected = (
            m.sort_values(
                ["soh_distance", "source_type", "capacity_ah_for_selection"],
                ascending=[True, True, False],
            )
            .iloc[0]
        )
        selection_note = f"nearest_to_target_soh_{TARGET_SOH}"
    else:
        selected = (
            m.sort_values(
                ["soh", "source_type", "capacity_ah_for_selection"],
                ascending=[False, True, False],
            )
            .iloc[0]
        )
        selection_note = "highest_soh_reference"

    selected_table = pd.read_csv(selected["ocv_csv"])
    selected_table["selection_note"] = selection_note

    out_csv = os.path.join(OUTPUT_DIR, "selected_reference_ocv.csv")
    selected_table.to_csv(out_csv, index=False)

    if SAVE_PLOTS:
        out_png = os.path.join(OUTPUT_DIR, "selected_reference_ocv.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(selected_table["soc"], selected_table["ocv_v"], linewidth=2.0)
        ax.set_xlabel("SOC")
        ax.set_ylabel("OCV / V")
        ax.set_title(
            f"Selected reference OCV: {selected['cell_id']}, "
            f"SOH={float(selected['soh']):.3f}, {selection_note}"
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        if DISPLAY_PLOTS:
            plt.show()
        plt.close(fig)

    print("\nSelected reference OCV:")
    print(f"  cell_id: {selected['cell_id']}")
    print(f"  SOH: {float(selected['soh']):.3f}")
    print(f"  source_type: {selected['source_type']}")
    print(f"  csv: {out_csv}")

    return out_csv


# ============================================================
# 6. Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = list_data_files(
        QOCV_DATA_PATH,
        use_s3=USE_S3,
        file_name_contains=FILE_NAME_CONTAINS,
        file_name_excludes=FILE_NAME_EXCLUDES,
        cell_id_contains=CELL_ID_CONTAINS,
    )

    curves = []
    manifest_rows = []

    for i, path in enumerate(files):
        print(f"\nReading [{i + 1}/{len(files)}]: {path}")
        try:
            curve = build_curve_from_file(path, use_s3=USE_S3)
            curves.append(curve)

            manifest_rows.append({
                "file_path": curve.file_path,
                "cell_id": curve.cell_id,
                "direction": curve.direction,
                "bm_index": curve.bm_index,
                "soh": curve.soh,
                "capacity_ah": curve.capacity_ah,
                "duration_h": curve.duration_h,
                "mean_abs_current_a": curve.mean_abs_current_a,
                "estimated_c_rate": curve.estimated_c_rate,
                "current_rel_std": curve.current_rel_std,
                "voltage_start_v": curve.voltage_start_v,
                "voltage_end_v": curve.voltage_end_v,
                "voltage_min_v": curve.voltage_min_v,
                "voltage_max_v": curve.voltage_max_v,
                "is_usable": curve.is_usable,
                "reject_reason": curve.reject_reason,
            })

            print(
                f"  direction={curve.direction}, SOH={curve.soh}, "
                f"capacity={curve.capacity_ah:.4f} Ah, "
                f"C-rate={curve.estimated_c_rate:.4f} C, "
                f"usable={curve.is_usable}, reason={curve.reject_reason}"
            )

        except Exception as e:
            print(f"  Failed: {e}")
            meta = parse_qocv_filename(path)
            manifest_rows.append({
                "file_path": str(path),
                "cell_id": meta["cell_id"],
                "direction": meta["direction"],
                "bm_index": meta["bm_index"],
                "soh": meta["soh"],
                "is_usable": False,
                "reject_reason": str(e),
            })

    file_manifest = pd.DataFrame(manifest_rows)
    file_manifest_path = os.path.join(OUTPUT_DIR, "qocv_file_manifest.csv")
    file_manifest.to_csv(file_manifest_path, index=False)

    print(f"\nSaved qOCV file manifest: {file_manifest_path}")

    usable_count = int(file_manifest["is_usable"].fillna(False).astype(bool).sum())
    print(f"Usable qOCV files: {usable_count}/{len(file_manifest)}")

    if usable_count == 0:
        raise RuntimeError(
            "No usable qOCV files. Check reject_reason in qocv_file_manifest.csv."
        )

    ocv_manifest, ocv_tables = build_ocv_per_soh(curves, OUTPUT_DIR)

    ocv_manifest_path = os.path.join(OUTPUT_DIR, "qocv_ocv_curves_manifest.csv")
    print(f"Saved OCV curves manifest: {ocv_manifest_path}")

    selected_csv = select_reference_ocv(ocv_manifest)

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Selected reference OCV: {selected_csv}")


if __name__ == "__main__":
    main()
