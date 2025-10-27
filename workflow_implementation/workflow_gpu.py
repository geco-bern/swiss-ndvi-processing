#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU NDVI Processor
==================
- Loads NDVI, params_lower, params_upper from Zarr.
- Reconstructs missing/invalid NDVI values using double logistic model.
- Detects outliers and smooths NDVI curves (L1/L2 hybrid).
- Runs entirely on GPU (PyTorch).
- Processes 2,000 pixels per batch for stability.
- Logs progress to both console (tqdm) and log file.

Author: Your Name
Date: 2025-10-20
"""

import torch
import numpy as np
import zarr
import pandas as pd
import re
import shutil
import time
from datetime import datetime, date
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ZARR = "/data_2/scratch/francesco/zarr_demo_daily/"
OUTPUT_ZARR = "/data_2/scratch/francesco/zarr_demo_daily_output"
LOG_FILE = "gpu_ndvi_processing.log"
CHUNK_SIZE = 2000  # pixels per batch
MAX_DATES = 3000   # set to None for all

# ============================================================
# LOGGING SETUP
# ============================================================

def log(msg):
    """Write log message to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============================================================
# GPU SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"✅ Using device: {device}")

torch.set_float32_matmul_precision("high")

# ============================================================
# ZARR SETUP
# ============================================================

log("🔄 Initializing Zarr dataset...")

shutil.copytree(INPUT_ZARR, OUTPUT_ZARR, dirs_exist_ok=True)

ds_in = zarr.open_group(INPUT_ZARR, mode="r")
ds_out = zarr.open_group(OUTPUT_ZARR, mode="r+")

ndvi_zarr = ds_out["ndvi"]
dates_zarr = ds_out["dates"]
params_lower_zarr = ds_out["params"]["params_lower"]
params_upper_zarr = ds_out["params"]["params_upper"]

# ============================================================
# DATE PARSING
# ============================================================

def _parse_zarr_date(d_val):
    if isinstance(d_val, (bytes, np.bytes_)):
        return pd.to_datetime(d_val.decode("utf-8"), errors="coerce")
    else:
        s = str(d_val)
        if s.startswith("np.bytes_("):
            m = re.search(r"b['\"]([^'\"]+)['\"]", s)
            if m:
                s = m.group(1)
        return pd.to_datetime(s, errors="coerce")

dates = []
for i in range(dates_zarr.shape[0]):
    d_arr = dates_zarr.get_basic_selection((i,))
    d_val = d_arr[()] if isinstance(d_arr, np.ndarray) and d_arr.shape == () else d_arr[0]
    d_dt = _parse_zarr_date(d_val)
    if pd.notna(d_dt):
        dates.append(d_dt)

dates = sorted(list(set(dates)))
if not dates:
    raise ValueError("❌ No valid dates found in Zarr dataset")

log(f"📅 Found {len(dates)} valid dates")

# ============================================================
# GPU LOGISTIC + STAT FUNCTIONS
# ============================================================

def double_logistic_function(t, params):
    """Vectorized double logistic function on GPU."""
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def calculate_median_gpu(doy_tensor, params_lower, params_upper):
    """Computes median NDVI model from lower & upper params."""
    t = doy_tensor / 365.0
    lower = double_logistic_function(t, params_lower)
    upper = double_logistic_function(t, params_upper)
    return 0.5 * (upper + lower)

# ============================================================
# OUTLIER DETECTION + SMOOTHING (GPU)
# ============================================================

def detect_and_smooth_gpu(ndvi_vals, medians, l1_weight=0.7, l2_weight=0.3):
    """
    Simple hybrid L1/L2 smoothing.
    Outliers are replaced by weighted mean between NDVI and model median.
    """
    diff = ndvi_vals - medians
    abs_diff = diff.abs()
    # Outlier mask: > 2 * MAD or > 2 * std (approx)
    threshold = 2 * torch.median(abs_diff) + 1e-6
    outlier_mask = abs_diff > threshold
    smoothed = (
        l1_weight * torch.median(ndvi_vals) +
        l2_weight * torch.mean(ndvi_vals)
    )
    # Replace outliers with smoothed + keep valid
    ndvi_vals = torch.where(outlier_mask, smoothed, ndvi_vals)
    return ndvi_vals

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_day_gpu(day, chunk_size=2000):
    """Process one day's NDVI slice on GPU with chunked batches."""
    start_time = time.time()
    base_date = pd.to_datetime(dates[0]).date()
    date_index = (day - base_date).days
    n_pixels = ndvi_zarr.shape[1]
    doy = day.timetuple().tm_yday

    log(f"🗓️ Processing {day} (index={date_index})")

    for start in tqdm(range(0, n_pixels, chunk_size), desc=f"Pixels for {day}", leave=False):
        end = min(start + chunk_size, n_pixels)

        # --- Load batch ---
        ndvi_vals = torch.as_tensor(
            ndvi_zarr[date_index, start:end], dtype=torch.float32, device=device
        ) / 10000.0
        params_lower = torch.as_tensor(
            params_lower_zarr[start:end, :], dtype=torch.float32, device=device
        )
        params_upper = torch.as_tensor(
            params_upper_zarr[start:end, :], dtype=torch.float32, device=device
        )

        doy_tensor = torch.full((end - start, 1), doy, dtype=torch.float32, device=device)

        # --- Compute model median ---
        medians = calculate_median_gpu(doy_tensor, params_lower, params_upper).squeeze()

        # --- Mask invalid NDVI and replace with model median ---
        valid_mask = (ndvi_vals > 0.0) & (ndvi_vals < 1.0)
        ndvi_vals = torch.where(valid_mask, ndvi_vals, medians)

        # --- Detect & smooth outliers ---
        ndvi_vals = detect_and_smooth_gpu(ndvi_vals, medians)

        # --- Clip + scale + convert ---
        ndvi_scaled = (ndvi_vals.clamp(0, 1) * 10000).to(torch.int16)

        # --- Write back to Zarr ---
        ndvi_zarr[date_index, start:end] = ndvi_scaled.cpu().numpy()

        torch.cuda.empty_cache()

    duration = time.time() - start_time
    log(f"✅ Finished {day} in {duration:.2f}s")

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    total_start = time.time()
    n_days = len(dates) if MAX_DATES is None else min(MAX_DATES, len(dates))
    log(f"🚀 Starting GPU NDVI processing for {n_days} days...")

    for d in tqdm(dates[:n_days], desc="Processing days"):
        process_day_gpu(pd.to_datetime(d).date(), chunk_size=CHUNK_SIZE)

    total_dur = (time.time() - total_start) / 60
    log(f"🎉 Completed all NDVI slices in {total_dur:.2f} minutes")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    log("=========================================")
    log("🔥 GPU NDVI Processor started")
    log("=========================================")
    main()
    log("🏁 Done.")
