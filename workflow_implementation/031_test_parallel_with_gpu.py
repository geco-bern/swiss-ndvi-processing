#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated continuous NDVI processing with CPU multiprocessing per-file.
- keeps per-day sequential processing (continuous ingestion simulation)
- uses GPU for double-logistic median computation and L1 interpolation numeric math
- keeps L2 smoothing (LOWESS) on CPU (statsmodels) for now
- reads/writes to Zarr (no global in-RAM storage)
Author: adapted for user request
"""

import os
import numpy as np
import pandas as pd
import zarr
import torch
import shutil
import gc
import concurrent.futures
import hashlib
from datetime import datetime, date
import math
import traceback
import statsmodels.api as sm
from typing import Sequence

# =====================================================
# CONFIGURATION - tune these
# =====================================================
INPUT_DIR = "/data_2/scratch/francesco/zarr_demo_daily/"
OUTPUT_DIR = "/data_2/scratch/francesco/zarr_demo_daily_processed_gpu/"
# number of zarr files to process (set to 1000 if you want to process all)
N_FILES = 2
# number of pixels to process per file (if equal to the total pixels in file, will effectively process all)
N_PIXELS_PER_FILE = 10
# max days to simulate ingestion for each file (None for all)
MAX_DAYS = 1500  # e.g., 1500 or None
# create output dir if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# DEVICE SETUP
# =====================================================
# The script assumes a GPU available. If not available, it will fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MAIN] Using device: {device}")

# NOTE: If you run many worker processes (ProcessPoolExecutor with many workers),
# they will all try to use the same GPU (unless you assign different CUDA_VISIBLE_DEVICES per process).
# That can degrade performance; consider using fewer concurrent processes when GPU is used heavily.

# =====================================================
# UTILITIES (unchanged semantics)
# =====================================================
def unwrap_scalar(x):
    while isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    return x

def zarr_date_to_date(zarr_date):
    zarr_date = unwrap_scalar(zarr_date)
    if isinstance(zarr_date, bytes):
        return datetime.strptime(zarr_date.decode("utf-8"), "%Y-%m-%d").date()
    elif isinstance(zarr_date, np.datetime64):
        return zarr_date.astype("M8[D]").astype(datetime).date()
    elif isinstance(zarr_date, datetime):
        return zarr_date.date()
    elif isinstance(zarr_date, date):
        return zarr_date
    else:
        raise TypeError(f"Unknown date type: {type(zarr_date)}")

# =====================================================
# TORCH-BASED MODEL: Double logistic + batched median
# =====================================================
def double_logistic_function_torch(t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    t: shape (D,) or (D,1) - fractional time in [0,1]
    params: shape (N,6) - parameters per series (sos, mat_minus_sos, sen, eos_minus_sen, M, m)
    Returns: tensor shaped (D, N) - function value for each t and each parameter row
    """
    # ensure shapes
    # t -> (D,1), params -> (N,6)
    if t.ndim == 1:
        t = t[:, None]  # (D,1)
    # params split into columns (N,1)
    sos = params[:, 0:1]            # (N,1)
    mat_minus_sos = params[:, 1:2] # (N,1)
    sen = params[:, 2:3]           # (N,1)
    eos_minus_sen = params[:, 3:4] # (N,1)
    M = params[:, 4:5]             # (N,1)
    m = params[:, 5:6]             # (N,1)

    # ensure float32 on right device
    sos = sos.to(dtype=torch.float32, device=device)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos.to(dtype=torch.float32, device=device))
    sen = sen.to(dtype=torch.float32, device=device)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen.to(dtype=torch.float32, device=device))
    M = M.to(dtype=torch.float32, device=device)
    m = m.to(dtype=torch.float32, device=device)
    t = t.to(dtype=torch.float32, device=device)

    # broadcast shapes: t (D,1), param pieces (1,N) via transpose
    # compute sigmoid terms with broadcasting
    # compute as: sigmoid_sos_mat: shape (D,N)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos.T + mat_minus_sos.T - 2 * t) / (mat_minus_sos.T + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen.T + eos_minus_sen.T - 2 * t) / (eos_minus_sen.T + 1e-10))

    result = (M.T - m.T) * (sigmoid_sos_mat - sigmoid_sen_eos) + m.T  # (D,N)
    return result  # Tensor on device

def calculate_median(doy: Sequence[float], params_lower: np.ndarray, params_upper: np.ndarray) -> np.ndarray:
    """
    Calculate median for given DOYs and parameter arrays using GPU Torch.

    doy: sequence-like of day-of-year integers (1..365) OR a single integer
    params_lower: (N,6) numpy array
    params_upper: (N,6) numpy array
    Returns: medians shape (len(doy), N) if len(doy)>1 else (N,) when single doy
    """
    # Normalize DOY to fractional year
    doys = np.atleast_1d(doy).astype(np.float32)
    t = torch.from_numpy(doys / 365.0).to(device=device, dtype=torch.float32)  # (D,)

    # Turn params into torch tensors on device
    params_lower_t = torch.from_numpy(params_lower.astype(np.float32)).to(device=device)
    params_upper_t = torch.from_numpy(params_upper.astype(np.float32)).to(device)

    # compute
    with torch.no_grad():
        lower = double_logistic_function_torch(t, params_lower_t)  # (D,N)
        upper = double_logistic_function_torch(t, params_upper_t)  # (D,N)
        median = 0.5 * (lower + upper)  # (D,N)

    median_cpu = median.cpu().numpy()
    if median_cpu.shape[0] == 1:
        return median_cpu.squeeze(0)  # (N,)
    return median_cpu  # (D,N)

# =====================================================
# L1 interpolation using GPU for medians & numeric work
# =====================================================
def L1_interpolation_gpu(delta_1: float, delta_2: float, date_1: date, date_2: date,
                         base_date: date, params_lower, params_upper, pixel_idx, ndvi_zarr):
    """
    Performs the L1 interpolation for a single pixel using GPU for medians.
    - params_lower/upper can be arrays of shape (6,) or (1,6)
    - writes into ndvi_zarr for the pixel_idx positions between date_1 and date_2 inclusive
    """
    # Normalize order
    if date_1 <= date_2:
        start_date, end_date = date_1, date_2
        start_delta, end_delta = delta_1, delta_2
    else:
        start_date, end_date = date_2, date_1
        start_delta, end_delta = delta_2, delta_1

    days = (end_date - start_date).days
    if days < 0:
        return
    if days == 0:
        doy = start_date.timetuple().tm_yday
        median = calculate_median(np.array([doy], dtype=np.float32), np.atleast_2d(params_lower), np.atleast_2d(params_upper))
        ndvi_val = start_delta + float(median.squeeze())
        ndvi_scaled = np.clip(ndvi_val * 10000.0, 0, 10000).astype(np.int16)
        idx = (start_date - base_date).days
        # write single value
        ndvi_zarr[idx, pixel_idx] = ndvi_scaled
        return

    # Build DOY sequence for interpolation
    doy_start = start_date.timetuple().tm_yday
    doy_end = end_date.timetuple().tm_yday
    days_diff = days

    if doy_end < doy_start:
        doys = np.linspace(doy_start, doy_end + 365, num=days_diff + 1) % 365
        doys = np.where(doys == 0, 365, doys)
    else:
        doys = np.linspace(doy_start, doy_end, num=days_diff + 1)
    doys = np.where((doys == 0) | (doys == 366), 365, doys).astype(np.float32)

    params_lower_arr = np.atleast_2d(params_lower).astype(np.float32)  # (1,6)
    params_upper_arr = np.atleast_2d(params_upper).astype(np.float32)

    # compute medians on GPU for the whole range of doys (returns shape (D,1))
    medians = calculate_median(doys, params_lower_arr, params_upper_arr).squeeze()  # (D,)

    # linear deltas
    deltas = np.linspace(start_delta, end_delta, num=days_diff + 1, dtype=np.float32)

    ndvi = deltas + medians  # (D,)
    ndvi_scaled = np.clip(ndvi * 10000.0, 0, 10000).astype(np.int16)

    idx_start = (start_date - base_date).days
    idx_end = (end_date - base_date).days
    ndvi_zarr[idx_start:idx_end + 1, pixel_idx] = ndvi_scaled

# =====================================================
# L2 smoothing: keep CPU version (statsmodels.lowess)
# =====================================================
def L2_smoothing_cpu(pixel_idx, init_position, params_lower, params_upper,
                     last_dates_zarr_local, ndvi_zarr_local, dates_list_local):
    """
    This is the L2 smoothing function using statsmodels.lowess (CPU).
    It closely follows your original logic but accepts local copies / references.
    """
    # load last dates (first 7 positions)
    last_dates_bytes = last_dates_zarr_local[:7, pixel_idx]
    last_dates = [zarr_date_to_date(d) for d in last_dates_bytes if not np.all(d == b'1900-01-01')]
    if len(last_dates) < 2:
        return

    ndvi_vals = []
    median_vals = []
    base_date = dates_list_local[0]
    for d in last_dates:
        idx = (d - base_date).days
        if 0 <= idx < ndvi_zarr_local.shape[0]:
            ndvi_val = ndvi_zarr_local[idx, pixel_idx] / 10000.0
        else:
            ndvi_val = np.nan
        median_val = float(calculate_median(np.array([d.timetuple().tm_yday], dtype=np.float32),
                                            np.atleast_2d(params_lower), np.atleast_2d(params_upper)))
        ndvi_vals.append(ndvi_val)
        median_vals.append(median_val)

    ndvi_vals = np.array(ndvi_vals, dtype=np.float32)
    median_vals = np.array(median_vals, dtype=np.float32)
    deltas_arr = ndvi_vals - median_vals

    start_date = last_dates[init_position]
    end_date = last_dates[-1]
    days_diff = (end_date - start_date).days
    if days_diff <= 0:
        return

    doy_start = start_date.timetuple().tm_yday
    doy_end = end_date.timetuple().tm_yday

    if doy_end < doy_start:
        doys = np.linspace(doy_start, doy_end + 365, num=days_diff + 1) % 365
        doys = np.where(doys == 0, 365, doys)
    else:
        doys = np.linspace(doy_start, doy_end, num=days_diff + 1)
    doys = np.where((doys == 0) | (doys == 366), 365, doys)

    idx = np.arange(len(ndvi_vals))

    if np.sum(deltas_arr < -0.2) >= 5:
        loess = sm.nonparametric.lowess(ndvi_vals, idx, frac=1, it=5, return_sorted=True)
        smoothed = np.interp(np.linspace(0, len(ndvi_vals)-1, days_diff+1), loess[:, 0], loess[:, 1])
    else:
        loess = sm.nonparametric.lowess(deltas_arr, idx, frac=1, it=5, return_sorted=True)
        smoothed_deltas = np.interp(np.linspace(0, len(deltas_arr)-1, days_diff+1), loess[:, 0], loess[:, 1])
        medians_seq = calculate_median(doys.astype(np.float32), np.atleast_2d(params_lower), np.atleast_2d(params_upper))
        medians_seq = medians_seq.squeeze()
        std_y = np.std(smoothed_deltas)
        if std_y > 0.015:
            window = 3 if std_y <= 0.03 else 5
            smoothed_deltas = pd.Series(smoothed_deltas).rolling(window=window, center=True, min_periods=1).mean().values
        smoothed = smoothed_deltas + medians_seq

    # remove spikes
    for i in range(1, len(smoothed)-1):
        if (abs(smoothed[i-1] - smoothed[i]) > 0.2) and (abs(smoothed[i] - smoothed[i+1]) > 0.2):
            smoothed[i] = 0.5 * (smoothed[i-1] + smoothed[i+1])

    base_idx_start = (start_date - dates_list_local[0]).days
    base_idx_end = (end_date - dates_list_local[0]).days
    base_idx_start = max(0, base_idx_start)
    base_idx_end = min(ndvi_zarr_local.shape[0]-1, base_idx_end)
    smoothed_scaled = np.clip(smoothed * 10000.0, 0, 10000).astype(np.int16)
    ndvi_zarr_local[base_idx_start:base_idx_end+1, pixel_idx] = smoothed_scaled[: base_idx_end - base_idx_start + 1]


"""def lowess_like_smoothing_gpu(delta_vals, n_iter=5, frac=1.0):

    n_times = delta_vals.shape[1]
    sigma = n_times * frac / 6.0
    kernel_size = int(6 * sigma + 1)
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, -1)

    deltas = delta_vals.unsqueeze(1)

    for _ in range(n_iter):
        smoothed = F.conv1d(deltas, kernel, padding="same")
        residuals = deltas - smoothed
        mad = torch.median(torch.abs(residuals), dim=2, keepdim=True)[0] + 1e-6
        weights = 1.0 / (1.0 + (residuals / (6 * mad)) ** 2)
        deltas = deltas * weights
        deltas = F.conv1d(deltas, kernel, padding="same")

    return deltas.squeeze(1)"""

# =====================================================
# Main continuous NDVI per-day batching
# =====================================================
def continous_ndvi_day(day_date: date, selected_pixels: np.ndarray,
                       ndvi_zarr, last_dates_zarr, params_lower_zarr, params_upper_zarr, dates_list):
    """
    Process one date for a batch of pixels. This function:
    - computes medians on GPU for the current day for all selected pixels
    - iterates per-pixel for the rest of the logic (outlier detection, potential retroactive, L1 interpolation)
    - uses GPU for numeric medians & interpolation calls, but keeps smoothing on CPU
    """
    base_date = dates_list[0]
    date_index = (day_date - base_date).days
    if not (0 <= date_index < ndvi_zarr.shape[0]):
        return

    doy = day_date.timetuple().tm_yday

    # Read raw NDVI for this day for selected pixels (no caching beyond this)
    ndvi_vals_raw = np.array(ndvi_zarr[date_index, selected_pixels], dtype=np.int16)  # int16
    ndvi_vals = ndvi_vals_raw.astype(np.float32) / 10000.0  # float NDVI

    # Read parameter arrays for the selected pixels (small transfer to CPU memory)
    params_lower_batch = np.array(params_lower_zarr[selected_pixels], dtype=np.float32)  # (P,6)
    params_upper_batch = np.array(params_upper_zarr[selected_pixels], dtype=np.float32)  # (P,6)

    # Compute medians for current DOY for all selected pixels on GPU
    medians_batch = calculate_median(np.array([doy], dtype=np.float32), params_lower_batch, params_upper_batch)
    # medians_batch shape -> (P,) because we passed single doy
    if medians_batch.ndim == 1:
        medians_batch = medians_batch
    else:
        medians_batch = medians_batch.squeeze(0)

    # For each pixel in the selected set, run the original logic
    for i_local, pixel_idx in enumerate(selected_pixels):
        try:
            ndvi_val = ndvi_vals[i_local]
            current_median = float(medians_batch[i_local])
            # last_date and potential date
            try:
                last_date_raw = last_dates_zarr[6, pixel_idx]
            except Exception:
                last_date_raw = b"1900-01-01"
            last_date = zarr_date_to_date(last_date_raw) if not np.all(last_date_raw == b"1900-01-01") else date(1900,1,1)

            potential_date_raw = last_dates_zarr[7, pixel_idx]
            potential_date = zarr_date_to_date(potential_date_raw) if not np.all(potential_date_raw == b"1900-01-01") else date(1900,1,1)

            params_lower = params_lower_batch[i_local]
            params_upper = params_upper_batch[i_local]

            if last_date != date(1900,1,1):
                last_doy = last_date.timetuple().tm_yday
                last_idx = (last_date - base_date).days
                if 0 <= last_idx < ndvi_zarr.shape[0]:
                    last_ndvi = ndvi_zarr[last_idx, pixel_idx] / 10000.0
                else:
                    last_ndvi = np.nan
                delta_prev = last_ndvi - float(calculate_median(np.array([last_doy], dtype=np.float32),
                                                                 np.atleast_2d(params_lower),
                                                                 np.atleast_2d(params_upper)).squeeze())
            else:
                delta_prev = None

            current_delta = ndvi_val - current_median if not np.isnan(ndvi_val) else None

            # handle invalid NDVI values
            if (ndvi_val >= 1) or (ndvi_val <= 0) or (np.isnan(ndvi_val)):
                if last_date != date(1900,1,1) and delta_prev is not None:
                    estimation = estimate_ndvi((day_date - last_date).days, current_median, delta_prev)
                    if 0 <= estimation <= 1:
                        ndvi_zarr[date_index, pixel_idx] = np.int16(np.clip(estimation * 10000.0, 0, 10000.0))
                continue

            # outlier detection
            if last_date != date(1900,1,1) and delta_prev is not None:
                true_value = outlier_detection_scalar(ndvi_val, current_median - 0.1, current_median + 0.1,
                                                      current_delta, delta_prev)
            else:
                true_value = True

            if true_value:
                if potential_date != date(1900,1,1):
                    pd_idx = (potential_date - base_date).days
                    if 0 <= pd_idx < ndvi_zarr.shape[0]:
                        potential_ndvi = ndvi_zarr[pd_idx, pixel_idx] / 10000.0
                    else:
                        potential_ndvi = np.nan

                    accepted = retroactive_outlier_detection_scalar(potential_date, potential_ndvi,
                                                                   ndvi_val, current_median,
                                                                   params_lower, params_upper)

                    if accepted:
                        # L1 interpolation: pot->current and pot->last
                        pdoy = potential_date.timetuple().tm_yday
                        potential_median = float(calculate_median(np.array([pdoy], dtype=np.float32),
                                                                  np.atleast_2d(params_lower),
                                                                  np.atleast_2d(params_upper)).squeeze())
                        potential_delta = potential_ndvi - potential_median

                        # GPU numeric interpolation function will compute medians for ranges on GPU
                        L1_interpolation_gpu(potential_delta, current_delta, potential_date, day_date,
                                             base_date, params_lower, params_upper, pixel_idx, ndvi_zarr)

                        if last_date != date(1900,1,1):
                            L1_interpolation_gpu(potential_delta, delta_prev, potential_date, last_date,
                                                 base_date, params_lower, params_upper, pixel_idx, ndvi_zarr)

                        # update last_dates window: shift and insert potential_date and day_date
                        old_dates = last_dates_zarr[:7, pixel_idx].copy()
                        old_dates = np.array(old_dates, dtype='S10')
                        shifted = np.empty(old_dates.shape, dtype='S10')
                        shifted[:-2] = old_dates[1:-1]
                        shifted[-2] = potential_date.strftime("%Y-%m-%d").encode("utf-8")
                        shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                        last_dates_zarr[:7, pixel_idx] = shifted

                        valid_window = [zarr_date_to_date(d) for d in shifted]
                        if all(d != date(1900, 1, 1) for d in valid_window):
                            # smoothing kept on CPU - call CPU smoothing
                            L2_smoothing_cpu(pixel_idx, 1, params_lower, params_upper,
                                             last_dates_zarr, ndvi_zarr, dates_list)

                        # clear potential slot
                        last_dates_zarr[7:, pixel_idx] = b"1900-01-01"

                    else:
                        # rejected potential: do single L1 from last to current (if last exists)
                        if last_date != date(1900,1,1):
                            L1_interpolation_gpu(delta_prev, current_delta, last_date, day_date,
                                                 base_date, params_lower, params_upper, pixel_idx, ndvi_zarr)
                        # update last_dates: shift and append day_date
                        old_dates = last_dates_zarr[:7, pixel_idx].copy()
                        old_dates = np.array(old_dates, dtype='S10')
                        shifted = np.empty(old_dates.shape, dtype='S10')
                        shifted[:-1] = old_dates[1:]
                        shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                        last_dates_zarr[:7, pixel_idx] = shifted
                        valid_window = [zarr_date_to_date(d) for d in shifted]
                        if all(d != date(1900, 1, 1) for d in valid_window):
                            L2_smoothing_cpu(pixel_idx, 2, params_lower, params_upper,
                                             last_dates_zarr, ndvi_zarr, dates_list)
                        # remove potential slot
                        last_dates_zarr[7:, pixel_idx] = b"1900-01-01"
                else:
                    # no potential entry: perform normal L1 (last->current), update last_dates, smoothing if needed
                    if last_date != date(1900,1,1):
                        L1_interpolation_gpu(delta_prev, current_delta, last_date, day_date,
                                             base_date, params_lower, params_upper, pixel_idx, ndvi_zarr)
                    old_dates = last_dates_zarr[:7, pixel_idx].copy()
                    old_dates = np.array(old_dates, dtype='S10')
                    shifted = np.empty(old_dates.shape, dtype='S10')
                    shifted[:-1] = old_dates[1:]
                    shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                    last_dates_zarr[:7, pixel_idx] = shifted
                    valid_window = [zarr_date_to_date(d) for d in shifted]
                    if all(d != date(1900, 1, 1) for d in valid_window):
                        L2_smoothing_cpu(pixel_idx, 2, params_lower, params_upper,
                                         last_dates_zarr, ndvi_zarr, dates_list)
                    last_dates_zarr[7:, pixel_idx] = b"1900-01-01"
            else:
                # flagged as potential
                date_to_potential = day_date.strftime("%Y-%m-%d").encode("utf-8")
                last_dates_zarr[7:, pixel_idx] = date_to_potential

        except Exception as e:
            # per-pixel exceptions should not crash the whole file
            print(f"[{pixel_idx}] ERROR in continous_ndvi_day: {e}")
            traceback.print_exc()
            continue

# =====================================================
# Remaining scalar helpers (small functions kept on CPU)
# =====================================================
def estimate_ndvi(days_diff, median, delta_prev):
    decrease_factor = math.exp(-math.log(2) * (days_diff / 15.0))
    return median + delta_prev * decrease_factor

def outlier_detection_scalar(obs, lower, upper, delta_current, delta_previous):
    inside_band = (obs >= lower) and (obs <= upper)
    delta_delta = delta_current - delta_previous if (delta_current is not None and delta_previous is not None) else 0.0
    if inside_band:
        return True
    if ((delta_current is not None and (delta_current > 0.05 or delta_current < -0.05))
        and ((delta_delta > 0.1) or (delta_delta < -0.1))):
        return False
    return True

def retroactive_outlier_detection_scalar(potential_date, potential_ndvi, obs, current_median, params_lower, params_upper):
    pdoy = potential_date.timetuple().tm_yday
    potential_median = float(calculate_median(np.array([pdoy], dtype=np.float32),
                                              np.atleast_2d(params_lower),
                                              np.atleast_2d(params_upper)).squeeze())
    delta_delta = ((obs - current_median) - (potential_ndvi - potential_median))
    return (delta_delta < 0.1) and (delta_delta > -0.1)

# =====================================================
# PROCESS SINGLE FILE (worker function)
# =====================================================
def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_DIR, file_name)

        tmp_out_path = out_path + f".tmp_{os.getpid()}"
        if os.path.exists(tmp_out_path):
            shutil.rmtree(tmp_out_path)
        shutil.copytree(file_path, tmp_out_path)
        os.rename(tmp_out_path, out_path)

        ds = zarr.open_group(out_path, mode="r+")
        global ndvi_zarr, last_dates_zarr, params_lower_zarr, params_upper_zarr, dates_list

        ndvi_zarr = ds["ndvi"]
        last_dates_zarr = ds["last_dates"]
        params_lower_zarr = ds["params"]["params_lower"]
        params_upper_zarr = ds["params"]["params_upper"]
        dates_zarr = ds["dates"]
        dates_list = [zarr_date_to_date(d) for d in dates_zarr[:]]

        n_pixels = ndvi_zarr.shape[1]
        seed = int(hashlib.md5(file_name.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        # if N_PIXELS_PER_FILE >= n_pixels we process all pixels
        to_select = min(N_PIXELS_PER_FILE, n_pixels)
        selected_pixels = rng.choice(np.arange(n_pixels), size=to_select, replace=False)
        print(f"[{file_name}] Selected {len(selected_pixels)} pixels out of {n_pixels}")

        # per-day continuous ingestion loop
        days_to_run = dates_list if MAX_DAYS is None else dates_list[:MAX_DAYS]
        for day_date in days_to_run:
            continous_ndvi_day(day_date, selected_pixels, ndvi_zarr, last_dates_zarr, params_lower_zarr, params_upper_zarr, dates_list)

        print(f"[{file_name}] Done.")
        gc.collect()
        return {"file": file_name, "status": "ok"}

    except Exception as e:
        print(f"[{file_path}] FAILED: {e}")
        traceback.print_exc()
        return {"file": file_path, "status": "error", "error": str(e)}

# =====================================================
# RUN PARALLEL (main)
# =====================================================
if __name__ == "__main__":
    all_entries = sorted(os.listdir(INPUT_DIR))
    zarr_files = [os.path.join(INPUT_DIR, e) for e in all_entries if e.endswith(".zarr")]
    if N_FILES is not None:
        zarr_files = zarr_files[:N_FILES]
    print(f"Starting analysis on {len(zarr_files)} files with up to {N_PIXELS_PER_FILE} pixels each.")

    # NOTE: keep the same ProcessPoolExecutor pattern. Each worker will open its own Zarr file.
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(zarr_files), os.cpu_count() or 1)) as executor:
        futures = {executor.submit(process_file, path): path for path in zarr_files}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            print("Result:", res)

    print("✅ All processing complete.")
