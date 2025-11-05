# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/03_test_parallel_with_dask.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/output/log/zarr_parallel_continous_ndvi_gpu_ssd_3.log &

import os
import sys
import time
import math
import gc
import zarr
import shutil
import hashlib
import traceback
import concurrent.futures
from datetime import datetime, date
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
import dask.array as da
from dask.distributed import Client
from dask import delayed
import xarray as xr


INPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"
OUTPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_processed/"
N_WORKERS = 10
N_PIXELS_PER_FILE = 1
device = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def unwrap_scalar(x):
    while isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    return x

def to_date(obj):
    """Convert various date/time objects to datetime.date."""
    if obj is None:
        return None
    if isinstance(obj, date) and not isinstance(obj, datetime):
        return obj
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, (np.datetime64,)):
        return obj.astype("M8[D]").astype(datetime).date()
    if isinstance(obj, (bytes, str)):
        s = obj.decode("utf-8") if isinstance(obj, bytes) else obj
        # Try multiple common formats
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
    raise TypeError(f"Cannot convert {type(obj)} to datetime.date")

def zarr_date_to_date(zarr_date):
    """
    Convert int or np.ndarray of int in YYYYMMDD format to datetime.date.
    Works for both single integers and numpy arrays.
    """
    # Handle numpy array of ints
    if isinstance(zarr_date, np.ndarray):
        if not np.issubdtype(zarr_date.dtype, np.integer):
            raise TypeError(f"Expected int array, got {zarr_date.dtype}")
        years = zarr_date // 10000
        months = (zarr_date % 10000) // 100
        days = zarr_date % 100
        return np.array(
            [date(int(y), int(m), int(d)) for y, m, d in zip(years, months, days)],
            dtype=object
        )

    # Handle single integer
    elif isinstance(zarr_date, (int, np.integer)):
        y = zarr_date // 10000
        m = (zarr_date % 10000) // 100
        d = zarr_date % 100
        return date(int(y), int(m), int(d))

    else:
        raise TypeError(f"Expected int or np.ndarray of int, got {type(zarr_date)}")


T_SCALE = 1.0 / 365.0

def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

def calculate_median(doys, params_lower, params_upper, device=torch.device("cpu")):
    doys = np.asarray(doys, dtype=np.float32)
    t = torch.tensor(doys * T_SCALE, dtype=torch.float32, device=device)
    pl = params_lower
    pu = params_upper
    if isinstance(pl, np.ndarray):
        pl = torch.from_numpy(pl)
    if isinstance(pu, np.ndarray):
        pu = torch.from_numpy(pu)
    pl = pl.to(device=device)
    pu = pu.to(device=device)
    if pl.ndim == 1:
        pl = pl.unsqueeze(0)
    if pu.ndim == 1:
        pu = pu.unsqueeze(0)
    lower = double_logistic_function(t, pl)
    upper = double_logistic_function(t, pu)
    med = 0.5 * (upper + lower)
    med_np = med.detach().cpu().numpy()
    if med_np.ndim == 2 and med_np.shape[1] == 1:
        med_np = med_np[:, 0]
    return med_np

def estimate_ndvi(days_diff, median, delta_prev):
    decrease_factor = math.exp(-math.log(2) * (days_diff / 15.0))
    return median + delta_prev * decrease_factor

def outlier_detection(obs, lower, upper, delta_current, delta_previous):
    inside_band = (obs >= lower) and (obs <= upper)
    delta_delta = delta_current - delta_previous
    if inside_band:
        return True
    if ((delta_current > 0.05) or (delta_current < -0.05)) and ((delta_delta > 0.1) or (delta_delta < -0.1)):
        return False
    return True

def retroactive_outlier_detection(potential_date, potential_ndvi, obs, current_median, params_lower, params_upper, device):
    pdoy = potential_date.timetuple().tm_yday
    potential_median = calculate_median([pdoy], params_lower, params_upper, device=device)[0]
    delta_delta = ((obs - current_median) - (potential_ndvi - potential_median))
    return (delta_delta < 0.1) and (delta_delta > -0.1)

def L1_interpolation(delta_1, delta_2, date_1, date_2, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device):
    if date_1 <= date_2:
        start_date, end_date = date_1, date_2
        start_delta, end_delta = delta_1, delta_2
    else:
        start_date, end_date = date_2, date_1
        start_delta, end_delta = delta_2, delta_1

    days = (end_date - start_date).days
    if days == 0:
        doy = start_date.timetuple().tm_yday
        median = calculate_median([doy], params_lower, params_upper, device=device)[0]
        ndvi_val = start_delta + median
        ndvi_scaled = np.clip(int(round(ndvi_val * 10000.0)), 0, 10000).astype(np.int16)
        idx = (start_date - base_date).days
        if 0 <= idx < ndvi_arr.shape[0]:
            ndvi_arr[idx, pixel_rel_idx] = ndvi_scaled
        return

    L1_deltas = np.linspace(start_delta, end_delta, num=days + 1, dtype=np.float32)
    doy_start = start_date.timetuple().tm_yday
    doy_end = end_date.timetuple().tm_yday
    days_diff = (end_date - start_date).days
    if doy_end < doy_start:
        doys = np.linspace(doy_start, doy_end + 365, num=days_diff + 1) % 365
        doys = np.where(doys == 0, 365, doys)
    else:
        doys = np.linspace(doy_start, doy_end, num=days_diff + 1)
    doys = np.where((doys == 0) | (doys == 366), 365, doys).astype(np.int32)
    medians = calculate_median(doys, params_lower, params_upper, device=device)
    ndvi = L1_deltas + medians
    ndvi_scaled = np.clip(np.round(ndvi * 10000.0).astype(np.int32), 0, 10000).astype(np.int16)
    idx_start = (start_date - base_date).days
    idx_end = (end_date - base_date).days
    a = max(0, idx_start)
    b = min(ndvi_arr.shape[0] - 1, idx_end)
    #ndvi_arr[a:b + 1, pixel_rel_idx] = ndvi_scaled[(a - idx_start):(b - idx_start) + 1]

def L2_smoothing(pixel_rel_idx, init_position, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device):
    last_dates_bytes = last_dates_arr[:7, pixel_rel_idx]
    last_dates = [zarr_date_to_date(d) for d in last_dates_bytes if not np.all(d == b'1900-01-01')]
    if len(last_dates) < 2:
        return
    ndvi_vals = []
    median_vals = []
    base_date = to_date(dates_list[0])

    for d in last_dates:

        d = to_date(d)
        idx = (d - base_date).days

        if 0 <= idx < ndvi_arr.shape[0]:
            ndvi_val = ndvi_arr[idx, pixel_rel_idx] / 10000.0
        else:
            ndvi_val = np.nan
        median_val = calculate_median([d.timetuple().tm_yday], params_lower, params_upper, device=device)[0]
        ndvi_vals.append(ndvi_val)
        median_vals.append(median_val)

    ndvi_vals = np.array(ndvi_vals, dtype=np.float32)
    median_vals = np.array(median_vals, dtype=np.float32)
    deltas_arr = ndvi_vals - median_vals
    start_date = last_dates[init_position]
    end_date = last_dates[-1]

    start_date = to_date(start_date)
    end_date = to_date(end_date)
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
        smoothed = np.interp(np.linspace(0, len(ndvi_vals) - 1, days_diff + 1), loess[:, 0], loess[:, 1])
    else:
        loess = sm.nonparametric.lowess(deltas_arr, idx, frac=1, it=5, return_sorted=True)
        smoothed_deltas = np.interp(np.linspace(0, len(deltas_arr) - 1, days_diff + 1), loess[:, 0], loess[:, 1])
        medians = calculate_median(doys, params_lower, params_upper, device=device)
        std_y = np.std(smoothed_deltas)
        if std_y > 0.015:
            window = 3 if std_y <= 0.03 else 5
            smoothed_deltas = pd.Series(smoothed_deltas).rolling(window=window, center=True, min_periods=1).mean().values
        smoothed = smoothed_deltas + medians

    for i in range(1, len(smoothed) - 1):
        if (abs(smoothed[i - 1] - smoothed[i]) > 0.2) and (abs(smoothed[i] - smoothed[i + 1]) > 0.2):
            smoothed[i] = 0.5 * (smoothed[i - 1] + smoothed[i + 1])

    base_idx_start = (start_date - dates_list[0]).days
    base_idx_end = (end_date - dates_list[0]).days
    base_idx_start = max(0, base_idx_start)
    base_idx_end = min(ndvi_arr.shape[0] - 1, base_idx_end)
    smoothed_scaled = np.clip(np.round(np.asarray(smoothed) * 10000.0).astype(np.int32), 0, 10000).astype(np.int16)
    #ndvi_arr[base_idx_start:base_idx_end + 1, pixel_rel_idx] = smoothed_scaled[: base_idx_end - base_idx_start + 1]

def continous_ndvi(day_date, pixel_rel_idx, pixel_global_idx, ndvi_arr, last_dates_arr, params_lower_arr, params_upper_arr, dates_list, device, timing):


    base_date = dates_list[0]

    day_date = to_date(day_date)
    base_date = to_date(dates_list[0])

    date_index = (day_date - base_date).days
    if not (0 <= date_index < ndvi_arr.shape[0]):
        return
    ndvi_val_raw = ndvi_arr[date_index, pixel_rel_idx]
    last_date_raw = last_dates_arr[6, pixel_rel_idx]
    last_date = zarr_date_to_date(last_date_raw) if not np.all(last_date_raw == b"1900-01-01") else date(1900,1,1)

    last_date = to_date(last_date)

    potential_date_raw = last_dates_arr[7, pixel_rel_idx]
    potential_date = zarr_date_to_date(potential_date_raw) if not np.all(potential_date_raw == b"1900-01-01") else date(1900,1,1)
    
    potential_date = to_date(potential_date)

    params_lower = params_lower_arr[pixel_rel_idx]
    params_upper = params_upper_arr[pixel_rel_idx]
    doy = day_date.timetuple().tm_yday

    if last_date != date(1900,1,1):
        last_doy = last_date.timetuple().tm_yday

        last_idx = (last_date - base_date).days
        last_ndvi = ndvi_arr[last_idx, pixel_rel_idx] / 10000.0
        delta_prev = last_ndvi - calculate_median([last_doy], params_lower, params_upper, device=device)[0]
    else:
        delta_prev = None
    ndvi_val = ndvi_val_raw / 10000.0 if ndvi_val_raw is not None else np.nan
    current_median = calculate_median([doy], params_lower, params_upper, device=device)[0]
    current_delta = ndvi_val - current_median if not np.isnan(ndvi_val) else None
    if ndvi_val >= 1 or ndvi_val <= 0 or np.isnan(ndvi_val):
        if last_date != date(1900,1,1) and delta_prev is not None:
            estimation = estimate_ndvi((day_date - last_date).days, current_median, delta_prev)
            #if 0 <= estimation <= 1:
                #ndvi_arr[date_index, pixel_rel_idx] = np.int16(np.clip(int(round(estimation * 10000.0)), 0, 10000))
        return
    if last_date != date(1900,1,1) and delta_prev is not None:
        true_value = outlier_detection(ndvi_val, current_median - 0.1, current_median + 0.1, current_delta, delta_prev)
    else:
        true_value = True
    if true_value:
        if potential_date != date(1900,1,1):
            pd_idx = (potential_date - base_date).days
            potential_ndvi = ndvi_arr[pd_idx, pixel_rel_idx] / 10000.0
            accepted = retroactive_outlier_detection(potential_date, potential_ndvi, ndvi_val, current_median, params_lower, params_upper, device)
            if accepted:
                pdoy = potential_date.timetuple().tm_yday
                potential_median = calculate_median([pdoy], params_lower, params_upper, device=device)[0]
                potential_delta = potential_ndvi - potential_median
                L1_interpolation(potential_delta, current_delta, potential_date, day_date, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device)
                if last_date != date(1900,1,1):
                    L1_interpolation(potential_delta, delta_prev, potential_date, last_date, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device)
                old_dates = last_dates_arr[:7, pixel_rel_idx].copy()
                old_dates = last_dates_arr[:7, pixel_rel_idx].copy().astype(np.int32)
                shifted = np.empty_like(old_dates, dtype=np.int32) 
                shifted[:-2] = old_dates[1:-1]
                shifted[-2] = potential_date.year * 10000 + potential_date.month * 100 + potential_date.day
                shifted[-1] = day_date.year * 10000 + day_date.month * 100 + day_date.day
                last_dates_arr[:7, pixel_rel_idx] = shifted
                valid_window = [zarr_date_to_date(d) for d in shifted]
                if all(d != date(1900,1,1) for d in valid_window):
                    r = 3
                    #L2_smoothing(pixel_rel_idx, 1, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
                last_dates_arr[7:, pixel_rel_idx] = 19000101
            else:
                if last_date != date(1900,1,1):
                    L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device)
                old_dates = last_dates_arr[:7, pixel_rel_idx].copy().astype(np.int32)
                shifted = np.empty_like(old_dates, dtype=np.int32)
                shifted[:-1] = old_dates[1:]
                shifted[-1] = day_date.year * 10000 + day_date.month * 100 + day_date.day
                last_dates_arr[:7, pixel_rel_idx] = shifted
                valid_window = [zarr_date_to_date(d) for d in shifted]
                if all(d != date(1900,1,1) for d in valid_window):
                    r = 3
                    #L2_smoothing(pixel_rel_idx, 2, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
                last_dates_arr[7:, pixel_rel_idx] = 19000101
        else:
            if last_date != date(1900,1,1):
                L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device)

            old_dates = last_dates_arr[:7, pixel_rel_idx].copy().astype(np.int32)
            shifted = np.empty_like(old_dates, dtype=np.int32)
            shifted[:-1] = old_dates[1:]
            shifted[-1] = day_date.year * 10000 + day_date.month * 100 + day_date.day
            last_dates_arr[:7, pixel_rel_idx] = shifted
            valid_window = [zarr_date_to_date(d) for d in shifted]
            if all(d != date(1900,1,1) for d in valid_window):
                r = 3
                #L2_smoothing(pixel_rel_idx, 2, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
            last_dates_arr[7:, pixel_rel_idx] = 19000101
    else:
        date_to_potential = day_date.year * 10000 + day_date.month * 100 + day_date.day
        last_dates_arr[7:, pixel_rel_idx] = date_to_potential


def process_block_for_apply_ufunc(ndvi_block, last_dates_block, params_lower_block, params_upper_block, dates_list, device="cpu"):
    """
    ndvi_block: 2D numpy array (time, pixels_in_block) dtype int16 (scaled by 10000)
    last_dates_block: 2D numpy array (band, pixels_in_block) dtype int32 or bytes representation
    params_lower_block: 2D numpy array (pixels_in_block, 6) float32
    params_upper_block: 2D numpy array (pixels_in_block, 6) float32

    Returns modified (ndvi_block, last_dates_block).
    """
    # Convert to writable arrays (should already be numpy arrays)
    ndvi_arr = np.array(ndvi_block, copy=False)
    last_dates_arr = np.array(last_dates_block, copy=False)
    params_lower_arr = np.array(params_lower_block, copy=False)
    params_upper_arr = np.array(params_upper_block, copy=False)

    # Local copies of helper functions are allowed (they are in global scope)
    # Use CPU device by default unless user supplied gpu-capable workers
    torch_device = torch.device(device)

    n_pixels = ndvi_arr.shape[1]
    # pre-convert dates_list to python dates for speed (may already be)
    base_dates_list = [to_date(d) for d in dates_list]

    # Iterate pixels and days similarly to your original logic
    for pix in range(n_pixels):
        # The functions in your code expect pixel index relative to the block
        # and modify ndvi_arr and last_dates_arr in-place.
        # We'll call continous_ndvi for each day in dates_list
        for day in base_dates_list:
            try:
                continous_ndvi(
                    day_date=day,
                    pixel_rel_idx=pix,
                    pixel_global_idx=pix,   # global index not available here; if required you can pass offset via kwargs
                    ndvi_arr=ndvi_arr,
                    last_dates_arr=last_dates_arr,
                    params_lower_arr=params_lower_arr,
                    params_upper_arr=params_upper_arr,
                    dates_list=base_dates_list,
                    device=torch_device,
                    timing=None
                )
            except Exception as e:
                # Avoid worker crash on single bad pixel — log and continue
                import traceback
                traceback.print_exc()
                # optionally continue
                continue

    # Must return numpy arrays (the modified blocks)
    return ndvi_arr.astype(np.int16), last_dates_arr.astype(np.int32)


# --- Open dataset with chunking (important) ---
# chunk on 'pixel' so each task gets a reasonable block

ds = xr.open_zarr(INPUT_DIR, consolidated=True)
ds = xr.open_zarr(INPUT_DIR, consolidated=False, chunks={"pixel": N_PIXELS_PER_FILE, "time": ds.sizes["time"]})
params = xr.open_zarr(INPUT_DIR, group="params", consolidated=False)

# get DataArrays

ndvi_da = ds["ndvi"]            # dims ("time","pixel")
last_dates_da = ds["last_dates"]# dims ("band","pixel")
params_lower_da = params["params_lower"] # dims ("pixel","param")
params_upper_da = params["params_upper"] # dims ("pixel","param")

block_size = 16
ndvi_da = ndvi_da.chunk({"pixel": block_size})
last_dates_da = last_dates_da.chunk({"pixel": block_size})
params_lower_da = params_lower_da.chunk({"pixel": block_size})
params_upper_da = params_upper_da.chunk({"pixel": block_size})

# dates_list as Python sequence (passed as kwargs)
dates_int = ds["dates"].values.astype(np.int32)
dates_list = [datetime.strptime(str(d), "%Y%m%d").date() for d in dates_int]

# Now call apply_ufunc
result = xr.apply_ufunc(
    process_block_for_apply_ufunc,
    ndvi_da,                    # input 1 (time,pixel)
    last_dates_da,              # input 2 (band,pixel)
    params_lower_da,            # input 3 (pixel,param)
    params_upper_da,            # input 4 (pixel,param)
    input_core_dims=[["time", "pixel"], ["band", "pixel"], ["pixel", "param"], ["pixel", "param"]],
    output_core_dims=[["time", "pixel"], ["band", "pixel"]],
    kwargs={"dates_list": dates_list, "device": "cpu"},
    vectorize=False,            # function consumes full core dims and returns full core dims
    dask="parallelized",
    dask_gufunc_kwargs={
        "allow_rechunk": True,
        "output_sizes": {"pixel": ds.sizes["pixel"]}
    },
    output_dtypes=[np.int16, np.int32],
)

# result is a tuple-like xarray object (xarray wraps multiple outputs)
processed_ndvi_da, processed_last_dates_da = result
print("done")

# processed_ndvi_da and processed_last_dates_da are lazy Dask-backed arrays.
# You can now write them to Zarr (lazy parallel write) or compute them explicitly.

# Example: write directly to Zarr lazily (recommended, efficient)
"""out_ds = xr.Dataset({
    "ndvi": processed_ndvi_da,
    "last_dates": processed_last_dates_da
}, coords={"time": ds["dates"], "pixel": ds["pixel"]})

# Choose chunking for output (align with inputs)
out_ds = out_ds.chunk({"pixel": N_PIXELS_PER_FILE, "time": ds.sizes["time"]})

# Write lazy result to zarr (this will compute the dask graph and write chunks)
out_ds.to_zarr(OUTPUT_DIR, mode="w", consolidated=True)"""
