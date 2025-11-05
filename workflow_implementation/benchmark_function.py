# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/benchmark_function.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/output/log/benchmark_function.log 

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
from zarr.storage import LocalStore as FSStore
import xarray as xr 





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

    if isinstance(zarr_date, np.ndarray):
        zarr_date = int(zarr_date.item()) 

    years = (zarr_date // 10000)
    months = ((zarr_date % 10000) // 100)
    days = (zarr_date % 100)

    date_to_return = date(int(years), int(months), int(days))
        
    return date_to_return 


T_SCALE = 1.0 / 365.0

def double_logistic_function(t, params):
    """
    t: 1D tensor/array of shape (n,)
    params: array-like with 6 parameters per row (shape (n_sets,6)) or a single 1D length-6 vector.
    Returns: tensor shaped (n, n_sets) or (n,1) for single param set.
    """
    # convert params -> (n_sets, 6)
    params_t = torch.as_tensor(params, dtype=torch.float32)
    if params_t.ndim == 1:
        params_t = params_t.unsqueeze(0)   # (1,6)

    # convert t -> 1D tensor
    t_t = torch.as_tensor(t, dtype=torch.float32)
    if t_t.ndim == 0:
        t_t = t_t.unsqueeze(0)

    # split along columns (dim=1) now valid
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params_t, 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)

    # broadcast t to (n,1) then compute
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t_t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t_t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

def calculate_median(doys, params_lower, params_upper, device=torch.device("cpu")):

    doys = np.asarray(doys, dtype=np.float32)
    t = torch.tensor(doys * T_SCALE, dtype=torch.float32, device=device)

    lower = double_logistic_function(t, params_lower).squeeze().cpu().numpy()
    upper = double_logistic_function(t, params_upper).squeeze().cpu().numpy()
    med = 0.5 * (upper + lower)

    return med

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
    potential_median = calculate_median(pdoy, params_lower, params_upper, device=device)
    delta_delta = ((obs - current_median) - (potential_ndvi - potential_median))
    return (delta_delta < 0.1) and (delta_delta > -0.1)

def L1_interpolation(delta_1, delta_2, date_1, date_2, base_date, params_lower, params_upper, ndvi_arr, device):

    if date_1 <= date_2:
        start_date, end_date = date_1, date_2
        start_delta, end_delta = delta_1, delta_2
    else:
        start_date, end_date = date_2, date_1
        start_delta, end_delta = delta_2, delta_1

    days = (end_date - start_date).days

    if days == 0:

        doy = start_date.timetuple().tm_yday
        median = calculate_median(doy, params_lower, params_upper, device=device)[0]
        ndvi_val = start_delta + median
        ndvi_scaled = np.clip(int(np.round(ndvi_val * 10000.0)), 0, 10000).astype(np.int16)
        idx = (start_date - base_date).days

        if 0 <= idx < ndvi_arr.shape[0]:
            ndvi_arr[idx] = ndvi_scaled

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

    medians = []
    for doy in doys:
        median = calculate_median(doy, params_lower, params_upper, device=device)
        if isinstance(median, np.ndarray):
            median = median.item()
        medians.append(median)

    medians = np.array(medians, dtype=np.float32)

    ndvi = L1_deltas + medians
    ndvi_scaled = np.round(ndvi * 10000).astype(np.int16)
    idx_start = (start_date - base_date).days
    idx_end = (end_date - base_date).days
    a = max(0, idx_start)
    b = min(ndvi_arr.shape[0] - 1, idx_end)
    ndvi_arr[(a - idx_start):(b - idx_start) + 1] = ndvi_scaled[(a - idx_start):(b - idx_start) + 1]

def L2_smoothing(init_position, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device):

    last_dates_bytes = last_dates_arr[:7]
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
            ndvi_val = ndvi_arr[idx] / 10000.0
        else:
            ndvi_val = np.nan
        d_doy = d.timetuple().tm_yday
        median_val = calculate_median(d_doy, params_lower, params_upper, device=device)
        if isinstance(median_val, np.ndarray):
            median_val = median_val.item()
        if isinstance(ndvi_val, np.ndarray):
            ndvi_val = ndvi_val.item()
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
    ndvi_arr[base_idx_start:base_idx_end + 1] = smoothed_scaled[: base_idx_end - base_idx_start + 1]

def continous_ndvi(day_date, ndvi_arr, last_dates_arr, params_lower, params_upper, dates_list, device):


    base_date = dates_list[0]

    doy = day_date.timetuple().tm_yday
    t_day = torch.tensor(doy * T_SCALE, dtype=torch.float32, device=device)

    lower = double_logistic_function(t_day, params_lower).squeeze().cpu().numpy()
    upper = double_logistic_function(t_day, params_upper).squeeze().cpu().numpy()   

    date_index = (day_date - base_date).days
    if not (0 <= date_index < ndvi_arr.shape[0]):
        return ndvi_arr, last_dates_arr
    ndvi_val_raw = ndvi_arr[date_index]
    last_date_raw = last_dates_arr[6]
    last_date = zarr_date_to_date(last_date_raw)

    last_date = to_date(last_date)

    potential_date_raw = last_dates_arr[7]
    potential_date = zarr_date_to_date(potential_date_raw)
    
    potential_date = to_date(potential_date)

    

    if last_date != date(1900,1,1):
        last_doy = last_date.timetuple().tm_yday

        last_idx = (last_date - base_date).days
        last_ndvi = ndvi_arr[last_idx] / 10000.0
        delta_prev = last_ndvi - calculate_median(last_doy, params_lower, params_upper, device=device)
    else:
        delta_prev = None

    ndvi_val = ndvi_val_raw / 10000.0 if ndvi_val_raw is not None else np.nan
    current_median = calculate_median(doy, params_lower, params_upper, device=device)

    current_delta = ndvi_val - current_median if not np.isnan(ndvi_val) else None

    if ndvi_val >= 1 or ndvi_val <= 0 or np.isnan(ndvi_val):

        if last_date != date(1900,1,1) and delta_prev is not None:

            estimation = estimate_ndvi((day_date - last_date).days, current_median, delta_prev)
            if 0 <= estimation <= 1:
                ndvi_arr[date_index] = np.int16(np.clip(int(np.round(estimation * 10000.0)), 0, 10000))
        return ndvi_arr, last_dates_arr
    
    if last_date != date(1900,1,1) and delta_prev is not None:

        true_value = outlier_detection(ndvi_val, lower, upper, current_delta, delta_prev)

    else:

        true_value = True

    if true_value:

        if potential_date != date(1900,1,1):

            pd_idx = (potential_date - base_date).days
            potential_ndvi = ndvi_arr[pd_idx] / 10000.0

            accepted = retroactive_outlier_detection(potential_date, potential_ndvi, ndvi_val, current_median, params_lower, params_upper, device)

            if accepted:

                pdoy = potential_date.timetuple().tm_yday
                potential_median = calculate_median(pdoy, params_lower, params_upper, device=device)

                potential_delta = potential_ndvi - potential_median
                L1_interpolation(potential_delta, current_delta, potential_date, day_date, base_date, params_lower, params_upper, ndvi_arr, device)

                if last_date != date(1900,1,1):
                    L1_interpolation(potential_delta, delta_prev, potential_date, last_date, base_date, params_lower, params_upper, ndvi_arr, device)

                old_dates = last_dates_arr[:7].copy()
                old_dates = last_dates_arr[:7].copy().astype(np.int32)

                shifted = np.empty_like(old_dates, dtype=np.int32) 
                shifted[:-2] = old_dates[1:-1]
                shifted[-2] = potential_date.year * 10000 + potential_date.month * 100 + potential_date.day
                shifted[-1] = day_date.year * 10000 + day_date.month * 100 + day_date.day
                last_dates_arr[:7] = shifted

                valid_window = [zarr_date_to_date(d) for d in shifted]

                if all(d != date(1900,1,1) for d in valid_window):
                    r = 3
                    L2_smoothing( 1, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
                last_dates_arr[7:] = 19000101

            else:

                if last_date != date(1900,1,1):

                    L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, ndvi_arr, device)

                old_dates = last_dates_arr[:7].copy().astype(np.int32)
                shifted = np.empty_like(old_dates, dtype=np.int32)
                shifted[:-1] = old_dates[1:]
                shifted[-1] = day_date.year * 10000 + day_date.month * 100 + day_date.day

                last_dates_arr[:7] = shifted
                valid_window = [zarr_date_to_date(d) for d in shifted]

                if all(d != date(1900,1,1) for d in valid_window):
                    r = 3
                    L2_smoothing( 2, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
                
                last_dates_arr[7:] = 19000101
        else:

            if last_date != date(1900,1,1):
                
                L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, ndvi_arr, device)

            old_dates = last_dates_arr[:7].copy().astype(np.int32)
            shifted = np.empty_like(old_dates, dtype=np.int32)
            shifted[:-1] = old_dates[1:]
            shifted[-1] = day_date.year * 10000 + day_date.month * 100 + day_date.day

            last_dates_arr[:7] = shifted

            valid_window = [zarr_date_to_date(d) for d in shifted]

            if all(d != date(1900,1,1) for d in valid_window):
                r = 3
                L2_smoothing( 2, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
            last_dates_arr[7:] = 19000101
    else:

        date_to_potential = day_date.year * 10000 + day_date.month * 100 + day_date.day
        last_dates_arr[7:] = date_to_potential

    return ndvi_arr, last_dates_arr


INPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"

#pixel_rel_idx = 905083

pixels = np.random.choice(999999, size=100, replace=False).astype(np.int_)

ds = xr.open_zarr(INPUT_DIR, consolidated=True)
params = xr.open_zarr(INPUT_DIR, group="params", consolidated=False)
device = "cpu"

# get DataArrays

ndvi_da = ds["ndvi"]            # dims ("time","pixel")
last_dates_da = ds["last_dates"]# dims ("band","pixel")
params_lower_da = params["params_lower"] # dims ("pixel","param")
params_upper_da = params["params_upper"] # dims ("pixel","param")

dates_int = ds["dates"].values.astype(np.int32)
dates_list = [datetime.strptime(str(d), "%Y%m%d").date() for d in dates_int]

# loading data

timing = {"loading": 0,'computing': 0.0, "writing" : 0}

t_start_loading = time.perf_counter()

ndvi_arr = ndvi_da.isel(pixel=pixels).values
print(ndvi_arr)
print(ndvi_arr.shape)
last_dates_arr = last_dates_da.isel(pixel=pixels).values 
params_lower_arr =params_lower_da.isel(pixel=pixels).values  
params_upper_arr = params_upper_da.isel(pixel=pixels).values  

t_loading = time.perf_counter() - t_start_loading
timing["loading"] = t_loading

# computing

t_start = time.perf_counter()

for idx_in_block, pixel_global_idx in enumerate(pixels):
    # per-pixel 1-D views (ndvi: time, last_dates: band)
    ndvi_pixel = ndvi_arr[:, idx_in_block].copy()
    last_dates_pixel = last_dates_arr[:, idx_in_block].copy()
    params_lower_pixel = params_lower_arr[idx_in_block]
    params_upper_pixel = params_upper_arr[idx_in_block]

    # process this pixel over all dates and update the pixel arrays
    for day_date in dates_list:
        ndvi_pixel, last_dates_pixel = continous_ndvi(
            day_date,
            ndvi_pixel,
            last_dates_pixel,
            params_lower_pixel,
            params_upper_pixel,
            dates_list,
            device
        )

    t_compute = time.perf_counter() - t_start
    timing["computing"] = t_compute

    # write the updated pixel back to the block arrays
    ndvi_arr[:, idx_in_block] = ndvi_pixel
    last_dates_arr[:, idx_in_block] = last_dates_pixel





"""# writing data
# ---- writing data (fixed) ----
OUTPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_v2_processed.zarr/"

# Remove existing folder if present
if os.path.exists(OUTPUT_DIR):
    print(f"Removing existing output Zarr store at {OUTPUT_DIR} ...")
    shutil.rmtree(OUTPUT_DIR)

print(f"Creating new output Zarr store at {OUTPUT_DIR} ...")

# Remove or neutralize incompatible compressor metadata to avoid zarr v3 codec issues
for v in ds.data_vars:
    ds[v].encoding.pop("compressor", None)
for v in ds.coords:
    ds[v].encoding.pop("compressor", None)

# Use no compressor for the output (safe). If you want Blosc as numcodecs object (z2), you can set it here.
encoding = {v: {"compressor": None} for v in ds.data_vars}

t_start_writing = time.perf_counter()

# Write dataset as Zarr v2 explicitly (no delayed compute)
# NOTE: zarr_version=2 forces xarray to create the legacy v2 layout if xarray supports the arg.
# compute=True ensures arrays are created on disk before we open and write slices.
ds.to_zarr(
    OUTPUT_DIR,
    mode="w",
    consolidated=True,
    encoding=encoding,
    zarr_version=2,
    compute=True,
)

# Re-open the store and write back the modified per-pixel arrays
# Use zarr.open_group or zarr.open depending on zarr version; zarr.open works for both.
zarr_out = zarr.open(OUTPUT_DIR, mode="r+")

# pixels is an array of ints (shape (1,) in your run)
for i, pixel_rel_idx in enumerate(pixels):
    pix = int(pixel_rel_idx)  # ensure Python int
    # ndvi_arr is 1D after squeeze: shape (time,)
    # last_dates_arr is 1D after squeeze: shape (band,)
    # so write directly
    zarr_out["ndvi"][:, pix] = ndvi_arr
    zarr_out["last_dates"][:, pix] = last_dates_arr

t_writing = time.perf_counter() - t_start_writing
timing["writing"] = t_writing"""

print(f"one pixel: loading ={t_loading:.2f}s, computing ={t_compute:.2f}s")
