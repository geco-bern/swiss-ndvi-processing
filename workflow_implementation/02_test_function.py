# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/02_test_function.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_function.log 

import numpy as np
import pandas as pd
import zarr
import torch
import math
from datetime import datetime, timedelta
import statsmodels.api as sm
import os
import gc
from datetime import datetime, date, timedelta 
import re
import shutil

# --- Load Zarr dataset ---
INPUT_ZARR = "/data_2/scratch/francesco/zarr_demo_daily/"
dst = "/data_2/scratch/francesco/zarr_demo_daily_output"
shutil.copytree(INPUT_ZARR, dst,  dirs_exist_ok=True)

ds = zarr.open_group(INPUT_ZARR, mode="r")
ds_out = zarr.open_group(dst, mode="r+") 

# read the copyed value (at this stage are identical to the original ones)
ndvi_zarr = ds_out["ndvi"]
dates_zarr = ds_out["dates"]
params_lower_zarr = ds_out["params"]["params_lower"]
params_upper_zarr = ds_out["params"]["params_upper"]
last_dates_zarr = ds_out["last_dates"]


# --- Helper: decode and parse Zarr date entries ---
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

# --- Extract all valid dates from Zarr ---
dates = []
for i in range(dates_zarr.shape[0]):
    d_arr = dates_zarr.get_basic_selection((i,))
    d_val = d_arr[()] if isinstance(d_arr, np.ndarray) and d_arr.shape == () else d_arr[0]
    d_dt = _parse_zarr_date(d_val)
    if pd.notna(d_dt):
        dates.append(d_dt)

dates = sorted(list(set(dates)))
if not dates:
    raise ValueError("No valid dates found in the Zarr dataset.")

# --- Double logistic function ---
def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = params
    mat_minus_sos = np.log1p(np.exp(mat_minus_sos))  # softplus
    eos_minus_sen = np.log1p(np.exp(eos_minus_sen))
    
    sigmoid_sos_mat = 1 / (1 + np.exp(-2 * (2*sos + mat_minus_sos - 2*t) / (mat_minus_sos + 1e-10)))
    sigmoid_sen_eos = 1 / (1 + np.exp(-2 * (2*sen + eos_minus_sen - 2*t) / (eos_minus_sen + 1e-10)))
    
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

def calculate_median(doy, params_lower, params_upper):
    t = doy / 365.0
    lower = double_logistic_function(t, params_lower)
    upper = double_logistic_function(t, params_upper)
    return 0.5 * (upper + lower)

def estimate_ndvi(days_diff, median, delta_prev):
    decrease_factor = math.exp(-math.log(2) * (days_diff / 15))
    return median + delta_prev * decrease_factor

def unwrap_scalar(x):
    """Recursively unwrap 0-d arrays until we get a real scalar."""
    while isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    return x

def zarr_date_to_date(zarr_date):
    zarr_date = unwrap_scalar(zarr_date)

    if isinstance(zarr_date, bytes):
        return datetime.strptime(zarr_date.decode("utf-8"), "%Y-%m-%d").date()
    elif isinstance(zarr_date, np.datetime64):
        return zarr_date.astype('M8[D]').astype(datetime).date()
    elif isinstance(zarr_date, datetime):
        return zarr_date.date()
    elif isinstance(zarr_date, date):
        return zarr_date
    else:
        raise TypeError(f"Unknown date type: {type(zarr_date)}")

def L1_interpolation(days_diff, delta_1, delta_2, date_1, date_2, base_date, params_lower, params_upper,pixel_idx):

    L1_deltas = np.linspace(delta_1, delta_2, num=days_diff + 1)
    doy_1 = date_1.timetuple().tm_yday
    doy_2 = date_2.timetuple().tm_yday
    idx_1 = (date_1 - base_date).days
    idx_2 = (date_2 - base_date).days

    doys = np.linspace(doy_1, doy_2, num=days_diff + 1)
    medians = calculate_median(doys, params_lower, params_upper)

    ndvi = L1_deltas + medians
    ndvi_scaled = np.clip(ndvi * 10000, 0, 10000).astype(np.int16)

    ndvi_zarr[idx_1: idx_2 + 1, pixel_idx] = ndvi_scaled


# --- Main function ---
def print_ndvi(day, pixel_idx):

    base_date = zarr_date_to_date(dates[0])
    date_index = (day - base_date).days
    ndvi_val = ndvi_zarr[date_index, pixel_idx]

    # Read last date from Zarr
    last_date_raw = last_dates_zarr[6, pixel_idx]
    last_date = zarr_date_to_date(last_date_raw)

    params_lower = params_lower_zarr[pixel_idx]
    params_upper = params_upper_zarr[pixel_idx]

    doy = day.timetuple().tm_yday

    if last_date !=  date(1900, 1, 1):
        last_doy = last_date.timetuple().tm_yday
        days_diff = (day - last_date).days

        last_idx = (last_date - base_date).days

        last_ndvi = ndvi_zarr[last_idx, pixel_idx]
        last_ndvi = last_ndvi / 10000.0
        delta_prev = last_ndvi - calculate_median(last_doy, params_lower, params_upper)

    ndvi_val = ndvi_val / 10000.0

    current_median = calculate_median(doy, params_lower, params_upper)
    current_delta = ndvi_val - current_median

    if ndvi_val >= 1 or ndvi_val <= 0:
        ndvi_val = np.nan

        # Write last date back to Zarr
        if last_date !=  date(1900, 1, 1):
            estimation = estimate_ndvi(days_diff, calculate_median(doy, params_lower, params_upper), delta_prev)
            ndvi_zarr[date_index, pixel_idx] = np.int16(estimation * 10000)

    else:

        if last_date !=  date(1900, 1, 1):
            L1_interpolation(days_diff, delta_prev, current_delta, last_date, day,base_date, params_lower, params_upper,pixel_idx)
        
        last_dates_zarr[6:7, pixel_idx] = np.array([day.strftime("%Y-%m-%d").encode("utf-8")], dtype=object)

    #print(f"Pixel {pixel_idx:>4} | {day} | DOY={doy} | NDVI={ndvi_val:.4f} | last date={last_date}")

# --- Example run ---
for i in range(3000):
    day = zarr_date_to_date(dates[i])
    print_ndvi(day, 42)

import matplotlib.pyplot as plt

# Reopen the NDVI zarr group
root = zarr.open("/data_2/scratch/francesco/zarr_demo_daily_output", mode="r")

# Access the actual NDVI array
ndvi_arr = root["ndvi"]

# Extract NDVI time series
base_date = zarr_date_to_date(dates[0])
date_list = [zarr_date_to_date(d) for d in dates[:3000]]
ndvi_series = ndvi_arr[:3000, 42] / 10000.0

# Plot
plt.figure(figsize=(10, 5))
plt.plot(date_list, ndvi_series, marker="o", label="NDVI (observed + estimated)")
plt.xlabel("Date")
plt.ylabel("NDVI")
plt.title(f"NDVI Time Series for Pixel 42")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join("/home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/figure", "L0.png"), dpi=300, bbox_inches="tight")

plt.show()
