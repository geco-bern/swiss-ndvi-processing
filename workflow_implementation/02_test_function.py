# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/02_test_function.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_function.log 

import numpy as np
import pandas as pd
import zarr
import torch
import math
import statsmodels.api as sm
import os
import gc
from datetime import datetime, date, timedelta 
import re
import shutil
import concurrent.futures

# --- Load Zarr dataset ---
INPUT_ZARR = "/data_2/scratch/francesco/zarr_demo_daily/"
dst = "/data_2/scratch/francesco/zarr_demo_daily_output"
# copy the dataset so I don't have to generate it again
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
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

def calculate_median(doy, params_lower, params_upper):
    # Ensure DOY is a torch tensor (1D)
    t = torch.as_tensor(np.atleast_1d(doy) / 365.0, dtype=torch.float32)

    # Add batch dimension if params are 1D (e.g., shape (6,))
    if params_lower.ndim == 1:
        params_lower = params_lower[None, :]
    if params_upper.ndim == 1:
        params_upper = params_upper[None, :]

    lower = double_logistic_function(t, params_lower).squeeze().numpy()
    upper = double_logistic_function(t, params_upper).squeeze().numpy()

    return 0.5 * (upper + lower)


def calculate_median_write(doy, params_lower, params_upper,pixel_idx,base_date,start_date):
    # Ensure DOY is a torch tensor (1D)
    t = torch.as_tensor(np.atleast_1d(doy) / 365.0, dtype=torch.float32)

    # Add batch dimension if params are 1D (e.g., shape (6,))
    if params_lower.ndim == 1:
        params_lower = params_lower[None, :]
    if params_upper.ndim == 1:
        params_upper = params_upper[None, :]

    lower = double_logistic_function(t, params_lower).squeeze().numpy()
    upper = double_logistic_function(t, params_upper).squeeze().numpy()

    median = 0.5 * (upper + lower)
    median_to_write = np.clip(median * 10000,  0, 10000).astype(np.int16) 
    idx = (start_date - base_date).days
    ndvi_zarr[idx, pixel_idx] = median_to_write

def date_to_fractional_year(date):
    """Convert a datetime.date or datetime.datetime to a continuous fractional year."""
    year_start = datetime(date.year, 1, 1)
    next_year_start = datetime(date.year + 1, 1, 1)
    year_length = (next_year_start - year_start).days
    fraction = (date - year_start).days / year_length
    return date.year + fraction


def outlier_detection(obs, lower, upper, delta_current, delta_previous):

    inside_band = (obs >= lower) and (obs <= upper)
    delta_delta = delta_current - delta_previous

    if inside_band:
        return True

    if ((delta_current  > 0.05) or (delta_current  < -0.05)) and ((delta_delta > 0.1) or (delta_delta < -0.1)):

        return False

    return True

def retroactive_outlier_detection(potential_date, potential_ndvi,
                                  obs, current_median, params_lower, params_upper,pixel_idx):

    pdoy = potential_date.timetuple().tm_yday
    potential_median = calculate_median(pdoy, params_lower, params_upper)

    delta_delta = ((obs - current_median) - (potential_ndvi - potential_median))

    if (delta_delta < 0.1) and (delta_delta > -0.1):

        return True

    else:

        return False

def estimate_ndvi(days_diff, median, delta_prev):
    decrease_factor = math.exp(-math.log(2) * (days_diff / 15))
    return median + delta_prev * decrease_factor

def unwrap_scalar(x):

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

def L1_interpolation(delta_1, delta_2, date_1, date_2, base_date, params_lower, params_upper, pixel_idx):

    # --- Ensure chronological order ---
    if date_1 <= date_2:
        start_date, end_date = date_1, date_2
        start_delta, end_delta = delta_1, delta_2
    else:
        start_date, end_date = date_2, date_1
        start_delta, end_delta = delta_2, delta_1

    # --- Compute number of days ---
    days = (end_date - start_date).days
    if days < 0:
        raise ValueError(f"Invalid date range: {date_1} - {date_2}")

    # --- Same-day case ---
    if days == 0:
        doy = start_date.timetuple().tm_yday
        median = calculate_median(doy, params_lower, params_upper)
        ndvi_val = start_delta + median
        ndvi_scaled = np.clip(ndvi_val * 10000, 0, 10000).astype(np.int16)
        idx = (start_date - base_date).days
        ndvi_zarr[idx, pixel_idx] = ndvi_scaled
        return

    # --- Interpolate linearly across the interval ---
    L1_deltas = np.linspace(start_delta, end_delta, num=days + 1)

    doy_start = start_date.timetuple().tm_yday
    doy_end = end_date.timetuple().tm_yday
    days_diff = (end_date - start_date).days
    
    if doy_end < doy_start:
        doys = np.linspace(doy_start, doy_end + 365, num=days_diff + 1) % 365
        doys = np.where(doys == 0, 365, doys)
    else:
        doys = np.linspace(doy_start, doy_end, num=days_diff + 1)

    doys = np.where((doys == 0) | (doys == 366), 365, doys)

    medians = calculate_median(doys, params_lower, params_upper)

    ndvi = L1_deltas + medians
    ndvi_scaled = np.clip(ndvi * 10000, 0, 10000).astype(np.int16)

    # --- Write interpolated values to NDVI array ---
    idx_start = (start_date - base_date).days
    idx_end = (end_date - base_date).days
    #median_to_write = np.clip(medians * 10000,  0, 10000).astype(np.int16) 
    ndvi_zarr[idx_start:idx_end + 1, pixel_idx] = ndvi_scaled #ndvi_scaled


def L2_smoothing(pixel_idx,init_position, params_lower, params_upper):

    # --- Load and prepare recent date history ---
    last_dates_bytes = last_dates_zarr[:7, pixel_idx]
    last_dates = [zarr_date_to_date(d) for d in last_dates_bytes if not np.all(d == b'1900-01-01')]


    # --- Extract NDVI and median values for those dates ---
    ndvi_vals = []
    median_vals = []

    base_date = zarr_date_to_date(dates[0])
    for d in last_dates:
        idx = (d - base_date).days
        ndvi_val = ndvi_zarr[idx, pixel_idx] / 10000.0
        median_val = calculate_median(d.timetuple().tm_yday, params_lower, params_upper)
        ndvi_vals.append(ndvi_val)
        median_vals.append(median_val)

    ndvi_vals = np.array(ndvi_vals)
    median_vals = np.array(median_vals)

    deltas_arr = ndvi_vals - median_vals

    # --- prick teh second or third and fourth value ---
    start_date = last_dates[init_position]
    end_date = last_dates[3]
    days_diff = (end_date - start_date).days

    doy_start = start_date.timetuple().tm_yday
    doy_end = end_date.timetuple().tm_yday

    if doy_end < doy_start:
        doys = np.linspace(doy_start, doy_end + 365, num=days_diff + 1) % 365
        doys = np.where(doys == 0, 365, doys)
    else:
        doys = np.linspace(doy_start, doy_end, num=days_diff + 1)

    doys = np.where((doys == 0) | (doys == 366), 365, doys)


    idx = np.arange(len(ndvi_vals))

    # --- Fire/storm case: smooth directly NDVI values ---
    if np.sum(deltas_arr < -0.2) >= 5:
        
        loess = sm.nonparametric.lowess(ndvi_vals, idx, frac=1, it=5, return_sorted=True)
        smoothed = np.interp(np.linspace(0, len(ndvi_vals)-1, days_diff+1), loess[:, 0], loess[:, 1])

    else:
        # Normal case: smooth deltas

        loess = sm.nonparametric.lowess(deltas_arr, idx, frac=1, it=5, return_sorted=True)
        smoothed_deltas = np.interp(np.linspace(0, len(deltas_arr)-1, days_diff+1), loess[:, 0], loess[:, 1])
        medians = calculate_median(doys, params_lower, params_upper)
        
        # check if the std is high

        std_y = np.std(smoothed_deltas)

        if std_y > 0.015:

            if std_y <= 0.03:
                window = 3
            else:
                window = 5

            smoothed_deltas = pd.Series(smoothed_deltas).rolling(window=window, center=True, min_periods=1).mean().values

        smoothed = smoothed_deltas + medians

    # avoid spiking

    for i in range(1,len(smoothed)):

        if ((((smoothed[i-1] - smoothed[i]) > 0.2) or ((smoothed[i-1] - smoothed[i]) < -0.2)) and 
            (((smoothed[i] - smoothed[i +1]) > 0.2) or ((smoothed[i] - smoothed[i +1]) < -0.2)) ):
            
            smoothed[i] = 0.5 * (smoothed[i-1] + smoothed[i+1])


    # --- Write smoothed NDVI values at daily scale ---
    base_idx_start = (start_date - base_date).days
    base_idx_end = (end_date - base_date).days
    smoothed_scaled = np.clip(smoothed * 10000, 0, 10000).astype(np.int16)
    ndvi_zarr[base_idx_start:base_idx_end+1, pixel_idx] = smoothed_scaled


# --- Main function ---
def continous_ndvi(day, pixel_idx):

    base_date = zarr_date_to_date(dates[0])
    date_index = (day - base_date).days
    ndvi_val = ndvi_zarr[date_index, pixel_idx]

    # Read last date from Zarr
    last_date_raw = last_dates_zarr[6, pixel_idx]
    last_date = zarr_date_to_date(last_date_raw)

    potential_date_raw = last_dates_zarr[7, pixel_idx]
    potential_date = zarr_date_to_date(potential_date_raw)

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

        # estimate ndvi using potential decay
        # Write last date back to Zarr
        if last_date !=  date(1900, 1, 1):
            estimation = estimate_ndvi(days_diff, current_median, delta_prev)
            ndvi_zarr[date_index, pixel_idx] = np.int16(estimation * 10000)

    else:

        t = torch.as_tensor(np.atleast_1d(doy) / 365.0, dtype=torch.float32)

        if params_lower.ndim == 1:
            params_lower = params_lower[None, :]
        if params_upper.ndim == 1:
            params_upper = params_upper[None, :]

        lower = double_logistic_function(t, params_lower).squeeze().numpy()
        upper = double_logistic_function(t, params_upper).squeeze().numpy()

        if last_date !=  date(1900, 1, 1):

            true_value = outlier_detection(ndvi_val, lower, upper,current_delta,delta_prev)
        
        else: 
            # needs an initial condition to start the ingestion
            true_value = True

        if true_value:

            if potential_date !=  date(1900, 1, 1):

                pd_idx = (potential_date - base_date).days
                potential_ndvi = ndvi_zarr[pd_idx, pixel_idx]
                potential_ndvi = potential_ndvi / 10000.0

                accepted = retroactive_outlier_detection(potential_date,potential_ndvi,ndvi_val,current_median,
                                                        params_lower,params_upper,pixel_idx)

                if accepted:

                    # perform 2 L1 gapfilling, from pot_date to last_date and from last_date to current date
                    # perform L2 smoothing from postion 1 to position 3

                    pdoy = potential_date.timetuple().tm_yday
                    potential_median = calculate_median(pdoy, params_lower, params_upper)

                    potential_delta = potential_ndvi - potential_median

                    L1_interpolation(potential_delta, current_delta, potential_date, day,base_date, params_lower, params_upper,pixel_idx)
                    L1_interpolation(potential_delta, delta_prev, potential_date, last_date ,base_date, params_lower, params_upper,pixel_idx)

                    # add the potential outlier confirmed in 6th postion and last date in 7th postion
                    # perform the smoothing between second position and fourth position, we do that because we add two element to the array not one

                    old_dates = last_dates_zarr[:7, pixel_idx].copy()
                    # force a fixed-width bytes array to avoid vlen/object promotion
                    old_dates = np.array(old_dates, dtype='S10')   # ensures dtype '|S10'
                    shifted = np.empty(old_dates.shape, dtype='S10')
                    shifted[:-2] = old_dates[1:-1]
                    shifted[-2] = potential_date.strftime("%Y-%m-%d").encode("utf-8")
                    shifted[-1] = day.strftime("%Y-%m-%d").encode("utf-8")
                    # Write to Zarr
                    last_dates_zarr[:7, pixel_idx] = shifted


                    valid_window = [zarr_date_to_date(d) for d in shifted]
                    """if all(d != date(1900, 1, 1) for d in valid_window):
                        L2_smoothing(pixel_idx,1, params_lower, params_upper)"""

                    # --- Clear potential outlier slot (index 7) ---
                    last_dates_zarr[7:, pixel_idx] = b"1900-01-01"


                else:

                    # perform 1 L1 gapfilling, from last_date to current date
                    # perform L2 smoothing from postion 2 to position 3

                    if last_date !=  date(1900, 1, 1):

                        L1_interpolation(delta_prev, current_delta, last_date, day,base_date, params_lower, params_upper,pixel_idx)
                    
                    # update the smoothed window dates
                    old_dates = last_dates_zarr[:7, pixel_idx].copy()
                    old_dates = np.array(old_dates, dtype='S10')
                    shifted = np.empty(old_dates.shape, dtype='S10')
                    shifted[:-1] = old_dates[1:]
                    shifted[-1] = day.strftime("%Y-%m-%d").encode("utf-8")

                    last_dates_zarr[:7, pixel_idx] = shifted


                    valid_window = [zarr_date_to_date(d) for d in shifted]
                    if all(d != date(1900, 1, 1) for d in valid_window):
                        L2_smoothing(pixel_idx,2, params_lower, params_upper)

                    # remove outlier detection
                    date_to_potential = b"1900-01-01"
                    last_dates_zarr[7:, pixel_idx] = date_to_potential

            else:

                if last_date !=  date(1900, 1, 1):

                    L1_interpolation(delta_prev, current_delta, last_date, day,base_date, params_lower, params_upper,pixel_idx)
                
                # update the smoothed window dates
                old_dates = last_dates_zarr[:7, pixel_idx].copy()
                old_dates = np.array(old_dates, dtype='S10')
                shifted = np.empty(old_dates.shape, dtype='S10')
                shifted[:-1] = old_dates[1:]
                shifted[-1] = day.strftime("%Y-%m-%d").encode("utf-8")

                last_dates_zarr[:7, pixel_idx] = shifted


                valid_window = [zarr_date_to_date(d) for d in shifted]
                if all(d != date(1900, 1, 1) for d in valid_window):
                    L2_smoothing(pixel_idx, 2, params_lower, params_upper)

            # remove outlier detection
            date_to_potential = b"1900-01-01"
            last_dates_zarr[7:, pixel_idx] = date_to_potential
        
        else:

            date_to_potential = day.strftime("%Y-%m-%d").encode("utf-8")
            last_dates_zarr[7:, pixel_idx] = date_to_potential




pixels = np.random.choice(np.arange(1_000_000), size=5, replace=False) # np.array([42,5645,129483,248738,582910,302]) # 


def process_pixel(pixel_idx):

    for i in range(1500):
        day = zarr_date_to_date(dates[i])
        continous_ndvi(pixel_idx, day)
        print(i)

# Run each pixel in parallel 
with concurrent.futures.ProcessThreadExecutor() as executor:
    executor.map(process_pixel, pixels)


"""for i in range(1500):
    day = zarr_date_to_date(dates[i])
    print(i)
    #continous_ndvi_2(day, pixels[0])

    for pixel in pixels:
        continous_ndvi(day, pixel)"""

print("done")

###------------######

# this part is needed only to show the NDVI series
import matplotlib.pyplot as plt

# Reopen the NDVI zarr group
root = zarr.open("/data_2/scratch/francesco/zarr_demo_daily_output", mode="r")
original_root = zarr.open("/data_2/scratch/francesco/zarr_demo_daily/", mode="r")

# Access the actual NDVI array
ndvi_arr = root["ndvi"]
observed_ndvi = original_root["ndvi"]

# Extract NDVI time series
base_date = zarr_date_to_date(dates[0])
date_list = [zarr_date_to_date(d) for d in dates[:1500]]

import matplotlib
matplotlib.use("Agg")   

import matplotlib.pyplot as plt
import os

out_dir = "/home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/figure"
os.makedirs(out_dir, exist_ok=True)  

ndvi_series = ndvi_arr[:1500, pixels[0]] / 10000.0

for pixel in pixels:
    ndvi_series = ndvi_arr[:1500, pixel] / 10000.0

    plt.figure(figsize=(10, 5))
    plt.plot(date_list, ndvi_series, lw=1, label="NDVI") 
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series for Pixel {pixel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"output Pixel_{pixel}.png")  
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close() 
    print(f"Saved {filename}")

    original_ndvi =  observed_ndvi[:1500, pixel] / 10000.0
    original_ndvi[(original_ndvi < 0) | (original_ndvi > 1)] = np.nan

    dates = pd.to_datetime([d.decode("utf-8") for d in root["dates"][:]])

    params = root["params"]
    params_lower = params["params_lower"]
    params_upper = params["params_upper"]

    T_SCALE = 1.0 / 365.0
    doy = dates.dayofyear
    t = torch.tensor(doy * T_SCALE, dtype=torch.float32)

    order = np.argsort(dates)
    dates_sorted = np.array(dates)[order]
    t_sorted = t[order]

    lower = double_logistic_function(t_sorted, params_lower[[pixel]]).squeeze().numpy()
    upper = double_logistic_function(t_sorted, params_upper[[pixel]]).squeeze().numpy()


    plt.figure(figsize=(10, 5))
    plt.plot(date_list, ndvi_series, lw=1, label="NDVI",color = "black") 
    plt.plot(dates_sorted, lower, color="tab:red", lw=1.5)
    plt.plot(dates_sorted, upper, color="tab:green", lw=1.5)
    plt.fill_between(dates_sorted, lower, upper, color="tab:red", alpha=0.1)
    plt.scatter(date_list,original_ndvi, label = "observed NDVI", color = "black")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series for Pixel {pixel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"Bands and observed NDVI Pixel_{pixel}.png") 
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close() 
    print(f"Saved {filename}")
