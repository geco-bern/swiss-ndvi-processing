# rm -rf /data_2/scratch/francesco/zarr_demo_daily_processed_2/*
# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/03_test_parallel.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_parallel_continous_ndvi_small.log &


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

# =====================================================
# CONFIGURATION
# =====================================================
INPUT_DIR = "/data_2/scratch/francesco/zarr_demo_daily/"
OUTPUT_DIR = "/data_2/scratch/francesco/zarr_demo_daily_processed_2/"
N_FILES = 2
N_PIXELS_PER_FILE = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# UTILS
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
# NDVI MODEL FUNCTIONS
# =====================================================
def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def calculate_median(doy, params_lower, params_upper):
    t = torch.as_tensor(np.atleast_1d(doy) / 365.0, dtype=torch.float32)
    if params_lower.ndim == 1:
        params_lower = params_lower[None, :]
    if params_upper.ndim == 1:
        params_upper = params_upper[None, :]
    lower = double_logistic_function(t, params_lower).squeeze().numpy()
    upper = double_logistic_function(t, params_upper).squeeze().numpy()
    return 0.5 * (upper + lower)


def estimate_ndvi(days_diff, median, delta_prev):
    decrease_factor = math.exp(-math.log(2) * (days_diff / 15))
    return median + delta_prev * decrease_factor


def outlier_detection(obs, lower, upper, delta_current, delta_previous):
    inside_band = (obs >= lower) and (obs <= upper)
    delta_delta = delta_current - delta_previous
    if inside_band:
        return True
    if ((delta_current > 0.05) or (delta_current < -0.05)) and ((delta_delta > 0.1) or (delta_delta < -0.1)):
        return False
    return True


def retroactive_outlier_detection(potential_date, potential_ndvi, obs, current_median, params_lower, params_upper):
    pdoy = potential_date.timetuple().tm_yday
    potential_median = calculate_median(pdoy, params_lower, params_upper)
    delta_delta = ((obs - current_median) - (potential_ndvi - potential_median))
    return (delta_delta < 0.1) and (delta_delta > -0.1)


def L1_interpolation(delta_1, delta_2, date_1, date_2, base_date, params_lower, params_upper, pixel_idx, ndvi_zarr):
    if date_1 <= date_2:
        start_date, end_date = date_1, date_2
        start_delta, end_delta = delta_1, delta_2
    else:
        start_date, end_date = date_2, date_1
        start_delta, end_delta = delta_2, delta_1

    days = (end_date - start_date).days
    if days == 0:
        doy = start_date.timetuple().tm_yday
        median = calculate_median(doy, params_lower, params_upper)
        ndvi_val = start_delta + median
        ndvi_scaled = np.clip(ndvi_val * 10000, 0, 10000).astype(np.int16)
        idx = (start_date - base_date).days
        ndvi_zarr[idx, pixel_idx] = ndvi_scaled
        return

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
    idx_start = (start_date - base_date).days
    idx_end = (end_date - base_date).days
    ndvi_zarr[idx_start:idx_end + 1, pixel_idx] = ndvi_scaled

def L2_smoothing(pixel_idx, init_position, params_lower, params_upper):
            # load last dates (first 7 positions)
            last_dates_bytes = last_dates_zarr[:7, pixel_idx]
            last_dates = [zarr_date_to_date(d) for d in last_dates_bytes if not np.all(d == b'1900-01-01')]
            if len(last_dates) < 2:
                return

            ndvi_vals = []
            median_vals = []
            base_date = dates_list[0]
            for d in last_dates:
                idx = (d - base_date).days
                if 0 <= idx < ndvi_zarr.shape[0]:
                    ndvi_val = ndvi_zarr[idx, pixel_idx] / 10000.0
                else:
                    ndvi_val = np.nan
                median_val = calculate_median(d.timetuple().tm_yday, params_lower, params_upper)
                ndvi_vals.append(ndvi_val)
                median_vals.append(median_val)

            ndvi_vals = np.array(ndvi_vals)
            median_vals = np.array(median_vals)
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
                medians = calculate_median(doys, params_lower, params_upper)
                std_y = np.std(smoothed_deltas)
                if std_y > 0.015:
                    window = 3 if std_y <= 0.03 else 5
                    smoothed_deltas = pd.Series(smoothed_deltas).rolling(window=window, center=True, min_periods=1).mean().values
                smoothed = smoothed_deltas + medians

            # avoid spikes
            for i in range(1, len(smoothed)-1):
                if (abs(smoothed[i-1] - smoothed[i]) > 0.2) and (abs(smoothed[i] - smoothed[i+1]) > 0.2):
                    smoothed[i] = 0.5 * (smoothed[i-1] + smoothed[i+1])

            base_idx_start = (start_date - dates_list[0]).days
            base_idx_end = (end_date - dates_list[0]).days
            base_idx_start = max(0, base_idx_start)
            base_idx_end = min(ndvi_zarr.shape[0]-1, base_idx_end)
            smoothed_scaled = np.clip(smoothed * 10000, 0, 10000).astype(np.int16)
            ndvi_zarr[base_idx_start:base_idx_end+1, pixel_idx] = smoothed_scaled[: base_idx_end - base_idx_start + 1]

# =====================================================
# MAIN NDVI FUNCTION (UNCHANGED)
# =====================================================
def continous_ndvi(day_date, pixel_idx):
    base_date = dates_list[0]
    date_index = (day_date - base_date).days
    if not (0 <= date_index < ndvi_zarr.shape[0]):
        return

    ndvi_val_raw = ndvi_zarr[date_index, pixel_idx]
    try:
        last_date_raw = last_dates_zarr[6, pixel_idx]
    except Exception:
        last_date_raw = b"1900-01-01"
    last_date = zarr_date_to_date(last_date_raw) if not np.all(last_date_raw == b"1900-01-01") else date(1900,1,1)

    potential_date_raw = last_dates_zarr[7, pixel_idx]
    potential_date = zarr_date_to_date(potential_date_raw) if not np.all(potential_date_raw == b"1900-01-01") else date(1900,1,1)

    params_lower = params_lower_zarr[pixel_idx]
    params_upper = params_upper_zarr[pixel_idx]

    doy = day_date.timetuple().tm_yday

    if last_date != date(1900,1,1):
        last_doy = last_date.timetuple().tm_yday
        days_diff = (day_date - last_date).days
        last_idx = (last_date - base_date).days
        last_ndvi = ndvi_zarr[last_idx, pixel_idx] / 10000.0
        delta_prev = last_ndvi - calculate_median(last_doy, params_lower, params_upper)
    else:
        delta_prev = None

    ndvi_val = ndvi_val_raw / 10000.0 if ndvi_val_raw is not None else np.nan

    current_median = calculate_median(doy, params_lower, params_upper)
    current_delta = ndvi_val - current_median if not np.isnan(ndvi_val) else None

    if ndvi_val >= 1 or ndvi_val <= 0 or np.isnan(ndvi_val):
        if last_date != date(1900,1,1) and delta_prev is not None:
            estimation = estimate_ndvi((day_date - last_date).days, current_median, delta_prev)
            if 0 <= estimation <= 1:
                ndvi_zarr[date_index, pixel_idx] = np.int16(np.clip(estimation * 10000, 0, 10000))
        return

    if last_date != date(1900,1,1) and delta_prev is not None:
        true_value = outlier_detection(ndvi_val, current_median - 0.1, current_median + 0.1, current_delta, delta_prev)
    else:
        true_value = True

    if true_value:

            if potential_date !=  date(1900, 1, 1):

                pd_idx = (potential_date - base_date).days
                potential_ndvi = ndvi_zarr[pd_idx, pixel_idx]
                potential_ndvi = potential_ndvi / 10000.0

                accepted = retroactive_outlier_detection(potential_date,potential_ndvi,ndvi_val,current_median,
                                                        params_lower,params_upper)

                if accepted:

                    # perform 2 L1 gapfilling, from pot_date to last_date and from last_date to current date
                    # perform L2 smoothing from postion 1 to position 3

                    pdoy = potential_date.timetuple().tm_yday
                    potential_median = calculate_median(pdoy, params_lower, params_upper)

                    potential_delta = potential_ndvi - potential_median

                    L1_interpolation(potential_delta, current_delta, potential_date, day_date,base_date, params_lower, params_upper,pixel_idx, ndvi_zarr)
                    L1_interpolation(potential_delta, delta_prev, potential_date, last_date ,base_date, params_lower, params_upper,pixel_idx, ndvi_zarr)

                    # add the potential outlier confirmed in 6th postion and last date in 7th postion
                    # perform the smoothing between second position and fourth position, we do that because we add two element to the array not one

                    old_dates = last_dates_zarr[:7, pixel_idx].copy()
                    # force a fixed-width bytes array to avoid vlen/object promotion
                    old_dates = np.array(old_dates, dtype='S10')   # ensures dtype '|S10'
                    shifted = np.empty(old_dates.shape, dtype='S10')
                    shifted[:-2] = old_dates[1:-1]
                    shifted[-2] = potential_date.strftime("%Y-%m-%d").encode("utf-8")
                    shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                    # Write to Zarr
                    last_dates_zarr[:7, pixel_idx] = shifted


                    valid_window = [zarr_date_to_date(d) for d in shifted]
                    if all(d != date(1900, 1, 1) for d in valid_window):
                        L2_smoothing(pixel_idx,1, params_lower, params_upper)

                    # --- Clear potential outlier slot (index 7) ---
                    last_dates_zarr[7:, pixel_idx] = b"1900-01-01"


                else:

                    # perform 1 L1 gapfilling, from last_date to current date
                    # perform L2 smoothing from postion 2 to position 3

                    if last_date !=  date(1900, 1, 1):

                        L1_interpolation(delta_prev, current_delta, last_date, day_date,base_date, params_lower, params_upper,pixel_idx, ndvi_zarr)
                    
                    # update the smoothed window dates
                    old_dates = last_dates_zarr[:7, pixel_idx].copy()
                    old_dates = np.array(old_dates, dtype='S10')
                    shifted = np.empty(old_dates.shape, dtype='S10')
                    shifted[:-1] = old_dates[1:]
                    shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")

                    last_dates_zarr[:7, pixel_idx] = shifted


                    valid_window = [zarr_date_to_date(d) for d in shifted]
                    if all(d != date(1900, 1, 1) for d in valid_window):
                        L2_smoothing(pixel_idx,2, params_lower, params_upper)

                    # remove outlier detection
                    date_to_potential = b"1900-01-01"
                    last_dates_zarr[7:, pixel_idx] = date_to_potential

            else:

                if last_date !=  date(1900, 1, 1):

                    L1_interpolation(delta_prev, current_delta, last_date, day_date,base_date, params_lower, params_upper,pixel_idx, ndvi_zarr)
                
                # update the smoothed window dates
                old_dates = last_dates_zarr[:7, pixel_idx].copy()
                old_dates = np.array(old_dates, dtype='S10')
                shifted = np.empty(old_dates.shape, dtype='S10')
                shifted[:-1] = old_dates[1:]
                shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")

                last_dates_zarr[:7, pixel_idx] = shifted


                valid_window = [zarr_date_to_date(d) for d in shifted]
                if all(d != date(1900, 1, 1) for d in valid_window):
                    L2_smoothing(pixel_idx, 2, params_lower, params_upper)

            # remove outlier detection
            date_to_potential = b"1900-01-01"
            last_dates_zarr[7:, pixel_idx] = date_to_potential
        
    else:

            date_to_potential = day_date.strftime("%Y-%m-%d").encode("utf-8")
            last_dates_zarr[7:, pixel_idx] = date_to_potential




# =====================================================
# PROCESS SINGLE FILE
# =====================================================

def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        final_out_path = os.path.join(OUTPUT_DIR, file_name)
        tmp_out_path = final_out_path + f".tmp_{os.getpid()}"

        # Clean up any previous temp dir for this process
        if os.path.exists(tmp_out_path):
            shutil.rmtree(tmp_out_path)

        # Make a private copy for this process
        shutil.copytree(file_path, tmp_out_path)

        # --- open and process BEFORE renaming ---
        ds = zarr.open_group(tmp_out_path, mode="r+")

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
        selected_pixels = rng.choice(np.arange(n_pixels), size=N_PIXELS_PER_FILE, replace=False)
        print(f"[{file_name}] Selected {len(selected_pixels)} random pixels")
        print(selected_pixels)

        for day_date in dates_list[:1000]:
            for pixel_idx in selected_pixels:
                continous_ndvi(day_date, pixel_idx)

        # (no flush, no close needed for Zarr v3)
        zarr.consolidate_metadata(tmp_out_path)

        # --- atomic finalization ---
        if not os.path.exists(final_out_path):
            os.rename(tmp_out_path, final_out_path)
            print(f"[{file_name}] ✅ Saved to {final_out_path}")
        else:
            print(f"[{file_name}] ⚠️ Output already exists — removing temp.")
            shutil.rmtree(tmp_out_path)

        return {"file": file_name, "status": "ok"}

    except Exception as e:
        traceback.print_exc()
        # cleanup on error
        if os.path.exists(tmp_out_path):
            shutil.rmtree(tmp_out_path)
        return {"file": file_path, "status": "error", "error": str(e)}

# =====================================================
# RUN PARALLEL
# =====================================================
if __name__ == "__main__":
    all_entries = sorted(os.listdir(INPUT_DIR))
    zarr_files = [os.path.join(INPUT_DIR, e) for e in all_entries if e.endswith(".zarr")]
    zarr_files = zarr_files[:N_FILES]
    print(f"Starting analysis on {len(zarr_files)} files with {N_PIXELS_PER_FILE} pixels each.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_FILES) as executor:
        futures = {executor.submit(process_file, path): path for path in zarr_files}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            print("Result:", res)

    print("✅ All processing complete.")
