# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/02_parallel_by_file.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_parallel_by_file.log &

import os
import numpy as np
import zarr
import pandas as pd
import math
import torch
import statsmodels.api as sm
import re
import concurrent.futures
import hashlib
from datetime import datetime, date
import gc
import traceback

# ---------------------------
# CONFIG
# ---------------------------
INPUT_DIR = "/data_2/scratch/francesco/zarr_demo_daily/"   # directory containing ~1000 .zarr files
N_FILES = 2           # take first 20 files
N_PIXELS_PER_FILE = 5  # pick 100 random pixels from each file
MAX_DAYS = 1000        # set to None to process all dates available in each zarr file, 
                       # or set e.g. 1500 to only loop first 1500 days (as in your previous script)

# ---------------------------
# Utilities
# ---------------------------
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

# ---------------------------
# Discover first 20 zarr files
# ---------------------------
all_entries = sorted(os.listdir(INPUT_DIR))
zarr_files = [os.path.join(INPUT_DIR, e) for e in all_entries if e.endswith(".zarr")]
if len(zarr_files) < N_FILES:
    raise RuntimeError(f"Found only {len(zarr_files)} .zarr files in {INPUT_DIR}, need at least {N_FILES}")

zarr_files = zarr_files[:N_FILES]
print(f"Selected {len(zarr_files)} files for parallel analysis (first {N_FILES}).")

# ---------------------------
# Worker: process a single file (one worker per file)
# ---------------------------
def process_file(file_path):
    """
    Process one zarr file: select N_PIXELS_PER_FILE random pixels (seeded by filename),
    then run your NDVI analysis for those pixels, writing results back to the same file.
    """
    try:
        file_name = os.path.basename(file_path)
        print(f"[{file_name}] Worker start.")

        # Open zarr in read+write mode (one worker per file, safe)
        ds = zarr.open_group(file_path, mode="r+")

        # Get arrays (these are array-like objects that support slicing and assignment)
        ndvi_zarr = ds["ndvi"]
        dates_zarr = ds["dates"]
        params_lower_zarr = ds["params"]["params_lower"]
        params_upper_zarr = ds["params"]["params_upper"]
        last_dates_zarr = ds["last_dates"]

        # Build date list (datetime.date)
        dates_list = [zarr_date_to_date(d) for d in dates_zarr[:]]
        n_dates = len(dates_list)
        days_to_process = n_dates if MAX_DAYS is None else min(MAX_DAYS, n_dates)

        # Number of pixels in this zarr
        n_pixels_file = ndvi_zarr.shape[1]
        if n_pixels_file < N_PIXELS_PER_FILE:
            raise RuntimeError(f"[{file_name}] File has only {n_pixels_file} pixels, requested {N_PIXELS_PER_FILE}")

        # Choose 100 pixels at random but reproducible per file using filename hash
        seed = int(hashlib.md5(file_name.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        selected_pixels = rng.choice(np.arange(n_pixels_file), size=N_PIXELS_PER_FILE, replace=False)
        selected_pixels = selected_pixels.tolist()
        print(f"[{file_name}] Selected {len(selected_pixels)} pixels (seed={seed}).")

        # ---------------------------
        # Define algorithm functions that close over ndvi_zarr, params_*, last_dates_zarr, dates_list
        # These are direct translations/adaptations of your original functions but using local arrays.
        # ---------------------------

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

        def L1_interpolation(delta_1, delta_2, date_1, date_2, base_date, params_lower, params_upper, pixel_idx):
            # Ensure chronological order
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
                median = calculate_median(doy, params_lower, params_upper)
                ndvi_val = start_delta + median
                ndvi_scaled = np.clip(ndvi_val * 10000, 0, 10000).astype(np.int16)
                idx = (start_date - base_date).days
                if 0 <= idx < ndvi_zarr.shape[0]:
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
            idx_start = max(0, idx_start)
            idx_end = min(ndvi_zarr.shape[0] - 1, idx_end)
            ndvi_zarr[idx_start:idx_end + 1, pixel_idx] = ndvi_scaled[: idx_end - idx_start + 1]

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

        # continous_ndvi adapted to local arrays
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
                # bad value or missing -> try estimate
                if last_date != date(1900,1,1) and delta_prev is not None:
                    estimation = estimate_ndvi((day_date - last_date).days, current_median, delta_prev)
                    if 0 <= estimation <= 1:
                        ndvi_zarr[date_index, pixel_idx] = np.int16(np.clip(estimation * 10000, 0, 10000))
                return

            # normal observed
            if last_date != date(1900,1,1) and delta_prev is not None:
                true_value = outlier_detection(ndvi_val, current_median - 0.1, current_median + 0.1, current_delta, delta_prev) # placeholders for lower/upper band
            else:
                true_value = True

            if true_value:
                if potential_date != date(1900,1,1):
                    pd_idx = (potential_date - base_date).days
                    if 0 <= pd_idx < ndvi_zarr.shape[0]:
                        potential_ndvi = ndvi_zarr[pd_idx, pixel_idx] / 10000.0
                    else:
                        potential_ndvi = np.nan

                    accepted = retroactive_outlier_detection(potential_date, potential_ndvi, ndvi_val, current_median, params_lower, params_upper)

                    if accepted:
                        potential_median = calculate_median(potential_date.timetuple().tm_yday, params_lower, params_upper)
                        potential_delta = potential_ndvi - potential_median
                        if delta_prev is None:
                            delta_prev = potential_delta
                        L1_interpolation(potential_delta, current_delta, potential_date, day_date, base_date, params_lower, params_upper, pixel_idx)
                        if last_date != date(1900,1,1):
                            L1_interpolation(potential_delta, delta_prev, potential_date, last_date, base_date, params_lower, params_upper, pixel_idx)

                        # shift last_dates (insert potential and current)
                        try:
                            old_dates = last_dates_zarr[:7, pixel_idx].copy()
                            old_dates = np.array(old_dates, dtype='S10')
                            shifted = np.empty(old_dates.shape, dtype='S10')
                            shifted[:-2] = old_dates[1:-1]
                            shifted[-2] = potential_date.strftime("%Y-%m-%d").encode("utf-8")
                            shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                            last_dates_zarr[:7, pixel_idx] = shifted
                            last_dates_zarr[7:, pixel_idx] = b"1900-01-01"
                        except Exception:
                            pass

                    else:
                        # non-accepted potential: do single interpolation and update last_dates
                        if last_date != date(1900,1,1) and delta_prev is not None:
                            L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, pixel_idx)
                        try:
                            old_dates = last_dates_zarr[:7, pixel_idx].copy()
                            old_dates = np.array(old_dates, dtype='S10')
                            shifted = np.empty(old_dates.shape, dtype='S10')
                            shifted[:-1] = old_dates[1:]
                            shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                            last_dates_zarr[:7, pixel_idx] = shifted
                            if all(d != b'1900-01-01' for d in shifted):
                                L2_smoothing(pixel_idx, 2, params_lower, params_upper)
                            last_dates_zarr[7:, pixel_idx] = b"1900-01-01"
                        except Exception:
                            pass

                else:
                    # no potential date, just interpolate or add to window
                    if last_date != date(1900,1,1) and delta_prev is not None:
                        L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, pixel_idx)
                    try:
                        old_dates = last_dates_zarr[:7, pixel_idx].copy()
                        old_dates = np.array(old_dates, dtype='S10')
                        shifted = np.empty(old_dates.shape, dtype='S10')
                        shifted[:-1] = old_dates[1:]
                        shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                        last_dates_zarr[:7, pixel_idx] = shifted
                        if all(d != b'1900-01-01' for d in shifted):
                            L2_smoothing(pixel_idx, 2, params_lower, params_upper)
                        last_dates_zarr[7:, pixel_idx] = b"1900-01-01"
                    except Exception:
                        pass

            else:
                # mark potential outlier
                last_dates_zarr[7:, pixel_idx] = day_date.strftime("%Y-%m-%d").encode("utf-8")

        # ---------------------------
        # Run the analysis loop: iterate days (outer) and selected pixels (inner)
        # This follows your previous pattern.
        # ---------------------------
        base_date = dates_list[0]
        for di in range(days_to_process):
            day_date = dates_list[di]
            # iterate the selected pixels
            for local_pixel in selected_pixels:
                continous_ndvi(day_date, local_pixel)
            if di % 250 == 0:
                print(f"[{file_name}] processed day {di}/{days_to_process}")

        # Flush & finalize
        gc.collect()
        print(f"[{file_name}] Worker finished successfully.")
        return {"file": file_name, "status": "ok", "processed_pixels": len(selected_pixels), "days": days_to_process}

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{file_path}] Worker failed: {e}\n{tb}")
        return {"file": file_path, "status": "error", "error": str(e)}

# ---------------------------
# Launch parallel processing of the 20 files
# ---------------------------
if __name__ == "__main__":
    print("Starting parallel processing of first 20 zarr files...")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_FILES) as exec:
        futures = {exec.submit(process_file, p): p for p in zarr_files}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            print("Result:", res)

    print("All workers completed.")
