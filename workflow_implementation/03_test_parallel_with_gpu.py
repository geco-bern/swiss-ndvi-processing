# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/03_test_parallel_with_gpu.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_parallel_continous_ndvi_gpu_ssd.log &

#!/usr/bin/env python3
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

INPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily/"
OUTPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_processed/"
N_FILES = 15
N_PIXELS_PER_FILE = 1
PROCESS_DATES_LIMIT = 3072

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    ndvi_arr[a:b + 1, pixel_rel_idx] = ndvi_scaled[(a - idx_start):(b - idx_start) + 1]

def L2_smoothing(pixel_rel_idx, init_position, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device):
    last_dates_bytes = last_dates_arr[:7, pixel_rel_idx]
    last_dates = [zarr_date_to_date(d) for d in last_dates_bytes if not np.all(d == b'1900-01-01')]
    if len(last_dates) < 2:
        return
    ndvi_vals = []
    median_vals = []
    base_date = dates_list[0]
    for d in last_dates:
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
    ndvi_arr[base_idx_start:base_idx_end + 1, pixel_rel_idx] = smoothed_scaled[: base_idx_end - base_idx_start + 1]

def continous_ndvi(day_date, pixel_rel_idx, pixel_global_idx, ndvi_arr, last_dates_arr, params_lower_arr, params_upper_arr, dates_list, device, timing):
    t0 = time.perf_counter()
    base_date = dates_list[0]
    date_index = (day_date - base_date).days
    if not (0 <= date_index < ndvi_arr.shape[0]):
        return
    timing['calls'] += 1
    ndvi_val_raw = ndvi_arr[date_index, pixel_rel_idx]
    last_date_raw = last_dates_arr[6, pixel_rel_idx]
    last_date = zarr_date_to_date(last_date_raw) if not np.all(last_date_raw == b"1900-01-01") else date(1900,1,1)
    potential_date_raw = last_dates_arr[7, pixel_rel_idx]
    potential_date = zarr_date_to_date(potential_date_raw) if not np.all(potential_date_raw == b"1900-01-01") else date(1900,1,1)
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
            if 0 <= estimation <= 1:
                ndvi_arr[date_index, pixel_rel_idx] = np.int16(np.clip(int(round(estimation * 10000.0)), 0, 10000))
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
                old_dates = np.array(old_dates, dtype='S10')
                shifted = np.empty(old_dates.shape, dtype='S10')
                shifted[:-2] = old_dates[1:-1]
                shifted[-2] = potential_date.strftime("%Y-%m-%d").encode("utf-8")
                shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                last_dates_arr[:7, pixel_rel_idx] = shifted
                valid_window = [zarr_date_to_date(d) for d in shifted]
                if all(d != date(1900,1,1) for d in valid_window):
                    L2_smoothing(pixel_rel_idx, 1, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
                last_dates_arr[7:, pixel_rel_idx] = b"1900-01-01"
            else:
                if last_date != date(1900,1,1):
                    L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device)
                old_dates = last_dates_arr[:7, pixel_rel_idx].copy()
                old_dates = np.array(old_dates, dtype='S10')
                shifted = np.empty(old_dates.shape, dtype='S10')
                shifted[:-1] = old_dates[1:]
                shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
                last_dates_arr[:7, pixel_rel_idx] = shifted
                valid_window = [zarr_date_to_date(d) for d in shifted]
                if all(d != date(1900,1,1) for d in valid_window):
                    L2_smoothing(pixel_rel_idx, 2, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
                last_dates_arr[7:, pixel_rel_idx] = b"1900-01-01"
        else:
            if last_date != date(1900,1,1):
                L1_interpolation(delta_prev, current_delta, last_date, day_date, base_date, params_lower, params_upper, pixel_rel_idx, ndvi_arr, device)
            old_dates = last_dates_arr[:7, pixel_rel_idx].copy()
            old_dates = np.array(old_dates, dtype='S10')
            shifted = np.empty(old_dates.shape, dtype='S10')
            shifted[:-1] = old_dates[1:]
            shifted[-1] = day_date.strftime("%Y-%m-%d").encode("utf-8")
            last_dates_arr[:7, pixel_rel_idx] = shifted
            valid_window = [zarr_date_to_date(d) for d in shifted]
            if all(d != date(1900,1,1) for d in valid_window):
                L2_smoothing(pixel_rel_idx, 2, params_lower, params_upper, ndvi_arr, last_dates_arr, dates_list, device)
            last_dates_arr[7:, pixel_rel_idx] = b"1900-01-01"
    else:
        date_to_potential = day_date.strftime("%Y-%m-%d").encode("utf-8")
        last_dates_arr[7:, pixel_rel_idx] = date_to_potential
    timing['time'] += time.perf_counter() - t0

def process_file(file_path):
    timing = {'load':0.0, 'write':0.0, 'median':0.0, 'time':0.0, 'calls':0}
    try:
        t_file_start = time.perf_counter()
        file_name = os.path.basename(file_path)
        final_out_path = os.path.join(OUTPUT_DIR, file_name)
        tmp_out_path = final_out_path + f".tmp_{os.getpid()}"
        if os.path.exists(tmp_out_path):
            shutil.rmtree(tmp_out_path)
        shutil.copytree(file_path, tmp_out_path)
        ds = zarr.open_group(tmp_out_path, mode="r+")
        ndvi_zarr = ds["ndvi"]
        last_dates_zarr = ds["last_dates"]
        params_lower_zarr = ds["params"]["params_lower"]
        params_upper_zarr = ds["params"]["params_upper"]
        dates_zarr = ds["dates"]
        dates_list = [zarr_date_to_date(d) for d in dates_zarr[:]]
        n_pixels = ndvi_zarr.shape[1]
        seed = int(hashlib.md5(file_name.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        selected_pixels_global = rng.choice(np.arange(n_pixels), size=N_PIXELS_PER_FILE, replace=False)
        print(selected_pixels_global)
        t0 = time.perf_counter()
        ndvi_arr = np.array(ndvi_zarr[:, selected_pixels_global], dtype=np.int16)
        last_dates_arr = np.array(last_dates_zarr[:, selected_pixels_global], dtype='S10')
        params_lower_arr = np.array(params_lower_zarr[selected_pixels_global, :], dtype=np.float32)
        params_upper_arr = np.array(params_upper_zarr[selected_pixels_global, :], dtype=np.float32)
        t_read = time.perf_counter() - t0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t_iter_start = time.perf_counter()
        for day_date in dates_list[:PROCESS_DATES_LIMIT]:
            for rel_idx, global_idx in enumerate(selected_pixels_global):
                continous_ndvi(day_date, rel_idx, global_idx, ndvi_arr, last_dates_arr, params_lower_arr, params_upper_arr, dates_list, device, timing)
        t_iter = time.perf_counter() - t_iter_start
        t0 = time.perf_counter()
        ndvi_zarr[:, selected_pixels_global] = ndvi_arr
        last_dates_zarr[:, selected_pixels_global] = last_dates_arr
        t_write = time.perf_counter() - t0
        zarr.consolidate_metadata(tmp_out_path)
        if not os.path.exists(final_out_path):
            os.rename(tmp_out_path, final_out_path)
        else:
            shutil.rmtree(tmp_out_path)
        t_file_end = time.perf_counter()
        print(f"[{file_name}] loaded={t_read:.3f}s iter={t_iter:.3f}s write={t_write:.3f}s total={t_file_end - t_file_start:.3f}s calls={timing['calls']} time_spent={timing['time']:.3f}s")
        return {"file": file_name, "status": "ok", "timing": timing}
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(tmp_out_path):
            shutil.rmtree(tmp_out_path)
        return {"file": file_path, "status": "error", "error": str(e)}

results = []  

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
            results.append(res)
    print("✅ All processing complete.")

rows = []
for r in results:
    row = {'file': r.get('file'), 'status': r.get('status')}
    # flatten timing info
    timing = r.get('timing', {})
    for k, v in timing.items():
        row[k] = v
    rows.append(row)

# Convert to DataFrame and save
df = pd.DataFrame(rows)
df.to_csv('workflow_implementation/output/timing_as_it_is.csv', index=False)
