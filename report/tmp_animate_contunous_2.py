#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tmp_animate_contunous_2.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/logs/animate_images.log 2>&1 &

"""
Simulate the continuous NDVI processing pipeline for 2022.

For each new observation date in 2022 the function evaluates ONLY the newest
observation: it checks whether the new point is an outlier by placing it as
the last point of a 7-point window built from the 6 most-recent valid
(non-outlier, in-range) observations.  If it is NOT an outlier the 7-point
LOESS window is smoothed, the smoothed delta is accepted for that new point,
the full time-series delta is re-interpolated, and the mask is updated.

Mask legend (identical to production pipeline):
  0 – not an observation, not yet smoothed
  1 – not an observation, smoothed (gap-filled by interpolation)
  2 – observation, not yet smoothed
  3 – observation, smoothed
  4 – observation flagged as outlier
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", message=".*Zarr version 3.*")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paths & Constants
# ─────────────────────────────────────────────────────────────────────────────
HIST_ZARR   = "/mnt/data1/UniBe-swiss-ndvi/backup/historical_backup.zarr"
OBS_ZARR    = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
OUT_DIR     = "./img_tmp"

NO_COVERAGE = 2**15 - 1 

os.makedirs(OUT_DIR, exist_ok=True)
selected_pixel_ids = np.array([90415334, 93677703, 46053259, 54232662,
                               84599346, 84468278,   427960, 73583022])

names =  ["Lowland broadleaf","Highland broadleaf","Lowland evergreen","Highland evergreen","Biscth fire affected","Biscth fire non affected","Drought affected","Vaia storm affected"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

# A. Load Historical Processed State (up to 2021)
hist_ds = xr.open_zarr(HIST_ZARR).sel(pixel=selected_pixel_ids)
hist_sub = hist_ds.sel(date=slice(None, "2021-12-31")).compute()
# B. Load Raw Observation Data for 2022
obs_ds = xr.open_dataset(OBS_ZARR, chunks={}, mask_and_scale=False)
obs_sub = obs_ds[["ndvi"]].sel(pixel=selected_pixel_ids).compute()
print(hist_sub)

print(obs_sub)
# Clean raw obs: floor datetimes to dates and take the first obs per day
obs_datetimes = pd.DatetimeIndex(obs_sub["datetime"].values)
obs_dates_floored = obs_datetimes.floor("D")
first_idx = np.flatnonzero(~obs_dates_floored.duplicated(keep="first"))
raw_daily = (
    obs_sub.isel(datetime=first_idx)
    .assign_coords(date=("datetime", obs_dates_floored[first_idx].values))
    .swap_dims({"datetime": "date"})
    .sel(date=slice("2022-01-01", "2022-12-31"))
)

# Identify observation dates for the loop
obs_2022_dates = pd.to_datetime(raw_daily.date.values)

# C. Lookup Table for Medians
lut = xr.open_zarr(LOOKUPTABLE).sel(pixel=selected_pixel_ids).compute()

def get_medians_for_dates(dates_pd: pd.DatetimeIndex) -> np.ndarray:
    doys = np.clip(dates_pd.dayofyear.values, 1, 365)
    return lut["median_ndvi"].values[:, doys - 1].T

# ─────────────────────────────────────────────────────────────────────────────
# 3. Processing Function (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def process_new_observation(ndvi_arr, medians, mask_array, is_observation_date, dates, new_obs_date):
    
    dates = dates.astype("datetime64[D]")
    new_obs_date = np.datetime64(new_obs_date, "D")
    mask_array = np.array(mask_array, copy=True, dtype=np.int8)
    ndvi_out = ndvi_arr.copy()
    outlier_info = None  # will be (date, ndvi_value) if outlier detected

    ndvi_s = ndvi_arr / 10000.0
    median_s = medians / 10000.0

    new_obs_candidates = np.where(is_observation_date & (dates == new_obs_date))[0]
    if len(new_obs_candidates) == 0:
        return ndvi_out, mask_array, outlier_info

    idx = new_obs_candidates[0]
    if not (0.0 < ndvi_s[idx] < 1.0):
        mask_array[idx] = 5
        outlier_info = (dates[idx], 15000 )
        return ndvi_out, mask_array, outlier_info

    mask_array[idx] = 2

    valid_before = (np.arange(len(ndvi_s)) < idx) & (ndvi_s > 0.0) & (ndvi_s < 1.0) & is_observation_date
    prev_valid_idx = np.where(valid_before)[0]

    if len(prev_valid_idx) < 6:
        return ndvi_out, mask_array, outlier_info

    # --- Outlier detection on the full valid observation history ---
    ndvi_valid   = ndvi_s[prev_valid_idx]
    median_valid = median_s[prev_valid_idx]

    delta_threshold       = 0.1
    delta_delta_threshold = 0.1

    delta_ndvi        = ndvi_valid - median_valid
    delta_delta_left  = delta_ndvi[2:]
    delta_delta_right = delta_ndvi[:-2]

    outlier_mask = (
        (np.abs(delta_ndvi[1:-1]) > delta_threshold) &
        (np.abs(delta_delta_left) > delta_delta_threshold) &
        (np.abs(delta_delta_right) > delta_delta_threshold)
    )
    outlier_positions = prev_valid_idx[1:-1][outlier_mask]
    mask_array[outlier_positions] = 5

    # Re-filter excluding newly detected outliers
    valid_before_no_outlier = valid_before & (mask_array != 4)
    prev_valid_idx = np.where(valid_before_no_outlier)[0]

    if len(prev_valid_idx) < 6:
        return ndvi_out, mask_array, outlier_info

    # 7-point window: 6 previous valid obs + new obs
    window_idx = np.append(prev_valid_idx[-6:], idx)
    delta_win = ndvi_s[window_idx] - median_s[window_idx]

    # --- Outlier detection for the current new observation ---
    d_new = delta_win[6]
    d_l1  = delta_win[5]
    if (abs(d_new) > delta_threshold) and (abs(d_new - d_l1) > delta_delta_threshold):
        mask_array[idx] = 5
        outlier_info = (dates[idx], ndvi_s[idx] * 10000.0)  # store date and original scaled value
        return ndvi_out, mask_array, outlier_info

    # LOESS over the 7-point window; center point [3] is the one being finalized
    loess = sm.nonparametric.lowess(delta_win, np.arange(7), frac=1.0, it=3, return_sorted=False)
    smoothing_cutoff_obs = window_idx[3]

    # Build delta array over all valid (non-outlier) points
    all_v_mask = (ndvi_s > 0.0) & (ndvi_s < 1.0) & (mask_array != 4) & is_observation_date
    all_v_idx = np.where(all_v_mask)[0]
    delta_all_v = ndvi_s[all_v_idx] - median_s[all_v_idx]

    # Replace center-of-window delta with LOESS-smoothed value
    pos_center = int(np.searchsorted(all_v_idx, smoothing_cutoff_obs))
    delta_all_v[pos_center] = loess[3]

    days_full = (dates - dates[0]) / np.timedelta64(1, "D")
    days_v = days_full[all_v_idx]
    interp_delta = np.interp(days_full, days_v, delta_all_v)

    ndvi_smoothed = 10000.0 * (interp_delta + median_s)

    obs_mask_full = is_observation_date & (ndvi_s > 0.0) & (ndvi_s < 1.0) & (mask_array != 4)
    before = np.arange(len(mask_array)) <= smoothing_cutoff_obs

    mask_outlier = mask_array == 5

    mask_array[before & obs_mask_full] = 3
    mask_array[before & ~obs_mask_full] = 1
    mask_array[before & mask_outlier] = 4
    mask_array[idx] = 2

    ndvi_out = ndvi_smoothed
    return ndvi_out, mask_array, outlier_info
# ─────────────────────────────────────────────────────────────────────────────
# 4. Continuous Simulation (Stacking Mode)
# ─────────────────────────────────────────────────────────────────────────────


# Initialize persistent state from backup
current_dates  = pd.to_datetime(hist_sub.date.values)

# and transpose (.T) from (pixel, date) to (date, pixel)
current_ndvi   = hist_sub.ndvi_processed.values.T.astype(float) 
current_mask   = hist_sub.mask_array.values.T.astype(np.int8)

current_raw    = current_ndvi.copy() # Keeps the raw observation history for plotting dots
current_is_obs = np.any((current_mask == 2) | (current_mask == 3) | (current_mask == 4), axis=1)

n_pixels = len(selected_pixel_ids)


for today in obs_2022_dates:
    
    # --- STACKING STEP ---
    # 1. Fill gaps between the last processed date and this observation date
    last_date = current_dates[-1]
    gap_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                              end=today - pd.Timedelta(days=1), freq='D')
    
    if not gap_dates.empty:
        n_gaps = len(gap_dates)
        gap_ndvi = np.full((n_gaps, n_pixels), NO_COVERAGE, dtype=float)
        gap_mask = np.zeros((n_gaps, n_pixels), dtype=np.int8)
        gap_is_obs = np.zeros(n_gaps, dtype=bool)
        
        current_ndvi   = np.vstack([current_ndvi, gap_ndvi])
        current_mask   = np.vstack([current_mask, gap_mask])
        current_dates  = current_dates.append(gap_dates)
        current_is_obs = np.concatenate([current_is_obs, gap_is_obs])

    # 2. Stack today's raw observation
    today_raw_vals = raw_daily.sel(date=today).ndvi.values.astype(float)
    current_ndvi   = np.vstack([current_ndvi, today_raw_vals])
    
    # Initial flag for new observation is 2
    today_init_mask = np.full(n_pixels, 2, dtype=np.int8)
    current_mask    = np.vstack([current_mask, today_init_mask])
    
    current_dates  = current_dates.append(pd.DatetimeIndex([today]))
    current_is_obs = np.append(current_is_obs, True)

    # --- PROCESSING STEP ---
    # Run analysis on the now-extended series
    medians_all = get_medians_for_dates(current_dates)
    dates_np    = current_dates.values.astype("datetime64[D]")

    outlier_log = {p: [] for p in range(n_pixels)}

    for p in range(n_pixels):
        # The function processes the series and updates history + new point
        new_ndvi_p, new_mask_p, outlier_info  = process_new_observation(
            ndvi_arr            = current_ndvi[:, p],
            medians             = medians_all[:, p],
            mask_array          = current_mask[:, p],
            is_observation_date = current_is_obs,
            dates               = dates_np,
            new_obs_date        = today.to_datetime64()
        )
        current_ndvi[:, p] = new_ndvi_p
        current_mask[:, p] = new_mask_p
        if outlier_info is not None:
            outlier_log[p].append(outlier_info) 

        # --- Plotting Logic ---
        # Logic for masks
        no_obs_to_smooth = new_mask_p == 0
        no_obs_smoothed  = new_mask_p == 1
        obs_to_smooth    = new_mask_p == 2
        obs_smoothed     = new_mask_p == 3
        outlier_smoothed = new_mask_p == 4
        outlier = new_mask_p == 5
        valid_obs        = (obs_smoothed | outlier_smoothed)
        
        # Identify "valid" for the continuous green line (smoothed gaps + smoothed obs)
        valid_line = (no_obs_smoothed | obs_smoothed)

        # take the last 30 days 

        # Find indices of all valid observations (mask 2, 3, or 4)
        all_obs_idx = np.where((new_mask_p == 2) | (new_mask_p == 3) | (new_mask_p == 4))[0]
        if len(all_obs_idx) >= 30:
            plot_start_idx = all_obs_idx[-30]
        elif len(all_obs_idx) > 0:
            plot_start_idx = all_obs_idx[0]
        else:
            plot_start_idx = max(0, len(dates_np) - 30)

        date_recent          = dates_np[plot_start_idx:]
        medians_recent       = medians_all[plot_start_idx:]
        valid_line_recent    = valid_line[plot_start_idx:]
        new_ndvi_p_recent    = new_ndvi_p[plot_start_idx:]
        valid_obs_recent     = valid_obs[plot_start_idx:]
        obs_to_smooth_recent = obs_to_smooth[plot_start_idx:]
        outlier_recent       = outlier_smoothed[plot_start_idx:]

        last_obs_arr = date_recent[valid_obs_recent]

        if len(last_obs_arr) == 0:
            lag = pd.Timedelta("NaT")
        else:
            last_obs = last_obs_arr[-1]
            lag = today - last_obs

        plt.figure(figsize=(7.2, 4), dpi=200)

        # 1. Median Line (Black)
        plt.plot(date_recent, medians_recent[:, p] / 10000.0, 
                 linestyle="-", linewidth=1.0, color="black", label="median NDVI values", alpha=0.7)

        # 2. Smoothed Continuous Line (Green)
        # Note: We only plot up to 'today' to keep it clean
        plt.plot(date_recent[valid_line_recent], new_ndvi_p_recent[valid_line_recent] / 10000.0, 
                 linestyle="-", linewidth=1.2, color="green", label="smoothed state")

        # 3. Observed Points (Green X)
        plt.plot(date_recent[valid_obs_recent], new_ndvi_p_recent[valid_obs_recent] / 10000.0, 
                 marker="o", linestyle="None", markersize=4, color="green", label="obs smoothed")
        
        plt.plot(date_recent[obs_to_smooth_recent], new_ndvi_p_recent[obs_to_smooth_recent] / 10000.0, 
                 marker="o", linestyle="None", markersize=4, color="yellow", label="obs to smooth")

        plt.plot(date_recent[outlier_recent], new_ndvi_p_recent[outlier_recent] / 10000.0, 
                 marker="o", linestyle="None", markersize=4, color="red", label="outlier or invalid filtered out and smoothed")
        
        # Plot outliers at their original raw values from outlier_log
        if outlier_log[p]:
            outlier_dates  = np.array([o[0] for o in outlier_log[p]])
            outlier_values = np.array([o[1] for o in outlier_log[p]])
            # filter to only those within the plot window
            in_window = (outlier_dates >= date_recent[0]) & (outlier_dates <= date_recent[-1])
            if in_window.any():
                plt.plot(outlier_dates[in_window], outlier_values[in_window] / 10000.0,
                         marker="o", linestyle="None", markersize=4, color="black",
                         label="outlier or invalid in the original position")

        # Styling
        plt.ylim(-0.2, 1.6) 
        plt.xlabel("Date")
        plt.ylabel("NDVI")
        plt.title(f"NDVI Time Series of {names[p]} –  {today.strftime('%Y-%m-%d')}")
        plt.suptitle(f"lag between current date and last smoothed date : {lag.days} days")        
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=0, ha='right')
        plt.legend(fontsize='x-small', loc='lower left', ncol=3)
        plt.tight_layout()


        # --- Fixed Path Logic ---
        # Create pixel-specific directory if it doesn't exist
        pixel_dir = os.path.join(OUT_DIR, names[p])
        os.makedirs(pixel_dir, exist_ok=True)
        
        # today.strftime('%Y-%m-%d') converts the timestamp to a string like "2022-01-01"
        plotpath = os.path.join(pixel_dir, f"{today.strftime('%Y-%m-%d')}.png")
        
        plt.savefig(plotpath)
        plt.close()

    print(f"Processed observation for {today.date()}. Total series length: {len(current_dates)}")

print("Simulation finished.")