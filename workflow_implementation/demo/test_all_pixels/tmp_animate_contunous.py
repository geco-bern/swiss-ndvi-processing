#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tmp_animate_contunous.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/logs/animate_images.log 2>&1 &

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
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message=".*Zarr version 3.*")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
OBS_ZARR    = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
OUT_DIR     = "./img_tmp"

NO_COVERAGE = 2**15 - 1    #  32767
INVALID     = -(2**15)     # -32768

# Last date of the already-processed historic baseline.
# Kept as numpy.datetime64[D] throughout so np.searchsorted never hits a
# type mismatch between pd.Timestamp and datetime.date.
CUTOFF_DATE = np.datetime64("2021-12-31", "D")

os.makedirs(OUT_DIR, exist_ok=True)

names              = ["low_broad", "high_broad", "low_ever", "high_ever",
                      "fire",      "non_fire",   "drought",  "storm"]
selected_pixel_ids = np.array([90415334, 93677703, 46053259, 54232662,
                                84599346, 84468278,   427960, 73583022])

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data prep
# ─────────────────────────────────────────────────────────────────────────────
obs_ds  = xr.open_dataset(OBS_ZARR, chunks={}, mask_and_scale=False)
lut     = xr.open_zarr(LOOKUPTABLE, chunks={})
obs_sub = obs_ds[["ndvi"]].sel(pixel=selected_pixel_ids).compute()

# Collapse to one value per day (keep first observation each day)
obs_datetimes     = pd.DatetimeIndex(obs_sub["datetime"].values)
obs_dates_floored = obs_datetimes.floor("D")
first_idx         = np.flatnonzero(~obs_dates_floored.duplicated(keep="first"))
ndvi_daily = (
    obs_sub
    .isel(datetime=first_idx)
    .assign_coords(date=("datetime", obs_dates_floored[first_idx].values))
    .swap_dims({"datetime": "date"})
)

# Reindex to a gapless daily timeline (gaps → NO_COVERAGE)
full_idx  = pd.date_range(start=obs_dates_floored.min(),
                           end=obs_dates_floored.max(), freq="D")
ndvi_full = ndvi_daily.reindex(date=full_idx,
                                fill_value={"ndvi": np.int16(NO_COVERAGE)})

lut_sub = lut["median_ndvi"].sel(pixel=selected_pixel_ids).compute()


def get_medians(dates_pd: pd.DatetimeIndex) -> np.ndarray:
    """Return median NDVI shape (n_dates, n_pixels), int16-scale."""
    doys = np.clip(dates_pd.dayofyear.values, 1, 365)
    return lut_sub.values[:, doys - 1].T


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Core per-step function
#
# Strategy (matches user requirement):
#   • Receive the full time series up to NOW (including the new obs at the end).
#   • Identify the single newest observation.
#   • Build a 7-point window: [6 previous valid non-outlier obs] + [new obs].
#   • Decide outlier / not using the same delta / delta-delta test as the
#     production pipeline (5_analyse_demo_efficient / 5_analyse_demo_francesco).
#   • If NOT an outlier: LOESS-smooth the 7-point window, accept the smoothed
#     delta for the new point, re-interpolate the full series, update mask.
#   • If IS an outlier: mark mask=4, leave NDVI at raw value, done.
#   • Persistent state (mask, smoothed NDVI) is carried forward across steps.
# ─────────────────────────────────────────────────────────────────────────────

def process_new_observation(
    ndvi_arr:            np.ndarray,   # float64, int16-scale (×10 000), full window
    medians:             np.ndarray,   # float64, int16-scale (×10 000), full window
    mask_array:          np.ndarray,   # int8, current mask state (modified in place copy)
    is_observation_date: np.ndarray,   # bool, full window
    dates:               np.ndarray,   # datetime64[D], full window
    new_obs_date:        np.datetime64,# datetime64[D] — the date to evaluate
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the single observation at `new_obs_date` and update the smooth.

    Returns updated (ndvi_smoothed, mask_array) for the full window.
    Both input arrays are treated as read-only; copies are made internally.
    """
    # --- type safety ---------------------------------------------------------
    dates        = dates.astype("datetime64[D]")
    new_obs_date = np.datetime64(new_obs_date, "D")
    mask_array   = np.array(mask_array, copy=True, dtype=np.int8)
    ndvi_out     = ndvi_arr.copy()

    # Scale to [−1, 1]
    ndvi_s   = ndvi_arr   / 10000.0
    median_s = medians    / 10000.0

    # --- locate the new observation ------------------------------------------
    new_obs_candidates = np.where(
        is_observation_date & (dates == new_obs_date)
    )[0]

    if len(new_obs_candidates) == 0:
        return ndvi_out, mask_array   # nothing to do

    new_obs_idx = new_obs_candidates[0]

    # Skip if the value is out of the valid NDVI range
    if not (0.0 < ndvi_s[new_obs_idx] < 1.0):
        return ndvi_out, mask_array

    # Mark as "observation, not yet smoothed" (may be overwritten below)
    mask_array[new_obs_idx] = 2

    # --- collect the 6 previous valid non-outlier observations ---------------
    # "valid" = NDVI in (0, 1) AND not already flagged as outlier (mask != 4)
    valid_before = (
        (np.arange(len(ndvi_s)) < new_obs_idx) &
        (ndvi_s > 0.0) & (ndvi_s < 1.0) &
        (mask_array != 4)
    )
    prev_valid_idx = np.where(valid_before)[0]

    if len(prev_valid_idx) < 6:
        # Not enough history yet — keep mask=2 and return
        return ndvi_out, mask_array

    six_prev_idx = prev_valid_idx[-6:]

    # --- 7-point window: [6 prev] + [new obs] --------------------------------
    window_idx  = np.append(six_prev_idx, new_obs_idx)   # length exactly 7
    ndvi_win    = ndvi_s[window_idx]
    median_win  = median_s[window_idx]
    delta_win   = ndvi_win - median_win

    # --- outlier detection on the new observation ----------------------------
    # The new obs sits at position 6 (last) in the window.
    # We check it as an "inner point" relative to its two left neighbours:
    #   delta_delta_left  = |delta[6] − delta[5]|   (immediate left)
    #   delta_delta_right = |delta[6] − delta[4]|   (second left)
    # This mirrors the production pipeline logic:
    #   delta_delta_left  = delta_ndvi[2:]    (right neighbour of centre)
    #   delta_delta_right = delta_ndvi[:-2]   (left neighbour of centre)
    delta_threshold       = 0.1
    delta_delta_threshold = 0.1

    d_new    = delta_win[6]
    d_left1  = delta_win[5]
    d_left2  = delta_win[4]

    is_outlier = (
        (abs(d_new)           > delta_threshold)       &
        (abs(d_new - d_left1) > delta_delta_threshold) &
        (abs(d_new - d_left2) > delta_delta_threshold)
    )

    if is_outlier:
        mask_array[new_obs_idx] = 4
        return ndvi_out, mask_array   # raw value kept, no interpolation update

    # --- LOESS on the 7-point window -----------------------------------------
    idx7  = np.arange(7, dtype=float)
    loess = sm.nonparametric.lowess(
        delta_win, idx7, frac=1.0, it=3, return_sorted=False
    )
    smoothed_delta_new = loess[6]   # smoothed delta for the new observation

    # --- rebuild the full interpolated NDVI ----------------------------------
    # Use ALL valid non-outlier points (including new obs) as interpolation knots.
    all_valid_mask = (
        (ndvi_s > 0.0) & (ndvi_s < 1.0) &
        (mask_array != 4)
    )
    all_valid_idx = np.where(all_valid_mask)[0]

    ndvi_all_v   = ndvi_s[all_valid_idx]
    median_all_v = median_s[all_valid_idx]
    delta_all_v  = ndvi_all_v - median_all_v

    # Replace the raw delta of the new obs with the LOESS-smoothed version
    pos_new = int(np.searchsorted(all_valid_idx, new_obs_idx))
    delta_all_v[pos_new] = smoothed_delta_new

    days_diff_full  = (dates - dates[0]) / np.timedelta64(1, "D")
    days_diff_valid = days_diff_full[all_valid_idx]

    # Interpolate using only the valid observation deltas as knots.
    # np.interp extrapolates flat (clamps) beyond the first/last knot,
    # so no false 0-anchors are needed — the curve passes exactly through
    # every accepted observation point.
    interp_delta  = np.interp(days_diff_full, days_diff_valid, delta_all_v)
    ndvi_smoothed = 10000.0 * (interp_delta + median_s)

    # --- update mask ---------------------------------------------------------
    obs_mask_full = (
        is_observation_date &
        (ndvi_s > 0.0) & (ndvi_s < 1.0) &
        (mask_array != 4)
    )

    # All dates strictly before the new obs are now finalised (smoothed)
    before = np.arange(len(mask_array)) < new_obs_idx
    mask_array[before & obs_mask_full]  = 3   # prior obs smoothed
    mask_array[before & ~obs_mask_full] = 1   # prior gap-fill smoothed
    mask_array[new_obs_idx]             = 3   # new obs just smoothed

    return ndvi_smoothed, mask_array


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

all_dates_pd = pd.DatetimeIndex(ndvi_full["date"].values)
all_dates_np = all_dates_pd.values.astype("datetime64[D]")

# Boolean flag: True on days that carry a real satellite observation
obs_dates_set = set(obs_dates_floored.normalize())
is_obs_all    = np.array([d in obs_dates_set for d in all_dates_pd.normalize()])

# All observation dates that fall in 2022
obs_2022 = all_dates_pd[(all_dates_pd.year == 2022) & is_obs_all]

# Pre-compute medians for every date  shape: (n_dates, n_pixels)
medians_all = get_medians(all_dates_pd)

n_pixels = len(names)
n_dates  = len(all_dates_np)

# Persistent state: smoothed NDVI and mask, carried across steps
n_state    = ndvi_full["ndvi"].values.astype(float).copy()   # (n_dates, n_pixels)
mask_state = np.zeros((n_dates, n_pixels), dtype=np.int8)

# Initialise mask=2 for all existing observation dates in the full history
for t in range(n_dates):
    if is_obs_all[t]:
        for p in range(n_pixels):
            if 0.0 < n_state[t, p] / 10000.0 < 1.0:
                mask_state[t, p] = 2


for step_i, current_obs_date in enumerate(obs_2022):

    # Work on a growing window up to the current observation date
    win_mask   = all_dates_pd <= current_obs_date
    d_win_np   = all_dates_np[win_mask]
    n_win_raw  = ndvi_full["ndvi"].values[win_mask, :]   
    m_win      = medians_all[win_mask, :]
    o_win      = is_obs_all[win_mask]

    n_win_state    = n_state[win_mask, :].copy()
    mask_win_state = mask_state[win_mask, :].copy()

    new_obs_date_np = current_obs_date.to_datetime64().astype("datetime64[D]")

    for p in range(n_pixels):
        # FIX 1: Pass n_win_raw instead of n_win_state to ensure the 
        # interpolation knots are the actual raw observation values.
        new_ndvi, new_mask = process_new_observation(
            ndvi_arr            = n_win_raw[:, p], # Use RAW data here
            medians             = m_win[:, p],
            mask_array          = mask_win_state[:, p],
            is_observation_date = o_win,
            dates               = d_win_np,
            new_obs_date        = new_obs_date_np,
        )
        n_win_state[:, p]    = new_ndvi
        mask_win_state[:, p] = new_mask

    # Write back into the persistent state arrays
    n_state[win_mask, :]    = n_win_state
    mask_state[win_mask, :] = mask_win_state

    if step_i < 40:
        continue   

    # ── Plotting ──────────────────────────────────────────────────────────────
    for p in range(n_pixels):
        obs_in_win_idx = np.where(
            o_win &
            (n_win_raw[:, p] / 10000.0 > 0.0) &
            (n_win_raw[:, p] / 10000.0 < 1.0)
        )[0]
        last17 = obs_in_win_idx[-17:] if len(obs_in_win_idx) >= 17 else obs_in_win_idx

        if len(last17) == 0:
            continue

        # Plot window boundaries
        plot_date_start = d_win_np[last17[0]]
        plot_date_end   = d_win_np[last17[-1]]
        p_mask = (d_win_np >= plot_date_start) & (d_win_np <= plot_date_end)

        # Split indices
        smoothed_obs_idx   = last17[:-3]   # green x markers
        unsmoothed_obs_idx = last17[-3:]   # yellow x markers

        dates_plot  = d_win_np[p_mask]
        mask_plot   = mask_win_state[p_mask, p]
        ndvi_smooth = n_win_state[p_mask, p] / 10000.0
        median_plot = m_win[p_mask, p] / 10000.0

        # FIX 2: Truncate the line at the last "green" observation
        # Identify the date of the very last green dot
        if len(smoothed_obs_idx) > 0:
            line_end_date = d_win_np[smoothed_obs_idx[-1]]
        else:
            line_end_date = dates_plot[0]

        # Modify the line mask to stop at line_end_date
        smoothed_mask = ((mask_plot == 1) | (mask_plot == 3)) & (dates_plot <= line_end_date)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

        # ── Green line: smoothed NDVI ────────
        green_y = ndvi_smooth.copy()
        green_y[~smoothed_mask] = np.nan
        ax.plot(dates_plot, green_y,
                linestyle="-", linewidth=1.5, color="green",
                label="Smoothed", zorder=3)

        # ── Black line: median NDVI ──────────
        ax.plot(dates_plot, median_plot,
                linestyle="-", linewidth=1.2, color="black",
                label="Median NDVI", zorder=2)

        # ── Green x markers ──────────────────
        ax.plot(d_win_np[smoothed_obs_idx],
                n_win_raw[smoothed_obs_idx, p] / 10000.0,
                marker="x", linestyle="None", markersize=6,
                color="green", label="Obs smoothed", zorder=5)

        # ── Yellow x markers ─────────────────
        ax.plot(d_win_np[unsmoothed_obs_idx],
                n_win_raw[unsmoothed_obs_idx, p] / 10000.0,
                marker="x", linestyle="None", markersize=6,
                color="gold", label="Obs to smooth", zorder=5)



        # ── Red x markers: outliers ────────────────────────────────────────
        if outlier_m.any():
            ax.plot(dates_plot[outlier_m],
                    n_win_raw[p_mask, p][outlier_m] / 10000.0,
                    marker="x", linestyle="None", markersize=6,
                    color="red", label="Outlier", zorder=5)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Date")
        ax.set_ylabel("NDVI")
        ax.set_title(f"NDVI Time Series of {names[p]} – Step {step_i + 1} ({current_obs_date.date()})")
        ax.grid(True)
        ax.legend(fontsize="x-small", loc="upper left", ncol=2)
        plt.tight_layout()

        p_dir = os.path.join(OUT_DIR, names[p])
        os.makedirs(p_dir, exist_ok=True)
        fig.savefig(os.path.join(p_dir, f"step_{step_i + 1:03d}.png"))
        plt.close(fig)

print("done")