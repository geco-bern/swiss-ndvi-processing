import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from scipy.signal import savgol_filter


def gapfill_ndvi(
    ndvi_series,
    lower,
    upper,
    forecasting=False,
    param_iqr=1.5,
    bottom_q=0.3,
    top_q=0.7,
):
    """
    NDVI gapfilling.

    Steps:
      1. Normalize & mask clouds.
      2. Detect outliers vs logistic model bounds.
      3. Remove outliers.
      4. Fill gaps using interpolation guided by logistic model median.

    Modes:
      - forecasting=False: behaves like gapfill_timeseries
          returns ndvi_gapfilled, outlier_mask
      - forecasting=True : behaves like sequential_gapfill
          returns ndvi_filled, outlier_mask, forecast_only
    """
    # --- normalize ---
    ndvi_series = ndvi_series.astype(float) / 10000.0
    ndvi_series = np.where((ndvi_series > 1) | (ndvi_series < 0), np.nan, ndvi_series)

    n = len(ndvi_series)
    outlier_mask = np.zeros(n, dtype=bool)

    # --------------------
    # Common outlier detection
    # --------------------
    valid_idx = np.where(np.isfinite(ndvi_series))[0]
    if len(valid_idx) < 2:
        if forecasting:
            return ndvi_series.copy(), outlier_mask, np.full(n, np.nan)
        else:
            return ndvi_series.copy(), outlier_mask

    valid_ndvi = ndvi_series[valid_idx]
    valid_upper = upper[valid_idx]
    valid_lower = lower[valid_idx]
    valid_iqr = valid_upper - valid_lower
    median_valid = valid_upper - valid_iqr / 2

    # threshold outliers
    th_hi = valid_upper + param_iqr * valid_iqr
    th_lo = valid_lower - param_iqr * valid_iqr
    is_outlier_threshold = (valid_ndvi > th_hi) | (valid_ndvi < th_lo)

    # slope outliers
    deltas = valid_ndvi - median_valid
    delta_diff = np.diff(deltas)
    if len(delta_diff) > 2:
        q_low, q_hi = np.quantile(delta_diff, [bottom_q, top_q])
        slope_outlier = np.zeros_like(is_outlier_threshold)
        slope_outlier[1:-1] = (
            (delta_diff[1:] > q_hi) | (delta_diff[1:] < q_low)
        ) & (
            (delta_diff[:-1] > q_hi) | (delta_diff[:-1] < q_low)
        )
        is_outlier = is_outlier_threshold & slope_outlier
    else:
        is_outlier = is_outlier_threshold

    outlier_mask[valid_idx] = is_outlier
    ndvi_series[valid_idx[is_outlier]] = np.nan

    # =========================================================
    # Non-sequential mode (batch interpolation like gapfill_timeseries)
    # =========================================================
    if not forecasting:
        ndvi_gapfilled = ndvi_series.copy()
        valid_idx = np.where(np.isfinite(ndvi_gapfilled))[0]
        if len(valid_idx) < 2:
            return ndvi_gapfilled, outlier_mask

        for i in range(len(valid_idx) - 1):
            start, end = valid_idx[i], valid_idx[i + 1]
            if end - start > 1:
                for j, idx in enumerate(range(start + 1, end)):
                    frac = (j + 1) / (end - start)
                    # plain linear interpolation between obs
                    obs_interp = (
                        ndvi_gapfilled[start]
                        + (ndvi_gapfilled[end] - ndvi_gapfilled[start]) * frac
                    )
                    # logistic model median
                    median_val = upper[idx] - (upper[idx] - lower[idx]) / 2
                    # bias toward median, max at center
                    diff = median_val - obs_interp
                    weight = frac * (1 - frac)
                    ndvi_gapfilled[idx] = obs_interp + 0.5 * weight * diff

        return ndvi_gapfilled, outlier_mask

    # =========================================================
    # Sequential mode (like sequential_gapfill with forecast_only)
    # =========================================================
    ndvi_filled = np.full(n, np.nan)
    forecast_only = np.full(n, np.nan)
    last_idx = None

    for t in range(n):
        obs = ndvi_series[t]
        if not np.isnan(obs):
            ndvi_filled[t] = obs
            if last_idx is not None and t - last_idx > 1:
                # retroactively interpolate between last_idx and t
                for j, idx in enumerate(range(last_idx + 1, t)):
                    frac = (j + 1) / (t - last_idx)
                    obs_interp = (
                        ndvi_filled[last_idx]
                        + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac
                    )
                    median_val = upper[idx] - (upper[idx] - lower[idx]) / 2
                    diff = median_val - obs_interp
                    weight = frac * (1 - frac)
                    val_retro = obs_interp + 0.5 * weight * diff
                    ndvi_filled[idx] = val_retro
                    # forecast_only keeps earlier forecasts, do not overwrite
            last_idx = t
        else:
            if last_idx is not None:
                # simple flat forecast relative to median
                med_t = upper[t] - (upper[t] - lower[t]) / 2
                delta_last = ndvi_filled[last_idx] - (
                    upper[last_idx] - (upper[last_idx] - lower[last_idx]) / 2
                )
                forecast_val = med_t + delta_last
                ndvi_filled[t] = forecast_val
                forecast_only[t] = forecast_val

    return ndvi_filled, outlier_mask, forecast_only



def plot_results(title,pixel_idx, ndvi_series, ndvi_gapfilled, outlier_arr, lower, upper, dates, save_path=None):
    """
    Plot NDVI time series for a pixel:
      - raw NDVI with outliers highlighted
      - logistic model bounds
      - gapfilled NDVI
    """
    colors = np.where(outlier_arr, "red", "green")

    fig, ax = plt.subplots(figsize=(12, 6))

    # logistic bounds
    ax.plot(dates, lower, label="Lower Bound", color="blue", alpha=0.7)
    ax.plot(dates, upper, label="Upper Bound", color="blue", alpha=0.7)
    ax.fill_between(dates, lower, upper, alpha=0.2, color="blue")

    # raw NDVI with outliers marked
    ax.scatter(dates, ndvi_series, s=10, color=colors, label="Raw NDVI", zorder=3)

    # gapfilled NDVI
    ax.plot(dates, ndvi_gapfilled, color="black", label="Gapfilled NDVI")

    ax.set_title(title)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.nn.functional.sigmoid(
        -2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10)
    )
    sigmoid_sen_eos = torch.nn.functional.sigmoid(
        -2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10)
    )
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def gapfill_ndvi_2(
    ndvi_series,
    lower,
    upper,
    forecasting=False,
    param_iqr=1.5,
    bottom_q=0.3,
    top_q=0.7,
    smooth_back=28,
    smooth_gap=14,
):
    """
    NDVI gapfilling with optional smoothing in forecasting mode.

    Smoothing window: [t-28, t-14], i.e. only past values, excluding the most recent 14 days.
    """
    # --- normalize ---
    ndvi_series = ndvi_series.astype(float) / 10000.0
    ndvi_series = np.where((ndvi_series > 1) | (ndvi_series < 0), np.nan, ndvi_series)

    n = len(ndvi_series)
    outlier_mask = np.zeros(n, dtype=bool)

    # --------------------
    # Common outlier detection
    # --------------------
    valid_idx = np.where(np.isfinite(ndvi_series))[0]
    if len(valid_idx) < 2:
        if forecasting:
            return ndvi_series.copy(), outlier_mask, np.full(n, np.nan)
        else:
            return ndvi_series.copy(), outlier_mask

    valid_ndvi = ndvi_series[valid_idx]
    valid_upper = upper[valid_idx]
    valid_lower = lower[valid_idx]
    valid_iqr = valid_upper - valid_lower
    median_valid = valid_upper - valid_iqr / 2

    # threshold outliers
    th_hi = valid_upper + param_iqr * valid_iqr
    th_lo = valid_lower - param_iqr * valid_iqr
    is_outlier_threshold = (valid_ndvi > th_hi) | (valid_ndvi < th_lo)

    # slope outliers
    deltas = valid_ndvi - median_valid
    delta_diff = np.diff(deltas)
    if len(delta_diff) > 2:
        q_low, q_hi = np.quantile(delta_diff, [bottom_q, top_q])
        slope_outlier = np.zeros_like(is_outlier_threshold)
        slope_outlier[1:-1] = (
            (delta_diff[1:] > q_hi) | (delta_diff[1:] < q_low)
        ) & (
            (delta_diff[:-1] > q_hi) | (delta_diff[:-1] < q_low)
        )
        is_outlier = is_outlier_threshold & slope_outlier
    else:
        is_outlier = is_outlier_threshold

    outlier_mask[valid_idx] = is_outlier
    ndvi_series[valid_idx[is_outlier]] = np.nan

    # =========================================================
    # Non-sequential mode
    # =========================================================
    if not forecasting:
        ndvi_gapfilled = ndvi_series.copy()
        valid_idx = np.where(np.isfinite(ndvi_gapfilled))[0]
        if len(valid_idx) < 2:
            return ndvi_gapfilled, outlier_mask

        for i in range(len(valid_idx) - 1):
            start, end = valid_idx[i], valid_idx[i + 1]
            if end - start > 1:
                for j, idx in enumerate(range(start + 1, end)):
                    frac = (j + 1) / (end - start)
                    obs_interp = (
                        ndvi_gapfilled[start]
                        + (ndvi_gapfilled[end] - ndvi_gapfilled[start]) * frac
                    )
                    median_val = upper[idx] - (upper[idx] - lower[idx]) / 2
                    diff = median_val - obs_interp
                    weight = frac * (1 - frac)
                    ndvi_gapfilled[idx] = obs_interp + 0.5 * weight * diff

        return ndvi_gapfilled, outlier_mask

    # =========================================================
    # Sequential mode
    # =========================================================
    ndvi_filled = np.full(n, np.nan)
    forecast_only = np.full(n, np.nan)
    last_idx = None

    for t in range(n):
        obs = ndvi_series[t]
        if not np.isnan(obs):
            ndvi_filled[t] = obs
            forecast_only[t] = obs
            if last_idx is not None and t - last_idx > 1:
                for j, idx in enumerate(range(last_idx + 1, t)):
                    frac = (j + 1) / (t - last_idx)
                    obs_interp = (
                        ndvi_filled[last_idx]
                        + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac
                    )
                    median_val = upper[idx] - (upper[idx] - lower[idx]) / 2
                    diff = median_val - obs_interp
                    weight = frac * (1 - frac)
                    val_retro = obs_interp + 0.5 * weight * diff
                    ndvi_filled[idx] = val_retro
            last_idx = t
        else:
            if last_idx is not None:
                med_t = upper[t] - (upper[t] - lower[t]) / 2
                delta_last = ndvi_filled[last_idx] - (
                    upper[last_idx] - (upper[last_idx] - lower[last_idx]) / 2
                )
                forecast_val = med_t + delta_last
                forecast_only[t] = forecast_val

    # =========================================================
    # Apply smoothing with window [t-28, t-14]
    # =========================================================
    smoothed = ndvi_filled.copy()
    for t in range(28,n):
        start = int(t -28)
        end = int(t-14)  # exclude the last 14 days
        smoothed[start:end] = savgol_filter(forecast_only[start:end], window_length=14, polyorder=2) 

    return smoothed, outlier_mask, forecast_only

    