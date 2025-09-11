import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm



def gapfill_ndvi(
    ndvi_series,
    lower,
    upper,
    forecasting=False,
    param_iqr=1.5,
    bottom_q=0.3,
    top_q=0.7,
    return_quantiles = False,
    weight_median = 0.5,
    q_hi = 0.05,
    q_low = -0.05,
    smoothing_method = "savgol",
    window_smoothing = 14,
    sigma = 4,
    frac = 0.2,
    lag_forecast = 14,
    use_tau = False,
    tau = 9999
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
    ndvi_filled = np.full(n, np.nan)
    forecast_only = np.full(n, np.nan)
    smoothed = np.full(n, np.nan)

    # --------------------
    #  outlier detection
    # --------------------
    valid_idx = np.where(np.isfinite(ndvi_series))[0]
    if len(valid_idx) < 2:
        if forecasting:
            return ndvi_series.copy(), outlier_mask, np.full(n, np.nan)
        else:
            return ndvi_series.copy(), outlier_mask
    # window_size = odd integer for local thresholds (use 5 or 7)
    window_size = 5

    valid_ndvi = ndvi_series[valid_idx]
    valid_upper = upper[valid_idx]
    valid_lower = lower[valid_idx]
    valid_iqr = valid_upper - valid_lower
    median_valid = (valid_upper + valid_lower) / 2.0   # equivalent to your formula

    # -----------------------
    # Global threshold band
    # -----------------------
    th_hi = valid_upper + param_iqr * valid_iqr
    th_lo = valid_lower - param_iqr * valid_iqr
    is_outlier_global = (valid_ndvi > th_hi) | (valid_ndvi < th_lo)

    # -----------------------
    # Local rolling threshold (to catch local sharp dips)
    # -----------------------
    n = valid_ndvi.size
    half = window_size // 2
    local_th_hi = np.empty(n)
    local_th_lo = np.empty(n)

    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        window_vals = valid_ndvi[a:b]
        if window_vals.size >= 3:
            q1, q3 = np.quantile(window_vals, [0.25, 0.75])
            iqr_local = q3 - q1
            local_th_hi[i] = q3 + param_iqr * iqr_local
            local_th_lo[i] = q1 - param_iqr * iqr_local
        else:
            # fallback to global band if window too small
            local_th_hi[i] = th_hi[i]
            local_th_lo[i] = th_lo[i]

    is_outlier_local = (valid_ndvi > local_th_hi) | (valid_ndvi < local_th_lo)

    # -----------------------
    # Slope-based detection
    # -----------------------
    deltas = valid_ndvi - median_valid                # shape (n,)
    delta_diff = np.diff(deltas)                      # shape (n-1,)
    slope_outlier = np.zeros(n, dtype=bool)

    if delta_diff.size >= 2:
        q_low, q_hi = np.quantile(delta_diff, [bottom_q, top_q])

        # same-direction extremes: both previous and next step extreme (same sign allowed)
        same_dir = (
            ((delta_diff[1:] > q_hi) | (delta_diff[1:] < q_low))
            & ((delta_diff[:-1] > q_hi) | (delta_diff[:-1] < q_low))
        )  # shape (n-2,)

        # V-shape: previous and next extreme with opposite signs (drop then rise, or rise then drop)
        vshape = (
            ((delta_diff[1:] > q_hi) & (delta_diff[:-1] < q_low))
            | ((delta_diff[1:] < q_low) & (delta_diff[:-1] > q_hi))
        )  # shape (n-2,)

        # single-step *very* extreme: use more extreme quantiles (e.g. 1%/99%)
        q_low_ext, q_hi_ext = np.quantile(delta_diff, [0.01, 0.99])
        single_step = (delta_diff > q_hi_ext) | (delta_diff < q_low_ext)  # shape (n-1,)

        # For a point i (1..n-2) we consider:
        # - same_dir or vshape involving delta_diff[i-1] and delta_diff[i]
        # - OR if either neighboring step is a single very-extreme step
        slope_outlier[1:-1] = (
            same_dir | vshape | (single_step[:-1] | single_step[1:])
        )

        # Optionally check the first/last positions using the single-step rule:
        slope_outlier[0] = single_step[0]  # change from 0->1
        slope_outlier[-1] = single_step[-1]  # change from n-2->n-1
    else:
        # not enough points for 2-step logic; fall back to single-step extremes
        if delta_diff.size == 1:
            slope_outlier[0] = slope_outlier[1] = ((delta_diff[0] > q_hi) | (delta_diff[0] < q_low))

    # -----------------------
    # Final outlier mask (must be slope-outlier AND (global OR local threshold outlier))
    # -----------------------
    is_outlier = slope_outlier & (is_outlier_global | is_outlier_local)

    inside_band = (valid_ndvi >= valid_lower) & (valid_ndvi <= valid_upper)
    is_outlier = is_outlier & ~inside_band

    outlier_mask[valid_idx] = is_outlier
    ndvi_series[valid_idx[is_outlier]] = np.nan

    # --------------------------
    # Gapfilling on full time serie
    # --------------------------
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
                    ndvi_gapfilled[idx] = obs_interp + weight_median * weight * diff

        if not return_quantiles:
            return ndvi_gapfilled, outlier_mask
        else:
            return ndvi_gapfilled, outlier_mask, q_hi, q_low

    else:
        # --- initialization ---
        n = len(ndvi_series)
        outlier_mask = np.zeros(n, dtype=bool)      # final confirmed outliers
        potential_mask = np.zeros(n, dtype=bool)    # tentative potentials
        potential_obs = np.full(n, np.nan)          # store raw obs for potentials
        potential_last_idx = None
        last_idx = None

        ndvi_filled = np.full(n, np.nan)
        forecast_only = np.full(n, np.nan)
        smoothed = np.full(n, np.nan)

        # helper: fill values strictly between start and end (start < end)
        def _fill_segment(start, end):
            if start is None or end is None:
                return
            span = end - start
            if span <= 1:
                return
            # fill indices start+1 .. end-1
            for j, idx in enumerate(range(start + 1, end)):
                frac = (j + 1) / span
                # require endpoints exist
                if not (np.isfinite(ndvi_filled[start]) and np.isfinite(ndvi_filled[end])):
                    continue
                obs_interp = ndvi_filled[start] + (ndvi_filled[end] - ndvi_filled[start]) * frac
                median_val = 0.5 * (upper[idx] + lower[idx])
                diff = median_val - obs_interp
                weight = frac * (1 - frac)
                ndvi_filled[idx] = obs_interp + weight_median * weight * diff
                forecast_only[idx] = ndvi_filled[idx]

        # main loop
        for t in range(n):
            obs = ndvi_series[t]

            if not np.isnan(obs):
                # reset flags for this t
                is_outlier_now = False    # immediate final outlier for t
                local_outlier = False

                # --- model/global band check ---
                median_iqr = param_iqr * ((upper[t] - lower[t]) / 2)
                th_hi = upper[t] + median_iqr
                th_lo = lower[t] - median_iqr

                # --- local backward check (last 5 accepted values) ---
                recent = ndvi_filled[max(0, t - 5):t]
                recent = recent[np.isfinite(recent)]
                if recent.size >= 3:
                    q1, q3 = np.quantile(recent, [0.25, 0.75])
                    iqr_local = q3 - q1
                    local_th_hi = q3 + param_iqr * iqr_local
                    local_th_lo = q1 - param_iqr * iqr_local
                    if obs > local_th_hi or obs < local_th_lo:
                        local_outlier = True

                # inside strict logistic band => always valid
                inside_band = (obs >= lower[t]) and (obs <= upper[t])

                # define whether this obs is a potential (outside bands OR local outlier)
                potential_now = (obs > th_hi or obs < th_lo) or local_outlier
                if inside_band:
                    potential_now = False

                # If it's a potential, do slope test vs last accepted
                if potential_now:
                    # if we have a last accepted point, compare slope dynamics
                    if last_idx is not None and np.isfinite(ndvi_filled[last_idx]):
                        median_last = 0.5 * (upper[last_idx] + lower[last_idx])
                        median_curr = 0.5 * (upper[t] + lower[t])
                        delta_previous = ndvi_filled[last_idx] - median_last
                        delta_current = obs - median_curr          # raw obs here
                        delta_delta = delta_current - delta_previous

                        # immediate slope extreme -> mark as potential (tentative)
                        slope_thresh = 0.5   # you used 0.5 in earlier code; tune if needed
                        if delta_delta > slope_thresh or delta_delta < -slope_thresh:
                            # store as potential; do NOT mark final outlier yet
                            potential_mask[t] = True
                            potential_obs[t] = obs
                            potential_last_idx = t
                            # do NOT write to outlier_mask yet (leave final decision to future)
                        else:
                            # slope not extreme -> accept as normal
                            potential_now = False
                    else:
                        # no last accepted point: keep as potential (can't slope-test)
                        potential_mask[t] = True
                        potential_obs[t] = obs
                        potential_last_idx = t

                # If not potential (or slope didn't trigger), accept the obs now
                if not potential_mask[t]:
                    # accept current observation as final
                    ndvi_filled[t] = obs
                    forecast_only[t] = obs
                    outlier_mask[t] = False

                    # If we have a pending potential candidate, evaluate it now retroactively
                    if potential_last_idx is not None and potential_last_idx > (last_idx if last_idx is not None else -1):
                        p = potential_last_idx
                        # use raw observation at p (from potential_obs or ndvi_series)
                        raw_p = potential_obs[p] if np.isfinite(potential_obs[p]) else ndvi_series[p]
                        median_p = 0.5 * (upper[p] + lower[p])
                        median_t = 0.5 * (upper[t] + lower[t])
                        delta_prev = raw_p - median_p
                        delta_curr = ndvi_filled[t] - median_t
                        delta_delta = delta_curr - delta_prev

                        # confirm vs q_hi/q_low (function params)
                        if (delta_delta > q_hi) or (delta_delta < q_low):
                            # confirmed outlier
                            outlier_mask[p] = True
                            # remove potential bookkeeping
                            potential_mask[p] = False
                            potential_obs[p] = np.nan
                        else:
                            # accept candidate p -> retro-fill and fill segments
                            outlier_mask[p] = False
                            potential_mask[p] = False
                            potential_obs[p] = np.nan

                            # ensure endpoint values exist for interpolation:
                            # if last_idx exists we will fill (last_idx -> p) and (p -> t)
                            # if last_idx is None, we set ndvi_filled[p] to raw_p first
                            if last_idx is None or not np.isfinite(ndvi_filled[last_idx]):
                                ndvi_filled[p] = raw_p
                                forecast_only[p] = raw_p
                            else:
                                # compute retro value for p biased toward median
                                span = t - last_idx
                                if span > 0:
                                    frac_p = (p - last_idx) / span
                                    obs_interp_p = ndvi_filled[last_idx] + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac_p
                                    median_val_p = 0.5 * (upper[p] + lower[p])
                                    diff_p = median_val_p - obs_interp_p
                                    weight_p = frac_p * (1 - frac_p)
                                    ndvi_filled[p] = obs_interp_p + weight_median * weight_p * diff_p
                                    forecast_only[p] = ndvi_filled[p]

                            # fill surrounding segments (if possible)
                            _fill_segment(last_idx, p)
                            _fill_segment(p, t)

                            # after retro fill we have filled up to t, so set last_idx = t
                            last_idx = t

                    else:
                        # no pending potential -> just fill gap last_idx -> t
                        if last_idx is not None and (t - last_idx) > 1:
                            _fill_segment(last_idx, t)
                        last_idx = t

                else:
                    # potential_mask[t] True: we keep current obs as tentative but DO NOT
                    # write to final outlier_mask yet. We store raw observation in potential_obs.
                    potential_obs[t] = obs
                    # (do not set ndvi_filled[t] or outlier_mask[t] at this point)

            else:
                # missing observation -> forecast only
                if last_idx is not None and np.isfinite(ndvi_filled[last_idx]):
                    med_t = 0.5 * (upper[t] + lower[t])
                    delta_last = ndvi_filled[last_idx] - (0.5 * (upper[last_idx] + lower[last_idx]))
                    if use_tau:
                        multiplicative_factor = math.exp(-math.log(2) * (t - last_idx) / tau)
                    else:
                        multiplicative_factor = 1.0
                    forecast_val = med_t + multiplicative_factor * delta_last
                    forecast_only[t] = forecast_val
                # else remain NaN

            # smoothing stage (unchanged)
            if t > 1 + lag_forecast + window_smoothing:
                a = t - lag_forecast - window_smoothing
                b = t - lag_forecast + 1
                data_ndvi = ndvi_filled[a:b]
                data_forecast = forecast_only[a:b]
                data_to_smooth = np.where(np.isnan(data_ndvi), data_forecast, data_ndvi)

                if smoothing_method == "savgol":
                    # ensure window_smoothing odd and valid
                    wl = window_smoothing if (window_smoothing % 2 == 1) else (window_smoothing + 1)
                    wl = min(wl, len(data_to_smooth) if (len(data_to_smooth) % 2 == 1) else len(data_to_smooth) - 1)
                    if wl >= 3:
                        smoothed[a:b] = savgol_filter(data_to_smooth, window_length=wl, polyorder=2)
                elif smoothing_method == "low_pass":
                    smoothed[a:b] = gaussian_filter1d(data_to_smooth, sigma=sigma)
                elif smoothing_method == "loess":
                    index = np.arange(len(data_to_smooth))
                    loess_smooth = sm.nonparametric.lowess(data_to_smooth, index, frac=frac)
                    smoothed[a:b] = loess_smooth[:, 1]

        return ndvi_filled, outlier_mask, forecast_only, smoothed



def plot_results(title, ndvi_series, ndvi_gapfilled, outlier_arr, lower, upper, dates, plot_points = False, show_iqr = False,param_iqr = 1.5, save_path=None):
    """
    Plot NDVI time series for a pixel:
      - raw NDVI with outliers highlighted
      - logistic model bounds
      - gapfilled NDVI
    """
    colors = np.where(outlier_arr, "red", "green")

    fig, ax = plt.subplots(figsize=(12, 6))

    # logistic bounds
    ax.plot(dates, lower, label="Lower Bound", color="red", alpha=0.7)
    ax.plot(dates, upper, label="Upper Bound", color="red", alpha=0.7)
    ax.fill_between(dates, lower, upper, alpha=0.2, color="red")

    # raw NDVI with outliers marked
    ax.scatter(dates, ndvi_series, s=10, color=colors, label="Raw NDVI", zorder=3)

    # gapfilled NDVI
    if not plot_points:
        ax.plot(dates, ndvi_gapfilled, color="black", label="Gapfilled NDVI")
    else:
        ax.scatter(dates, ndvi_gapfilled, color="black", label="Gapfilled NDVI")

    if show_iqr == True and param_iqr !=1:
        iqr = (upper - lower) /2
        iqr_upper = upper + iqr * param_iqr
        iqr_lower = lower - iqr * param_iqr
        ax.plot(dates, iqr_upper, linestyle='--', color='blue', linewidth=2, label="IQR")
        ax.plot(dates, iqr_lower, linestyle='--', color='blue', linewidth=2)


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


# this is just a extra safety backup

"""
            if not outlier:
                
                ndvi_filled[t] = obs
                forecast_only[t] = obs
                last_idx = t

                if potential_previous == True:
                    
                    delta_previous =  (ndvi_filled[potential_last_idx] - upper[potential_last_idx] - (upper[potential_last_idx] - lower[potential_last_idx]) / 2)
                    delta_current = (ndvi_filled[t] - upper[t] - (upper[t] - lower[t]) / 2)
                    delta_delta = delta_current - delta_previous

                    if delta_delta > 0.05 or delta_delta < -0.05:
                        outlier_mask[potential_last_idx] = True

                    else:

                        outlier_mask[potential_last_idx] = False
                        last_idx = potential_last_idx

                    # Fill the potential outlier with interpolation between last valid and current obs
                    if last_idx is not None:
                        frac = (potential_last_idx - last_idx) / (t - last_idx)
                        obs_interp = (
                            ndvi_filled[last_idx]
                            + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac
                        )
                        median_val = upper[potential_last_idx] - (upper[potential_last_idx] - lower[potential_last_idx]) / 2
                        diff = median_val - obs_interp
                        weight = frac * (1 - frac)
                        val_retro = obs_interp + weight_median * weight * diff
                        ndvi_filled[potential_last_idx] = val_retro
                    
                    # now safe to update last_idx
                    last_idx = potential_last_idx


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
                        val_retro = obs_interp + weight_median * weight * diff
                        ndvi_filled[idx] = val_retro
                        
                """

# almost correct

"""    valid_ndvi = ndvi_series[valid_idx]
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
        #print((delta_diff > q_hi) | (delta_diff < q_low))
        slope_outlier = np.zeros_like(is_outlier_threshold)
        slope_outlier[1:-1] = (
            ((delta_diff[1:] > q_hi) | (delta_diff[1:] < q_low))
            & ((delta_diff[:-1] > q_hi) | (delta_diff[:-1] < q_low))
        ) | (
            # opposite-direction "V-shape" check
            ((delta_diff[1:] > q_hi) & (delta_diff[:-1] < q_low))
            | ((delta_diff[1:] < q_low) & (delta_diff[:-1] > q_hi))
        )
        is_outlier = is_outlier_threshold & slope_outlier
    else:
        is_outlier = is_outlier_threshold

    outlier_mask[valid_idx] = is_outlier
    ndvi_series[valid_idx[is_outlier]] = np.nan"""


# continour integration almost working

"""# =========================================================
    # Sequential mode (like sequential_gapfill with forecast_only)
    # =========================================================
    ndvi_filled = np.full(n, np.nan)
    forecast_only = np.full(n, np.nan)
    smoothed = np.full(n, np.nan)
    potential_previous = False
    last_idx = None

    for t in range(n):
        
        obs = ndvi_series[t]

        if not np.isnan(obs):
            outlier = False

            # oultier detection
            median_iqr = param_iqr * ((upper[t] - lower[t]) / 2)

            th_hi = upper[t] + median_iqr
            th_lo = lower[t] - median_iqr

            if obs > th_hi  or obs < th_lo:

                if last_idx is not None:
                    delta_previous =  (ndvi_filled[last_idx] - upper[last_idx] - (upper[last_idx] - lower[last_idx]) / 2)
                    delta_current = (obs- upper[t] - (upper[t] - lower[t]) / 2)
                    delta_delta = delta_current - delta_previous

                    if delta_delta > 0.5 or delta_delta < -0.5:
                        potential_previous = True
                        potential_last_idx = t
                        outlier = True
                    
            if not outlier:
                # accept current observation
                ndvi_filled[t] = obs
                forecast_only[t] = obs

                if potential_previous:
                    # check if the earlier potential outlier should be kept or rejected
                    delta_previous = (
                        ndvi_filled[potential_last_idx]
                        - upper[potential_last_idx]
                        - (upper[potential_last_idx] - lower[potential_last_idx]) / 2
                    )
                    delta_current = (
                        ndvi_filled[t]
                        - upper[t]
                        - (upper[t] - lower[t]) / 2
                    )
                    delta_delta = delta_current - delta_previous

                    if delta_delta > q_hi or delta_delta < q_low:
                        # confirm it was an outlier
                        outlier_mask[potential_last_idx] = True
                    else:
                        # accept it, fill its value by interpolation
                        outlier_mask[potential_last_idx] = False
                        if last_idx is not None:
                            frac = (potential_last_idx - last_idx) / (t - last_idx)
                            obs_interp = (
                                ndvi_filled[last_idx]
                                + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac
                            )
                            median_val = upper[potential_last_idx] - (upper[potential_last_idx] - lower[potential_last_idx]) / 2
                            diff = median_val - obs_interp
                            weight = frac * (1 - frac)
                            val_retro = obs_interp + weight_median * weight * diff
                            ndvi_filled[potential_last_idx] = val_retro
                            forecast_only[potential_last_idx] = val_retro
                        # move last_idx forward
                        last_idx = potential_last_idx
                else:
                    last_idx = t

                # retroactive interpolation for missing values between last_idx and t
                if last_idx is not None and t - last_idx > 1:
                    for j, idx in enumerate(range(last_idx + 1, t)):
                        if not np.isnan(ndvi_filled[idx]):  # skip if already filled
                            continue
                        frac = (j + 1) / (t - last_idx)
                        obs_interp = (
                            ndvi_filled[last_idx]
                            + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac
                        )
                        median_val = upper[idx] - (upper[idx] - lower[idx]) / 2
                        diff = median_val - obs_interp
                        weight = frac * (1 - frac)
                        val_retro = obs_interp + weight_median * weight * diff
                        ndvi_filled[idx] = val_retro

                outlier_mask[t] = outlier

        else:
                # missing obs -> forecast only
                if last_idx is not None and not np.isnan(ndvi_filled[last_idx]):
                    med_t = upper[t] - (upper[t] - lower[t]) / 2
                    delta_last = ndvi_filled[last_idx] - (
                        upper[last_idx] - (upper[last_idx] - lower[last_idx]) / 2
                    )
                    if use_tau:
                        multiplicative_factor = math.exp(-math.log(2) * (t - last_idx) / tau)
                    else:
                        multiplicative_factor = 1
                    forecast_val = med_t + multiplicative_factor * delta_last
                    forecast_only[t] = forecast_val
                # if last_idx is None or NaN, do nothing (stay NaN)

            # smoothing stage
        if t >  1 + lag_forecast + window_smoothing:
                data_ndvi = ndvi_filled[t - lag_forecast - window_smoothing: t - lag_forecast +1]
                data_forecast = forecast_only[t - lag_forecast - window_smoothing: t - lag_forecast+1]
                data_to_smooth = np.where(np.isnan(data_ndvi), data_forecast, data_ndvi)

                if smoothing_method == "savgol":

                    smoothed[t - lag_forecast - window_smoothing : t - lag_forecast +1] = savgol_filter(
                    data_to_smooth, window_length=window_smoothing, polyorder=2
                    )
                elif smoothing_method == "low_pass":
                    smoothed[t - lag_forecast - window_smoothing : t - lag_forecast +1] = gaussian_filter1d(
                    data_to_smooth, sigma = sigma
                    ) 
                elif smoothing_method == "loess":
                    index = np.arange(len(data_to_smooth))
                    loess_smooth = sm.nonparametric.lowess(data_to_smooth, index, frac=frac)
                    smoothed[t - lag_forecast - window_smoothing : t - lag_forecast +1] = loess_smooth[:, 1]

    return ndvi_filled, outlier_mask, forecast_only, smoothed"""

# updated

"""" else:
        # --- initialization ---
        potential_flag = False
        potential_last_idx = None
        last_idx = None
        n = len(ndvi_series)
        outlier_mask = np.zeros(n, dtype=bool)
        ndvi_filled = np.full(n, np.nan)
        forecast_only = np.full(n, np.nan)
        smoothed = np.full(n, np.nan)

        # small helper to fill a segment (start < end). Fills indices start+1 .. end-1
        def _fill_segment(start, end):
            if start is None or end is None:
                return
            span = end - start
            if span <= 0:
                return
            for j, idx in enumerate(range(start + 1, end)):
                frac = (j + 1) / span
                # if endpoints are not finite, skip
                if not (np.isfinite(ndvi_filled[start]) and np.isfinite(ndvi_filled[end])):
                    continue
                obs_interp = ndvi_filled[start] + (ndvi_filled[end] - ndvi_filled[start]) * frac
                median_val = 0.5 * (upper[idx] + lower[idx])
                diff = median_val - obs_interp
                weight = frac * (1 - frac)
                val_interp = obs_interp + weight_median * weight * diff
                ndvi_filled[idx] = val_interp
                forecast_only[idx] = val_interp

        for t in range(n):
            obs = ndvi_series[t]

            if not np.isnan(obs):
                outlier = False
                local_outlier = False

                # --- global (model) band thresholds ---
                median_iqr = param_iqr * ((upper[t] - lower[t]) / 2)
                th_hi = upper[t] + median_iqr
                th_lo = lower[t] - median_iqr

                # --- local backward check (last 5 accepted values, causal) ---
                recent = ndvi_filled[max(0, t - 5):t]
                recent = recent[np.isfinite(recent)]
                if recent.size >= 3:
                    q1, q3 = np.quantile(recent, [0.25, 0.75])
                    iqr_local = q3 - q1
                    local_th_hi = q3 + param_iqr * iqr_local
                    local_th_lo = q1 - param_iqr * iqr_local
                    if obs > local_th_hi or obs < local_th_lo:
                        local_outlier = True

                # --- define potential candidate (outside model band OR local outlier) ---
                potential_outlier = (obs > th_hi or obs < th_lo) or local_outlier

                # inside [lower, upper] = always valid => clear potential flag
                if (obs >= lower[t]) and (obs <= upper[t]):
                    potential_outlier = False

                # --- if potential, store it and test slope vs last_idx if possible ---
                if potential_outlier:
                    # if we have a last accepted point, compute slope delta vs that point
                    if last_idx is not None and np.isfinite(ndvi_filled[last_idx]):
                        median_last = 0.5 * (upper[last_idx] + lower[last_idx])
                        median_curr = 0.5 * (upper[t] + lower[t])
                        delta_previous = ndvi_filled[last_idx] - median_last
                        delta_current = obs - median_curr              # use raw observation here!
                        delta_delta = delta_current - delta_previous

                        if delta_delta > 0.5 or delta_delta < -0.5:
                            # mark as potential; store candidate observation so we can evaluate later
                            potential_last_idx = t
                            potential_flag = True
                            ndvi_filled[t] = obs          # temporarily store observed value
                            forecast_only[t] = obs
                            outlier_mask[t] = True       # tentative
                            outlier = True
                    else:
                        # no last accepted point: keep as potential candidate (store obs)
                        potential_last_idx = t
                        potential_flag = True
                        ndvi_filled[t] = obs
                        forecast_only[t] = obs
                        outlier_mask[t] = True
                        outlier = True

                # --- if not flagged as an immediate outlier, accept the point ---
                if not outlier:
                    ndvi_filled[t] = obs
                    forecast_only[t] = obs

                    # If we had an earlier potential candidate, evaluate it now against current accepted point t
                    if potential_flag and potential_last_idx is not None and (last_idx is None or potential_last_idx > last_idx):
                        p = potential_last_idx
                        # compute deltas using raw obs at p (ndvi_series) and the accepted current point
                        median_p = 0.5 * (upper[p] + lower[p])
                        median_t = 0.5 * (upper[t] + lower[t])
                        delta_previous = (ndvi_series[p] - median_p)   # raw observation at p
                        delta_current = (ndvi_filled[t] - median_t)   # accepted value at t
                        delta_delta = delta_current - delta_previous

                        if delta_delta > q_hi or delta_delta < q_low:
                            # confirm it as outlier
                            outlier_mask[p] = True
                            # revert temporary storage if we set it earlier
                            ndvi_filled[p] = np.nan
                            forecast_only[p] = np.nan
                        else:
                            # accept it and retroactively fill values
                            outlier_mask[p] = False
                            if last_idx is not None and t - last_idx > 0:
                                # compute retro value for p (interpolation bias toward logistic median)
                                frac_p = (p - last_idx) / (t - last_idx)
                                obs_interp_p = ndvi_filled[last_idx] + (ndvi_filled[t] - ndvi_filled[last_idx]) * frac_p
                                median_val_p = 0.5 * (upper[p] + lower[p])
                                diff_p = median_val_p - obs_interp_p
                                weight_p = frac_p * (1 - frac_p)
                                val_retro_p = obs_interp_p + weight_median * weight_p * diff_p
                                ndvi_filled[p] = val_retro_p
                                forecast_only[p] = val_retro_p

                                # now fill last_idx -> p and p -> t segments
                                _fill_segment(last_idx, p)
                                _fill_segment(p, t)
                            # move last_idx to p (we retro-filled up to p)
                            last_idx = p

                        # reset the potential candidate
                        potential_flag = False
                        potential_last_idx = None

                    else:
                        # regular accept: fill any gap last_idx -> t
                        if last_idx is not None and (t - last_idx) > 1:
                            _fill_segment(last_idx, t)
                        last_idx = t

                    outlier_mask[t] = outlier

            else:
                # missing obs -> forecast only
                if last_idx is not None and not np.isnan(ndvi_filled[last_idx]):
                    med_t = 0.5 * (upper[t] + lower[t])
                    delta_last = ndvi_filled[last_idx] - (0.5 * (upper[last_idx] + lower[last_idx]))
                    if use_tau:
                        multiplicative_factor = math.exp(-math.log(2) * (t - last_idx) / tau)
                    else:
                        multiplicative_factor = 1.0
                    forecast_val = med_t + multiplicative_factor * delta_last
                    forecast_only[t] = forecast_val
                # else stays NaN

            # --- smoothing stage (unchanged logic, kept here) ---
            if t > 1 + lag_forecast + window_smoothing:
                a = t - lag_forecast - window_smoothing
                b = t - lag_forecast + 1
                data_ndvi = ndvi_filled[a:b]
                data_forecast = forecast_only[a:b]
                data_to_smooth = np.where(np.isnan(data_ndvi), data_forecast, data_ndvi)

                if smoothing_method == "savgol":
                    # ensure window_smoothing is odd and <= len(data_to_smooth)
                    wl = window_smoothing if (window_smoothing % 2 == 1) else (window_smoothing + 1)
                    wl = min(wl, len(data_to_smooth) if len(data_to_smooth) % 2 == 1 else len(data_to_smooth) - 1)
                    if wl >= 3:
                        smoothed[a:b] = savgol_filter(data_to_smooth, window_length=wl, polyorder=2)
                elif smoothing_method == "low_pass":
                    smoothed[a:b] = gaussian_filter1d(data_to_smooth, sigma=sigma)
                elif smoothing_method == "loess":
                    index = np.arange(len(data_to_smooth))
                    loess_smooth = sm.nonparametric.lowess(data_to_smooth, index, frac=frac)
                    smoothed[a:b] = loess_smooth[:, 1]

        return ndvi_filled, outlier_mask, forecast_only, smoothed"""