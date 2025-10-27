# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/03_test_with_df.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_df_function.log 


import os
import shutil
import math
import re
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import zarr
import torch
import statsmodels.api as sm

# -----------------------------
# Config / paths (edit these)
# -----------------------------
INPUT_ZARR = "/data_2/scratch/francesco/zarr_demo_daily/"   # source dataset (read-only)
DST = "/data_2/scratch/francesco/zarr_demo_daily_output"    # daily working copy (we will write here)
# If DST exists and you want to refresh from INPUT_ZARR, uncomment the copyline below
shutil.copytree(INPUT_ZARR, DST, dirs_exist_ok=True)

# Which pixels to process (example: 3 random)
np.random.seed(42)
pixels = np.random.choice(np.arange(1_000_000), size=3, replace=False)

# -----------------------------
# Helper: parse Zarr date entries
# -----------------------------
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

def zarr_date_to_date(zarr_date):
    """Convert bytes / np.datetime64 / datetime / date to python.date.
       Raise TypeError for unknown types"""
    if isinstance(zarr_date, np.ndarray) and zarr_date.shape == ():
        zarr_date = zarr_date[()]
    if isinstance(zarr_date, bytes):
        return datetime.strptime(zarr_date.decode("utf-8"), "%Y-%m-%d").date()
    elif isinstance(zarr_date, np.bytes_):
        return datetime.strptime(bytes(zarr_date).decode("utf-8"), "%Y-%m-%d").date()
    elif isinstance(zarr_date, np.datetime64):
        return pd.to_datetime(zarr_date).date()
    elif isinstance(zarr_date, datetime):
        return zarr_date.date()
    elif isinstance(zarr_date, date):
        return zarr_date
    else:
        raise TypeError(f"Unknown date type: {type(zarr_date)}")

def date_to_zarr_bytes(d: date):
    return d.strftime("%Y-%m-%d").encode("utf-8")

# -----------------------------
# Double logistic (numpy wrapper)
# -----------------------------
def double_logistic_function(t, params):
    """Torch implementation returning numpy array like in your scripts.
       t: 1D tensor of time [0..1], params: torch tensor shape (n,6) or (1,6)"""
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

def double_logistic_function_numpy(single_date, params):
    """Compute double-logistic at a single pandas Timestamp or python.date.
       Returns a numpy scalar (0-d numpy array) to match .item() usage in your function."""
    # Convert date -> day of year as fraction
    if isinstance(single_date, (pd.Timestamp, datetime)):
        doy = single_date.timetuple().tm_yday
    elif isinstance(single_date, date):
        doy = single_date.timetuple().tm_yday
    else:
        # if it's index-like (numpy datetime64) try pd.to_datetime
        try:
            dt = pd.to_datetime(single_date)
            doy = dt.timetuple().tm_yday
        except Exception:
            raise TypeError("Unsupported date type for double_logistic_function_numpy")
    t = torch.as_tensor(np.atleast_1d(doy) / 365.0, dtype=torch.float32)
    # ensure params shape (1,6)
    if isinstance(params, np.ndarray):
        p = torch.as_tensor(params, dtype=torch.float32)
    else:
        p = params
    if p.ndim == 1:
        p = p[None, :]
    val = double_logistic_function(t, p).squeeze().numpy()
    # val may be array (len 1) or scalar -- return numpy scalar
    return np.asarray(val).reshape(())

# -----------------------------
# Paste your ndvi_continous_final_2() EXACTLY as provided
# -----------------------------
# (I paste it here unchanged — verbatim from the version you gave)
def ndvi_continous_final_2(
    df,
    date,
    last_date,
    last_potential_date,
    deltas_arr,
    dates_delta_arr,
    params_upper,
    params_lower,
    y_delta_l=0.02,
    y_delta_h=0.02,
    y_iqr=0.1,
    r_delta_l=-0.1,
    r_delta_h=0.1,
    r_iqr=0.3,
    tau=14,
    smoothing_values = 7,
    use_real_idx = False,
    frac = 1,
    latency = [],
    current_date_latency = [],
):

    if smoothing_values % 2 == 1:
        middle = int((smoothing_values + 1) / 2 -1)
    else:
        middle = int(smoothing_values / 2 -1)

    obs = df.at[date, "ndvi"]

    # thresholds
    upper = double_logistic_function_numpy(date, params_upper).item()
    lower = double_logistic_function_numpy(date, params_lower).item()
    median_t = 0.5 * (upper + lower)

    df.at[date, "upper"] = upper
    df.at[date, "lower"] = lower

    if np.isfinite(obs) and obs > 0:

        inside_band = (obs >= lower) and (obs <= upper)

        outlier = False
        potential_outlier = False

        if not inside_band:

            ratio = abs(obs - median_t)
            potential_now = ratio > y_iqr
            df.at[date, "ratio"] = ratio


            delta_delta = 0

            if last_date is not None and np.isfinite(df.at[last_date, "ndvi"]):

                last_upper =  double_logistic_function_numpy(last_date, params_upper).item()
                last_lower =  double_logistic_function_numpy(last_date, params_lower).item()
                median_last = 0.5 * (last_upper + last_lower)
                delta_prev = df.at[last_date, "ndvi"] - median_last
                delta_curr = obs - median_t
                delta_delta = delta_curr - delta_prev
                df.at[date, "delta_delta"] = delta_delta



            # rules
            extreme_outlier = (
                (delta_delta > r_delta_h) or (delta_delta < r_delta_l) or (ratio > r_iqr)
            )
            potential_outlier = (
                ((delta_delta > y_delta_h) or (delta_delta < y_delta_l)) and potential_now
            )

            if extreme_outlier:
                df.at[date, "outlier"] = True
                df.at[date, "deltas"] = np.nan
                outlier = True
            
        if not outlier:

            if not potential_outlier:

                # confirmed obs
                delta_curr = obs - median_t
                df.at[date, "outlier"] = False
                df.at[date, "deltas"] = delta_curr

                deltas_arr.append(delta_curr)
                dates_delta_arr.append(date)

                if len(deltas_arr) > smoothing_values:
                    deltas_arr = deltas_arr[-smoothing_values:]
                    dates_delta_arr = dates_delta_arr[-smoothing_values:]

            # check for potential outlier
            if last_potential_date is not None:

                last_upper =  double_logistic_function_numpy(last_potential_date, params_upper).item()
                last_lower =  double_logistic_function_numpy(last_potential_date, params_lower).item()
                median_potential_last = 0.5 * (last_upper + last_lower)

                # read data
                delta_potential_prev = df.at[last_potential_date, "ndvi"] - median_potential_last
                delta_potential = delta_potential_prev - median_potential_last

                # read data
                delta_delta_p = median_t - delta_potential

                potential_outlier_2 = (delta_delta_p > y_delta_h) or (delta_delta_p < y_delta_l)

                if  potential_outlier_2:

                    # confirm outlier
                    df.at[last_potential_date, "outlier"] = True
                    df.at[last_potential_date, "deltas"] = np.nan

                else:

                    # accept as true value
                    df.at[last_potential_date, "outlier"] = False
                    df.at[date, "outlier"] = False
                    potential_outlier = False

                    # insert between current date and last known obs
                    deltas_arr.insert(len(dates_delta_arr) -1, delta_potential)
                    dates_delta_arr.insert(len(dates_delta_arr) -1, last_potential_date)

                                    
                    # perform smoothing from last_known date to last_pot date
                    if len(deltas_arr) > smoothing_values:
                        deltas_arr = deltas_arr[-smoothing_values:]
                        dates_delta_arr = dates_delta_arr[-smoothing_values:]

                    if len(deltas_arr) == smoothing_values:

                        # Convert list of dates to DatetimeIndex
                        dates_delta_arr_dt = pd.to_datetime(dates_delta_arr)

                        if use_real_idx:
                            idx = (dates_delta_arr_dt - dates_delta_arr_dt[0]).days
                        else:
                            idx = np.arange(len(dates_delta_arr_dt))

                        # LOESS smoothing over the full window
                        loess = sm.nonparametric.lowess(deltas_arr, idx, frac = frac, return_sorted=True)

                        start_date = dates_delta_arr_dt[middle -1 ]
                        end_date = dates_delta_arr_dt[middle]
                        dates_interp = pd.date_range(start=start_date, end=end_date, freq='D')
                        dates_interp = dates_interp.intersection(df.index)

                        if use_real_idx:
                            x_interp = (dates_interp - dates_delta_arr_dt[0]).days
                        else:
                            x_interp = np.linspace(idx[middle -1], idx[middle], len(dates_interp))

                        # Interpolate LOESS values
                        y_interp = np.interp(x_interp, loess[:, 0], loess[:, 1])

                        # Assign smoothed values
                        df.loc[dates_interp, "delta_smoothed"] = y_interp

                    # perform deltas L1
                    if last_date is not None:

                        last_upper =  double_logistic_function_numpy(last_date, params_upper).item()
                        last_lower =  double_logistic_function_numpy(last_date, params_lower).item()
                        median_last = 0.5 * (last_upper + last_lower)
                        delta_prev = df.at[last_date, "ndvi"] - median_last
                        
                        days_diff = (date - last_date).days
                        L1_deltas = np.linspace(delta_prev, delta_curr, num = days_diff +1)
                        df.loc[last_date:date, "deltas_L1"] = L1_deltas
                    
                
                # clean the last_potential date
                if potential_outlier:
                    last_potential_date = date
                else:
                    last_potential_date = None

            else:

                # Write data: perform smoothing from last_date to current date
                if len(dates_delta_arr) == smoothing_values:

                    # in case of extreme events (fire, storm NOT drought), ignore iqr
                    if np.sum(np.array(deltas_arr) < -0.2) >= 5:
                        # retrieve the original NDVI
                        dates_delta_arr_dt = pd.to_datetime(dates_delta_arr)
                        ndvi_values = df.loc[dates_delta_arr_dt, "ndvi"]

                        # perform the smoothing
                        idx = np.arange(len(dates_delta_arr_dt))

                        # LOESS smoothing over the full window
                        loess = sm.nonparametric.lowess(ndvi_values, idx, frac=frac, it = 7,return_sorted=True)

                        # Dates to interpolate: from second-to-last to last, inclusive
                        start_date = dates_delta_arr_dt[middle -1 ]
                        end_date = dates_delta_arr_dt[middle]
                        dates_interp = pd.date_range(start=start_date, end=end_date, freq='D')
                        dates_interp = dates_interp.intersection(df.index)

                        x_interp = np.linspace(idx[middle -1], idx[middle], len(dates_interp))

                        # Interpolate LOESS values
                        y_interp = np.interp(x_interp, loess[:, 0], loess[:, 1])
                        # reframe the values so that do not follow the iqr
                        df.loc[dates_interp, "use_delta"] = False


                    else:

                        # Convert list of dates to DatetimeIndex
                        dates_delta_arr_dt = pd.to_datetime(dates_delta_arr)

                        if use_real_idx:
                            idx = (dates_delta_arr_dt - dates_delta_arr_dt[0]).days
                        else:
                            idx = np.arange(len(dates_delta_arr_dt))

                        # LOESS smoothing over the full window
                        loess = sm.nonparametric.lowess(deltas_arr, idx, frac=frac, it = 3,return_sorted=True)

                        # Dates to interpolate: from second-to-last to last, inclusive
                        start_date = dates_delta_arr_dt[middle -1 ]
                        end_date = dates_delta_arr_dt[middle]
                        dates_interp = pd.date_range(start=start_date, end=end_date, freq='D')
                        dates_interp = dates_interp.intersection(df.index)

                        if use_real_idx:
                            x_interp = (dates_interp - dates_delta_arr_dt[0]).days
                        else:
                            x_interp = np.linspace(idx[middle -1], idx[middle], len(dates_interp))

                        # Interpolate LOESS values
                        y_interp = np.interp(x_interp, loess[:, 0], loess[:, 1])

                        std_y = np.std(y_interp)

                        if std_y > 0.015:

                            if std_y <= 0.03:
                                window = 3
                            else:
                                window = 5

                            y_interp = pd.Series(y_interp).rolling(window=window, center=True, min_periods=1).mean().values

                    # Assign smoothed values
                    df.loc[dates_interp, "delta_smoothed"] = y_interp
                    df.loc[dates_interp, "check_std"] = np.std(y_interp)

                    days_diff = (date - dates_delta_arr[3]).days
                    latency.append(days_diff)
                    current_date_latency.append(date)

            # perform deltas L1
            if last_date is not None:

                last_upper =  double_logistic_function_numpy(last_date, params_upper).item()
                last_lower =  double_logistic_function_numpy(last_date, params_lower).item()
                median_last = 0.5 * (last_upper + last_lower)
                delta_prev = df.at[last_date, "ndvi"] - median_last
                        
                days_diff = (date - last_date).days
                L1_deltas = np.linspace(delta_prev, delta_curr, num = days_diff +1)
                df.loc[last_date:date, "deltas_L1"] = L1_deltas
            # update the last known date
            last_date = date

        else:
            # esitmate NDVI based on last obs
            if last_date is not None and np.isfinite(df.at[last_date, "ndvi"]):

                last_upper =  double_logistic_function_numpy(last_date, params_upper).item()
                last_lower =  double_logistic_function_numpy(last_date, params_lower).item()
                median_last = 0.5 * (last_upper + last_lower)

                # read data
                delta_prev = df.at[last_date, "ndvi"] - median_last
                days_diff = (date - last_date).days

                decrease_factor = math.exp(-math.log(2) * (days_diff / tau))
                forecast_val = median_t + delta_prev * decrease_factor

                # write data
                df.at[date, "forecast"] = forecast_val

    else:

        df.at[date, "outlier"] = True
        # esitmate NDVI based on last obs
        if last_date is not None and np.isfinite(df.at[last_date, "ndvi"]):

            last_upper =  double_logistic_function_numpy(last_date, params_upper).item()
            last_lower =  double_logistic_function_numpy(last_date, params_lower).item()
            median_last = 0.5 * (last_upper + last_lower)

            # read data
            delta_prev = df.at[last_date, "ndvi"] - median_last
            days_diff = (date - last_date).days

            decrease_factor = math.exp(-math.log(2) * (days_diff / tau))
            forecast_val = median_t + delta_prev * decrease_factor

            # write data
            df.at[date, "forecast"] = forecast_val

    return last_date, last_potential_date, deltas_arr, dates_delta_arr, latency, current_date_latency

# -----------------------------
# Open destination Zarr (r+)
# -----------------------------
root = zarr.open_group(DST, mode="r+")
ndvi_zarr = root["ndvi"]
dates_zarr = root["dates"]
params_lower_zarr = root["params"]["params_lower"]
params_upper_zarr = root["params"]["params_upper"]
last_dates_zarr = root["last_dates"]  # shape (8, n_pixels) dtype=S10

# -----------------------------
# Parse all dates from Zarr into python dates list
# -----------------------------
dates = []
for i in range(dates_zarr.shape[0]):
    d_arr = dates_zarr.get_basic_selection((i,))
    d_val = d_arr[()] if isinstance(d_arr, np.ndarray) and d_arr.shape == () else d_arr[0]
    d_dt = _parse_zarr_date(d_val)
    if pd.notna(d_dt):
        dates.append(d_dt.date())
dates = sorted(list(set(dates)))
if not dates:
    raise ValueError("No valid dates found in Zarr dataset.")
base_date = dates[0]
n_days = len(dates)

# -----------------------------
# Utility to reconstruct df for a pixel up to current day index
# -----------------------------
def reconstruct_df_for_pixel(pixel_idx, day_index):
    """
    Build a DataFrame indexed by daily dates from base_date..current day,
    with column 'ndvi' (float [0..1] or nan) derived from ndvi_zarr up to day_index.
    Additional columns used by ndvi_continous_final_2 are created and initialized.
    """
    idx_dates = pd.date_range(start=base_date, periods=day_index + 1, freq="D")  # inclusive
    df = pd.DataFrame(index=idx_dates)
    # read ndvi values for this pixel up to day_index
    raw = ndvi_zarr[: day_index + 1, pixel_idx].astype(np.int32)
    ndvi_vals = raw / 10000.0
    # convert sentinel values outside [0,1] to nan
    ndvi_vals = ndvi_vals.astype(float)
    ndvi_vals[(ndvi_vals <= 0) | (ndvi_vals >= 1)] = np.nan
    df["ndvi"] = ndvi_vals
    # create columns used by function
    df["forecast"] = np.nan
    df["upper"] = np.nan
    df["lower"] = np.nan
    df["gapfilled"] = np.nan
    df["outlier"] = True
    df["delta_smoothed"] = np.nan
    df["idx"] = np.arange(len(df))
    df["deltas_L1"] = np.nan
    df["ratio"] = np.nan
    df["delta_delta"] = np.nan
    df["use_delta"] = True
    df["check_std"] = np.nan
    df["deltas"] = np.nan
    return df

# -----------------------------
# Utility to reconstruct deltas_arr & dates_delta_arr from df history
# -----------------------------
def reconstruct_deltas_from_df(df, params_lower, params_upper, smoothing_values):
    """
    For all dates in df where df['ndvi'] is finite, compute delta = ndvi - median(date).
    Return last 'smoothing_values' deltas and their dates as lists.
    """
    observed_mask = df["ndvi"].notna()
    if not observed_mask.any():
        return [], []
    obs_dates = df.index[observed_mask]
    deltas = []
    for d in obs_dates:
        upper = double_logistic_function_numpy(d, params_upper).item()
        lower = double_logistic_function_numpy(d, params_lower).item()
        median = 0.5 * (upper + lower)
        deltas.append(df.at[d, "ndvi"] - median)
    # take last smoothing_values
    if len(deltas) > smoothing_values:
        deltas = deltas[-smoothing_values:]
        obs_dates = obs_dates[-smoothing_values:]
    return list(deltas), list(obs_dates)

# -----------------------------
# Main daily loop: iterate days, process pixels, persist state
# -----------------------------
SMOOTHING_VALUES = 7  # keep consistent with function default
for day_idx in range(n_days):
    current_day = base_date + timedelta(days=day_idx)
    print(f"Processing day {day_idx+1}/{n_days} -> {current_day}")

    # For each pixel process sequentially (policy: open/close per day)
    for pixel in pixels:
        # params for this pixel
        params_l = params_lower_zarr[pixel]
        params_u = params_upper_zarr[pixel]

        # Reconstruct df for this pixel up to current day
        df = reconstruct_df_for_pixel(pixel, day_idx)

        # Reconstruct last_date and last_potential_date from last_dates_zarr
        try:
            raw_last = last_dates_zarr[6, pixel]  # this was used in your earlier code
            raw_pot  = last_dates_zarr[7, pixel]
            last_date = None
            last_potential_date = None
            if isinstance(raw_last, (bytes, np.bytes_)) and raw_last != b"1900-01-01":
                last_date = datetime.strptime(raw_last.decode("utf-8"), "%Y-%m-%d").date()
            if isinstance(raw_pot, (bytes, np.bytes_)) and raw_pot != b"1900-01-01":
                last_potential_date = datetime.strptime(raw_pot.decode("utf-8"), "%Y-%m-%d").date()
        except Exception:
            last_date = None
            last_potential_date = None

        # If last_date is None, try to discover most recent observed day from df
        if last_date is None:
            obs_mask = df["ndvi"].notna()
            if obs_mask.any():
                last_date = df.index[obs_mask][-1].date()

        # Reconstruct deltas_arr & dates_delta_arr from df history
        deltas_arr, dates_delta_arr = reconstruct_deltas_from_df(df, params_l, params_u, SMOOTHING_VALUES)

        # convert dates_delta_arr to python.date if they are Timestamps
        dates_delta_arr = [d.date() if isinstance(d, pd.Timestamp) else d for d in dates_delta_arr]

        # prepare latency accumulators (persisting these is optional)
        latency = []
        current_date_latency = []

        # Convert the 'date' argument to the format expected by the function: a Timestamp or date present in df.index
        # Our df index uses pandas Timestamp; pass pandas Timestamp for the function
        date_for_func = pd.Timestamp(current_day)

        # Call the function (it mutates df)
        last_date_after, last_potential_after, deltas_arr_after, dates_delta_arr_after, latency_after, current_date_latency_after = ndvi_continous_final_2(
            df=df,
            date=date_for_func,
            last_date=last_date,
            last_potential_date=last_potential_date,
            deltas_arr=deltas_arr,
            dates_delta_arr=dates_delta_arr,
            params_upper=params_u,
            params_lower=params_l,
            y_delta_l=0.02,
            y_delta_h=0.02,
            y_iqr=0.1,
            r_delta_l=-0.1,
            r_delta_h=0.1,
            r_iqr=0.3,
            tau=14,
            smoothing_values=SMOOTHING_VALUES,
            use_real_idx=False,
            frac=1,
            latency=latency,
            current_date_latency=current_date_latency,
        )

        # After function call, df contains:
        # - delta_smoothed over an interpolated date window (if any)
        # - deltas_L1 over last_date:date range (if any)
        # We'll combine these columns and write back to ndvi_zarr for the corresponding dates.

        # Combined smoothed delta: use delta_smoothed when present, otherwise use deltas_L1
        combined_delta = df["delta_smoothed"].copy()
        mask_L1 = df["deltas_L1"].notna()
        combined_delta[mask_L1] = df.loc[mask_L1, "deltas_L1"]

        # For rows where use_delta is False, the function expects to use raw LOESS on ndvi -> we'll set delta
        # but ensure we preserve df["use_delta"] (the function itself sets it)
        # Convert combined_delta + median -> ndvi values and write to Zarr
        for idx_date, delta_val in combined_delta.dropna().items():
            # idx_date is Timestamp
            if not np.isfinite(delta_val):
                continue
            # compute median for that date
            upper = double_logistic_function_numpy(idx_date, params_u).item()
            lower = double_logistic_function_numpy(idx_date, params_l).item()
            median_t = 0.5 * (upper + lower)
            ndvi_val = median_t + delta_val
            # saturate and scale
            scaled = int(np.clip(ndvi_val * 10000.0, 0, 10000))
            zidx = (idx_date.date() - base_date).days
            if 0 <= zidx < ndvi_zarr.shape[0]:
                ndvi_zarr[zidx, pixel] = np.int16(scaled)

        # Also, if df["forecast"] was written for current day when obs is missing, we might want to write it to ndvi_zarr
        # (Earlier logic wrote estimations when missing). We'll write if current day had forecast and no observed ndvi.
        if pd.isna(df.at[date_for_func, "ndvi"]) and pd.notna(df.at[date_for_func, "forecast"]):
            ndvi_est = df.at[date_for_func, "forecast"]
            scaled = int(np.clip(ndvi_est * 10000.0, 0, 10000))
            ndvi_zarr[day_idx, pixel] = np.int16(scaled)

        # Persist updated last_date & last_potential_date into last_dates_zarr slots 6 and 7
        # Use b"1900-01-01" as placeholder for None
        ld_bytes = date_to_zarr_bytes(last_date_after) if last_date_after is not None else b"1900-01-01"
        pot_bytes = date_to_zarr_bytes(last_potential_after) if last_potential_after is not None else b"1900-01-01"
        last_dates_zarr[6, pixel] = ld_bytes
        last_dates_zarr[7, pixel] = pot_bytes

        # Optionally: flush to disk (Zarr writes are immediate, but ensure metadata updated)
        # zarr doesn't require an explicit sync call typically, but you can .store or .oindex if using a specific store.
        # We'll just continue; the next day's run will open this DST and read updated arrays.

    # End of per-pixel loop for the day
    # (You can add periodic gc or logging here)
    print(f"Day {current_day} processed for pixels {pixels.tolist()}")

print("All days processed. Done.")
