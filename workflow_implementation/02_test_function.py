# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/02_test_function.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_function.log 

import numpy as np
import pandas as pd
import zarr
import torch
import math
import statsmodels.api as sm
import os
import gc
from datetime import datetime

# CONFIGURATION
INPUT_ZARR = "/data_2/scratch/francesco/zarr_demo/"  

# path where the data are stored
ds = zarr.open_group(INPUT_ZARR, mode="r")
dates = pd.to_datetime([d.decode() if isinstance(d, (bytes, np.bytes_)) else str(d) for d in ds["dates"][:]])

# FUNCTIONS

def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

def calculate_median(dates_input):
    """
    Flexible: accepts scalar or iterable of:
      - datetime.datetime
      - numpy.datetime64
      - pandas.Timestamp
      - ISO date strings "YYYY-MM-DD"
      - bytes / numpy.bytes_ containing an ISO date
    Returns:
      - a numpy array (or scalar) median value computed for each input date
    Uses globals: params_lower, params_upper
    """

    # Normalize input into a list of pandas timestamps
    if isinstance(dates_input, (list, tuple, np.ndarray, pd.Index)):
        raw_list = list(dates_input)
    else:
        raw_list = [dates_input]

    # helper to convert one element to pd.Timestamp
    def _to_timestamp(x):
        # handle numpy scalar types
        if isinstance(x, (np.datetime64,)):
            return pd.to_datetime(str(x))
        # bytes / numpy.bytes_ -> decode
        if isinstance(x, (bytes, np.bytes_)):
            try:
                return pd.to_datetime(x.decode("utf-8"))
            except Exception:
                # fallback: try to decode repr like "np.bytes_(b'2024-04-02')"
                s = x.decode("utf-8", errors="ignore")
                s = s.strip()
                # strip common wrappers if present
                if s.startswith("np.bytes_(") and s.endswith(")"):
                    inner = s[s.find(b"b'".decode()):] if False else s
                return pd.to_datetime(s, errors="coerce")
        # if it's already a string
        if isinstance(x, str):
            # sometimes we receive a repr like "np.bytes_(b'2024-04-02')"
            if x.startswith("np.bytes_("):
                # try to extract the inner b'...' content
                import re
                m = re.search(r"b['\"]([^'\"]+)['\"]", x)
                if m:
                    return pd.to_datetime(m.group(1))
            # normal ISO string
            return pd.to_datetime(x)
        # datetime-like (datetime, Timestamp)
        return pd.to_datetime(x)

    # convert all
    ts_list = []
    for v in raw_list:
        try:
            ts = _to_timestamp(v)
        except Exception:
            ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Could not parse date: {v!r}")
        ts_list.append(ts)

    # Extract day-of-year values (1..366) as floats and scale
    T_SCALE = 1.0 / 365.0
    doy = np.array([ts.dayofyear for ts in ts_list], dtype=np.float32)
    t = torch.tensor(doy * T_SCALE, dtype=torch.float32)

    # compute thresholds using your double-logistic with the global params
    # params_lower and params_upper must already be torch tensors shaped like (1, 6)
    lower = double_logistic_function(t, params_lower).squeeze().numpy()
    upper = double_logistic_function(t, params_upper).squeeze().numpy()

    median = 0.5 * (upper + lower)

    # return scalar when input was scalar
    if not isinstance(dates_input, (list, tuple, np.ndarray, pd.Index)):
        return median[0] if hasattr(median, "__len__") else median
    
    return median


def outlier_detection():

    delta_delta = abs(delta_current - delta_previous)
    inside_band = (obs >= lower) and (obs <= upper)

    true_value = True

    if not inside_band:

        if (abs(delta_current) > 0.05) and (delta_delta > 0.25):

            # potential outlier
            true_value = False
    
    return true_value

# used if the proevious value is a potential outlier
def retroactive_outlier_detection():

    potential_median = calculate_median(potential_date)
    
    delta_delta = abs((obs - current_median) - (potential_ndvi - potential_median))

    if delta_delta < 0.25:

        # true observation, insert it in the last_dates array
        pass


def estimate_ndvi():

    decrease_factor = math.exp(-math.log(2) * (days_diff / 45)) # tau = 45
    estimation = current_median + delta_prev * decrease_factor

    ndvi_series[date_index] = estimation



def L1_interpolation(date_1, date_2, ndvi_series):

    days_diff = (date_2 - date_1).days
    if days_diff < 1:
        return  # nothing to interpolate

    index_1 = dates_cache.get(date_1.strftime("%Y-%m-%d"))
    index_2 = dates_cache.get(date_2.strftime("%Y-%m-%d"))
    if index_1 is None or index_2 is None:
        print(f"[WARN] date indices not found: {date_1}, {date_2}")
        return

    ndvi_1 = ndvi_series[index_1].item()
    ndvi_2 = ndvi_series[index_2].item()

    median_1 = calculate_median(date_1)
    median_2 = calculate_median(date_2)

    delta_1 = ndvi_1 - median_1
    delta_2 = ndvi_2 - median_2

    #  Linear interpolation of deltas
    L1_deltas = np.linspace(delta_1, delta_2, num=days_diff + 1)

    start_ordinal = date_1.toordinal()
    end_ordinal = date_2.toordinal()
    ordinals = np.arange(start_ordinal, end_ordinal + 1)
    dates_to_interpolate = np.array([datetime.fromordinal(o) for o in ordinals])

    medians = calculate_median(dates_to_interpolate)
    interpolated_ndvi = L1_deltas + medians

    #  Write interpolated values back to ndvi_series 
    for dt, value in zip(dates_to_interpolate, interpolated_ndvi):
        idx = dates_cache.get(dt.strftime("%Y-%m-%d"))
        if idx is not None:
            ndvi_series[idx] = value 


def create_tiff(date):

    # todo in future
    print("ciao ", date)


def L2_smoothing():

    last_dates = last_dates[:7] # must be only the first 7 elements

    # extract previous values
    previous_ndvi = 4
    previous_medians = calculate_median(last_dates)

    # Dates to interpolate: from second-to-last to last, inclusive
    start_date = last_dates[2]
    end_date = last_dates[3]
    dates_interp = pd.date_range(start=start_date, end=end_date, freq='D')
    idx = np.arange(len(previous_ndvi))

    deltas_arr = previous_ndvi - previous_medians

    # in case of extreme events (fire, storm NOT drought), ignore iqr
    if np.sum(np.array(deltas_arr) < -0.2) >= 5:

        # LOESS smoothing over the NDVI values
        loess = sm.nonparametric.lowess(previous_ndvi, idx, frac=1, it = 5,return_sorted=True)

        # Interpolate LOESS values
        y_interp = np.interp(idx, loess[:, 0], loess[:, 1])
        
        # write data

    else:

        # LOESS smoothing over the deltas
        loess = sm.nonparametric.lowess(deltas_arr, idx, frac=1, it = 5,return_sorted=True)

        # Interpolate LOESS values
        y_interp = np.interp(idx, loess[:, 0], loess[:, 1])
        
        # write data

    # increase counter for the smoothed date

    # if counter reach the threshold, create the tiff


def ingest_data(date,pixel_idx):

    
    ndvi_zarr = ds["ndvi"]
    counter_zarr = ds["counter"]
    last_dates_zarr = ds["last_dates"]
    params_lower_zarr = ds["params"]["params_lower"]
    params_upper_zarr = ds["params"]["params_upper"]

    date_str = date.strftime("%Y-%m-%d")
    date_index = dates_cache.get(date_str)
    if date_index is None:
        print(f"[WARN] Date {date_str} not found.")
        return None
    
    global params_lower, params_upper

    #counter_value = counter_zarr.get_basic_selection(date_index) 
    ndvi_series = ndvi_zarr.get_basic_selection((slice(None), pixel_idx))
    last_dates_pixel = last_dates_zarr.get_basic_selection((slice(None), pixel_idx))
    last_dates_pixel = [d.tobytes().decode("utf-8") for d in last_dates_pixel]

    params_lower = params_lower_zarr.get_basic_selection((pixel_idx, slice(None)))
    params_upper = params_upper_zarr.get_basic_selection((pixel_idx, slice(None)))

    # dates

    last_dates_objects = []
    for d_str in last_dates_pixel:
        last_dates_objects.append(datetime.strptime(d_str, "%Y-%m-%d"))

    last_date = last_dates_objects[6]

    potential_pending = False

    if last_dates_objects[7] != "01-01-0000":

        potential_date = last_dates_objects[8]
        potential_index = dates_cache.get(potential_date)
        potential_ndvi = ndvi_series[potential_index].item()
        potential_pending = True

    #  NDVI values
    current_ndvi = ndvi_series[date_index].item()

    last_index = dates_cache.get(last_date)
    last_ndvi = ndvi_series[last_index].item() if last_index is not None else np.nan


    params_lower = torch.tensor(params_lower, dtype=torch.float32).unsqueeze(0)
    params_upper = torch.tensor(params_upper, dtype=torch.float32).unsqueeze(0)

    current_median =  calculate_median(date)

    last_median = calculate_median(last_date)
    last_ndvi = ndvi_series[last_date]

    delta_prev = last_ndvi - last_median

    days_diff = (date - last_date).days

    # check if the data is nan or not
    if np.isfinite(current_ndvi) and current_ndvi > 0:

        delta_current = current_ndvi - current_median
        delta_previous = last_ndvi - last_median
        true_value = outlier_detection()

        if true_value:

            if last_date != "01-01-0000":

                if potential_pending:

                    retroactive_outlier_detection()

                else:

                    L1_interpolation(last_date, date)

                current_str = date.strftime("%Y-%m-%d")
                new_dates = [current_str] + last_dates_pixel[:-1] 
                new_dates_bytes = np.array([d.encode("utf-8") for d in new_dates], dtype="S10")
                last_dates_zarr.set_basic_selection((slice(None), pixel_idx), new_dates_bytes)
                
                # no potential outlier
                placeholder = np.array([b"01-01-0000"], dtype="S10")
                last_dates_zarr.set_basic_selection((7, pixel_idx), placeholder)

                L2_smoothing()

            else:

                # start the run (need at least 1 observation)

                current_str = date.strftime("%Y-%m-%d")
                new_dates = [current_str] + last_dates_pixel[:-1] 
                new_dates_bytes = np.array([d.encode("utf-8") for d in new_dates], dtype="S10")
                last_dates_zarr.set_basic_selection((slice(None), pixel_idx), new_dates_bytes)

    else:
        estimated_ndvi = estimate_ndvi()

def ingest_data_test(date,pixel_idx):

    date_str = date
    ndvi_zarr = ds["ndvi"]
    counter_zarr = ds["counter"]
    last_dates_zarr = ds["last_dates"]
    params_lower_zarr = ds["params"]["params_lower"]
    params_upper_zarr = ds["params"]["params_upper"]


    date_index = dates_cache.get(date_str)
    if date_index is None:
        print(f"[WARN] Date {date_str} not found.")
        return None
    
    global params_lower, params_upper

    # counter_value = counter_zarr.get_basic_selection(date_index) 
    ndvi_series = ndvi_zarr.get_basic_selection((slice(None), pixel_idx))
    last_dates_pixel = last_dates_zarr.get_basic_selection((slice(None), pixel_idx))

    last_dates_pixel = [d.decode("utf-8") if isinstance(d, bytes) else str(d) for d in last_dates_pixel]
    last_dates_objects = [datetime.strptime(d, "%Y-%m-%d") for d in last_dates_pixel]

    params_lower = params_lower_zarr.get_basic_selection((pixel_idx, slice(None)))
    params_upper = params_upper_zarr.get_basic_selection((pixel_idx, slice(None)))

    # dates


    last_date = last_dates_pixel[6]

    potential_pending = False

    if last_dates_pixel[7] != "1900-01-01":

        potential_date = last_dates_objects[7]
        potential_index = dates_cache.get(potential_date)
        potential_ndvi = float(ndvi_series[potential_index].ravel()[0])
        potential_pending = True

    #  NDVI values
    current_ndvi = ndvi_series[date_index].item()

    last_index = dates_cache.get(last_date)
    last_ndvi = ndvi_series[last_index].item() if last_index is not None else np.nan


    params_lower = torch.tensor(params_lower, dtype=torch.float32).unsqueeze(0)
    params_upper = torch.tensor(params_upper, dtype=torch.float32).unsqueeze(0)

    current_median =  calculate_median(date)

    last_median = calculate_median(last_date)
    last_ndvi = ndvi_series[last_date]

    days_diff = (date - last_date).days

    # check if the data is nan or not
    if np.isfinite(current_ndvi) and current_ndvi > 0:

        delta_current = current_ndvi - current_median
        delta_previous = last_ndvi - last_median
        true_value = outlier_detection()

        if true_value:

            if last_date != "1900-01-01":

                r = 3 

            else:

                # start the run (need at least 1 observation)


                last_dates_zarr.set_basic_selection((6, pixel_idx), date_str)


    else:
        if last_date != "1900-01-01":
            estimate_ndvi()


def ingest_data_test(date, pixel_idx):
    # date here is a string like "YYYY-MM-DD"
    date_str = date
    try:
        date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        # fallback to pandas parser
        date_dt = pd.to_datetime(date_str).to_pydatetime()

    ndvi_zarr = ds["ndvi"]
    last_dates_zarr = ds["last_dates"]
    params_lower_zarr = ds["params"]["params_lower"]
    params_upper_zarr = ds["params"]["params_upper"]

    date_index = dates_cache.get(date_str)
    if date_index is None:
        print(f"[WARN] Date {date_str} not found.")
        return None

    # ... rest of your code unchanged, but:
    # - when you call calculate_median use date_dt or last_date_dt (datetime objects)
    #   e.g. current_median = calculate_median(date_dt)
    # - when you compute days_diff use (date_dt - last_date_dt).days

def ingest_data_test_2(date_input, pixel_idx):
    """
    Test ingestion function for NDVI data.

    Args:
        date_input: str | datetime — the current date (e.g. "2024-04-02")
        pixel_idx: int — pixel index to process
    """

    # --- Convert and normalize the input date ---
    if isinstance(date_input, datetime):
        date_dt = date_input
        date_str = date_dt.strftime("%Y-%m-%d")
    else:
        # decode bytes, np.bytes_, or string
        if isinstance(date_input, (bytes, np.bytes_)):
            date_str = date_input.decode("utf-8")
        else:
            date_str = str(date_input)
        date_dt = datetime.strptime(date_str, "%Y-%m-%d")

    # --- Load datasets from Zarr ---
    ndvi_zarr = ds["ndvi"]
    last_dates_zarr = ds["last_dates"]
    params_lower_zarr = ds["params"]["params_lower"]
    params_upper_zarr = ds["params"]["params_upper"]

    # --- Lookup current date index ---
    date_index = dates_cache.get(date_str)
    if date_index is None:
        print(f"[WARN] Date {date_str} not found in cache.")
        return None

    # --- Load NDVI and date arrays for this pixel ---
    ndvi_series = ndvi_zarr.get_basic_selection((slice(None), pixel_idx))
    last_dates_pixel = last_dates_zarr.get_basic_selection((slice(None), pixel_idx))

    # Decode all stored bytes to strings
    last_dates_pixel = [
        d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d)
        for d in last_dates_pixel
    ]

    # Parse into datetime objects
    last_dates_objects = []
    for d in last_dates_pixel:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            # if placeholder or invalid, use a default
            dt = datetime(1900, 1, 1)
        last_dates_objects.append(dt)

    # --- Load parameter tensors ---
    global params_lower, params_upper
    params_lower = torch.tensor(
        params_lower_zarr.get_basic_selection((pixel_idx, slice(None))),
        dtype=torch.float32,
    ).unsqueeze(0)
    params_upper = torch.tensor(
        params_upper_zarr.get_basic_selection((pixel_idx, slice(None))),
        dtype=torch.float32,
    ).unsqueeze(0)

    # --- Handle potential outlier / pending state ---
    potential_pending = False
    if last_dates_objects[7] != datetime(1900, 1, 1):
        potential_date = last_dates_objects[7]
        potential_str = potential_date.strftime("%Y-%m-%d")
        potential_index = dates_cache.get(potential_str)
        if potential_index is not None:
            potential_ndvi = float(ndvi_series[potential_index].ravel()[0])
            potential_pending = True
        else:
            potential_ndvi = np.nan

    # --- Current and previous NDVI values ---
    current_ndvi = ndvi_series[date_index].item()
    last_date_dt = last_dates_objects[6]
    last_date_str = last_date_dt.strftime("%Y-%m-%d")

    last_index = dates_cache.get(last_date_str)
    last_ndvi = (
        ndvi_series[last_index].item() if last_index is not None else np.nan
    )

    # --- Calculate medians ---
    current_median = calculate_median(date_dt)
    last_median = calculate_median(last_date_dt)

    days_diff = (date_dt - last_date_dt).days if last_date_dt.year > 1900 else 0

    # --- Outlier detection / interpolation logic ---
    if np.isfinite(current_ndvi) and current_ndvi > 0:
        delta_current = current_ndvi - current_median
        delta_previous = last_ndvi - last_median

        true_value = outlier_detection()

        if true_value:
            if last_date_dt != datetime(1900, 1, 1):
                # handle potential retroactive correction
                if potential_pending:
                    retroactive_outlier_detection()
                else:
                    L1_interpolation(last_date_dt, date_dt, ndvi_series)

                # update last_dates in Zarr
                current_str = date_dt.strftime("%Y-%m-%d")
                new_dates = [current_str] + last_dates_pixel[:-1]
                new_dates_bytes = np.array(
                    [d.encode("utf-8") for d in new_dates], dtype="S10"
                )
                last_dates_zarr.set_basic_selection((slice(None), pixel_idx), new_dates_bytes)

                # reset placeholder (no pending)
                placeholder = np.array([b"1900-01-01"], dtype="S10")
                last_dates_zarr.set_basic_selection((7, pixel_idx), placeholder)

                L2_smoothing()
            else:
                # Initialize first run
                current_str = date_dt.strftime("%Y-%m-%d")
                new_dates = [current_str] + last_dates_pixel[:-1]
                new_dates_bytes = np.array(
                    [d.encode("utf-8") for d in new_dates], dtype="S10"
                )
                last_dates_zarr.set_basic_selection((slice(None), pixel_idx), new_dates_bytes)

    else:
        # Estimate NDVI if missing
        if last_date_dt != datetime(1900, 1, 1):
            estimate_ndvi()


# final loop after all the functions are ok

dates_zarr = ds["dates"]
dates_cache = {}

for i in range(ds["dates"].shape[0]):

    d_arr = ds["dates"].get_basic_selection((i,))
    d_val = d_arr[()] if isinstance(d_arr, np.ndarray) and d_arr.shape == () else d_arr[0]

    # --- FIX: clean up np.bytes_ and weird representations ---
    if isinstance(d_val, (bytes, np.bytes_)):
        d_str = d_val.decode("utf-8")
    else:
        d_str = str(d_val)
        if d_str.startswith("np.bytes_("):
            m = re.search(r"b['\"]([^'\"]+)['\"]", d_str)
            if m:
                d_str = m.group(1)
    # ---------------------------------------------------------

    dates_cache[d_str] = i


    for pixel in range(0,2):

        ingest_data_test_2(d_str,pixel)
