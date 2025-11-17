# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/notebook/update_parametercopy.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/update_2.log 



# library loading, nothing interesting
from IPython.display import IFrame, Image, display
import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import torch
import torch.nn as nn
import pandas as pd
import statsmodels.api as sm

import gc
import imageio
from io import BytesIO
from affine import Affine
import matplotlib.dates as mdates
import os
from functions import *

save_path = "figure2/"

os.makedirs(save_path, exist_ok=True)

ds = zarr.open_group("/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr",mode = "r")
params = ds["params"]
params_lower = params["params_lower"]
params_upper = params["params_upper"]
ndvi = ds["ndvi"]
ndsi = ds["ndsi"]
dates = pd.to_datetime([d.decode("utf-8") for d in ds["dates"][:]])

T_SCALE = 1.0 / 365.0
doy = dates.dayofyear
t = torch.tensor(doy * T_SCALE, dtype=torch.float32)

def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

order = np.argsort(dates)
dates_sorted = np.array(dates)[order]
t_sorted = t[order]

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"


def extract_pixel(UL_x, UL_y,BR_x, BR_y ):

    # ----- compute pixel window (row 0 = top) -----
    x_min, x_max = min(UL_x, BR_x), max(UL_x, BR_x)
    y_min, y_max = min(UL_y, BR_y), max(UL_y, BR_y)

    col_min = int(math.floor((x_min - left) / px))
    col_max = int(math.floor((x_max - left) / px))

    row_min = int(math.floor((top - y_max) / px))
    row_max = int(math.floor((top - y_min) / px))

    # clip to bounds
    col_min = max(0, min(width - 1, col_min))
    col_max = max(0, min(width - 1, col_max))
    row_min = max(0, min(height - 1, row_min))
    row_max = max(0, min(height - 1, row_max))

    win_cols = col_max - col_min + 1
    win_rows = row_max - row_min + 1
    print(f"Window cols {col_min}..{col_max} ({win_cols}), rows {row_min}..{row_max} ({win_rows})")

    # ----- load mask -----
    mask = np.load(mask_path)
    assert mask.shape == (height, width), f"Mask shape {mask.shape} != raster {(height, width)}"

    mask_flat = mask.ravel(order="C")
    masked_positions = np.flatnonzero(mask_flat)
    n_masked = masked_positions.size
    print(f"Mask has {n_masked} True pixels.")

    # build index map from full array -> masked array
    idx_map = np.full(mask_flat.shape[0], -1, dtype=np.int64)
    idx_map[masked_positions] = np.arange(n_masked, dtype=np.int64)

    # ----- compute flat indices in window -----
    rows = np.arange(row_min, row_max + 1, dtype=np.int64)
    cols = np.arange(col_min, col_max + 1, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    full_flat_idx = (rr * width + cc).ravel()

    masked_idx_in_window = idx_map[full_flat_idx]
    is_masked = masked_idx_in_window >= 0
    n_masked_in_window = is_masked.sum()
    print(f"Pixels in window: {full_flat_idx.size}, masked pixels: {n_masked_in_window}")

    if n_masked_in_window == 0:
        raise RuntimeError("No masked pixels in window!")

    sel = masked_idx_in_window[is_masked].tolist()


    return(sel)

def analysis(pixels):
    # --- loop over pixels 12 to 24 and create a 4x3 figure ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()


    lower = double_logistic_function(t_sorted, params_lower[[pixel]]).squeeze().numpy()
    upper = double_logistic_function(t_sorted, params_upper[[pixel]]).squeeze().numpy()
    ndvi_vals = np.asarray(ndvi[pixel], dtype=np.float32)[order]
    ndsi_vals = np.asarray(ndsi[pixel])[order]
    is_snow = ndsi_vals >= 0.43

    is_valid = ((ndvi_vals > 0) & (ndvi_vals <= 10000.0))
    ndvi_vals[~is_valid] = np.nan
    ndvi_vals[is_snow] = np.nan
    ndvi_valid = ndvi_vals[is_valid] / 10000.0
    dates_sorted_valid = dates_sorted[is_valid]

    df = pd.DataFrame({
            'date': dates_sorted_valid,
            'ndvi': ndvi_valid
            })

    df_sorted = df.sort_values(by='date')

    ndvi_sorted = df_sorted['ndvi'].values
    y_delta_l , y_delta_h,r_delta_h, r_delta_l  = -0.15, 0.15,1,-1 #np.quantile(delta_diff, [0.1,0.9,0.95,0.05])
    y_iqr, r_iqr = 0.05,2

        # --- 1. Create initial df ---
    df_starting = df

        # --- 2. Handle duplicate dates ---
    df_starting = (
            df_starting.groupby("date", as_index=False)
            .mean(numeric_only=True)   # average if duplicates exist
            .set_index("date")
        )

        # --- 3. Reindex to full daily series ---
    full_index = pd.date_range(df_starting.index.min(), df_starting.index.max(), freq="D")
    df = df_starting.reindex(full_index)

        # add forecast column for later use
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

    df.index.name = "date"

        # remove data veofre marh 2018
    df = df.loc[(df.index >= pd.Timestamp("2018-04-25")) & (df.index <= pd.Timestamp("2021-09-01"))]

    


        # --- run your function row by row ---
        # need an inital condition
    last_date =  None #pd.Timestamp("2018-04-25")
    last_potential_date =None
    deltas_arr = []
    dates_delta_arr = []
        # this is extra just for the plotting
    latency = []
    current_date_latency = []

    for date in df.index[df.index >= pd.Timestamp("2018-04-26")]:
            last_date, last_potential_date_arr, deltas_arr, dates_delta_arr, latency,  current_date_latency = ndvi_continous_final_2(
                df=df,
                date=date,
                params_lower =  params_lower[[pixel]],
                params_upper =  params_upper[[pixel]],
                last_date=last_date,
                deltas_arr = deltas_arr,
                dates_delta_arr = dates_delta_arr,
                last_potential_date = last_potential_date,
                y_delta_l = y_delta_l, 
                y_delta_h = y_delta_h,
                r_delta_h = r_delta_h, 
                r_delta_l = r_delta_l,
                y_iqr= y_iqr, 
                r_iqr = r_iqr,
                tau = 45,
                smoothing_values = 5,
                latency = latency,
                current_date_latency = current_date_latency

            )

    print("ciao", pixel)

    return df["delta_smoothed"][ df.index.get_loc(pd.Timestamp("2021-08-01"))]




# save all the performance metrics
all_metrics = []

# storm
center_x, center_y =  2644218.94, 1134325.81

UL_x, UL_y = center_x - 200, center_y - 200 
BR_x, BR_y = center_x + 200, center_y + 200
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 

values = []

for pixel in sel_1:

    value = analysis(pixels= pixel)
    values.append(value)

# --- Create output DataFrame ---
df_results = pd.DataFrame({
    "pixel": sel_1,
    "value": values,
})

df_results["bettle"] = df_results["value"] < -0.1

print(df_results)

df_results.to_csv("pixel_bettle_results.csv", index=False)
