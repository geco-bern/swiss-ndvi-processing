#   nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/notebook/lowland_evergreen.py >  /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/low_ever.log 
# 2>/dev/null &

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
from scipy.signal import savgol_filter
import gc
import imageio
from io import BytesIO
from functions import *

import geopandas as gpd
from rasterio import features
from affine import Affine
from shapely.geometry import Point
import time


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

# data loading and raster initialization
# ----- Config -----

# fitting and smoothing
# ----- seasonal cycle fitting -----
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


# area extracion

# ----- Config -----
zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

z = zarr.open(zarr_path, mode="r")

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# ----- center cooridnates  -----
center_x, center_y = 2761097.61, 1194613.45

# Rectangle corners (UL and BR)
UL_x, UL_y = center_x - 200, center_y - 200 
BR_x, BR_y = center_x + 200, center_y + 200


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
# ----- open Zarr -----
N, T = z.shape
assert N == n_masked, f"Zarr first-dim {N} != mask True count {n_masked}"

# ----- plotting extent -----
extent = (
    left + col_min * px,
    left + (col_max + 1) * px,
    top - (row_max + 1) * px,
    top - row_min * px,
)


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

print("Window extent:", extent)
print("win_rows, win_cols:", win_rows, win_cols)
print("Masked pixels in window:", n_masked_in_window)


# ----- open Zarr -----
N, T = z.shape
assert N == n_masked, f"Zarr first-dim {N} != mask True count {n_masked}"

# ----- plotting extent -----
extent = (
    left + col_min * px,
    left + (col_max + 1) * px,
    top - (row_max + 1) * px,
    top - row_min * px,
)

full_index = pd.date_range(min(dates_sorted), max(dates_sorted), freq="D")

start_date = "2018-04-01"

end_date = "2025-01-01"

full_index = full_index[(full_index >= start_date) & (full_index <= end_date)]


T = len(full_index)


# Limit to first timesteps
d_frames = min(3000, T)

# new zarr creation
# ----- open new zarr output -----
ndvi_chunk = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)
ndvi_chuck_smoothed = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)

# ----- GAPFILLING LOOP -----
for i, pixel in enumerate(sel):

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
        y_delta_l , y_delta_h,r_delta_h, r_delta_l  = -0.25, 0.25,1,-1 #np.quantile(delta_diff, [0.1,0.9,0.95,0.05])
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

        df.index.name = "date"

        spinup = "2018-04-01"
        # remove uninteresting data
        df = df.loc[df.index >= pd.Timestamp(spinup)]

        # --- run your function row by row ---
        # need an inital condition
        last_date =  None 
        last_potential_date =None
        deltas_arr = []
        dates_delta_arr = []
        # this is extra just for the plotting
        latency = []
        current_date_latency = []

        start_time = time.time()

        for date in df.index[df.index >= pd.Timestamp(spinup)]:
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
                smoothing_values = 9,
                latency = latency,
                current_date_latency = current_date_latency

            )

        end_time = time.time()

        print(f"Execution time: {end_time - start_time:.2f} seconds")


        # --- Final NDVI reconstruction ---
        df_plot = df[((df.index >= start_date) & (df.index <= end_date))]
        #df_plot = df_plot.reindex(full_index)

        df_plot["smoothed_combined"] = np.where(
        df_plot["use_delta"],
        0.5 * (df_plot["upper"] + df_plot["lower"]) + df_plot["delta_smoothed"],
        df_plot["delta_smoothed"]
        )

        # some very wierd bug appears at the interface between the 2 option

        # hot fix, take the mean between the 2 nearest point
        left = df_plot["smoothed_combined"].shift(1)
        right = df_plot["smoothed_combined"].shift(-1)
        mask_neighbors_high = (left > 0.1) & (right > 0.1)

        # replace center point with the mean of the two neighbors
        df_plot.loc[mask_neighbors_high, "smoothed_combined"] = (left + right) / 2

        final_value= df_plot["smoothed_combined"]

        final_ndvi = np.where(df_plot["outlier"],
                              np.nan,
                              df["ndvi"])   
        ndvi_chuck_smoothed[i, :] = final_value
        ndvi_chunk[i,:] = final_ndvi

print("finished gapfilling")

# !!! is mp4 not gif becuase gif is limtied t0 1000 frames
out_gif_combined_1 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/lowland_evergreen.mp4"

# Prepare writers (stream to disk instead of keeping frames in memory)
writer1 = imageio.get_writer(out_gif_combined_1, fps=30, format='ffmpeg', codec='libx264', ffmpeg_params=['-pix_fmt','yuv420p','-crf','18','-preset','fast'], quality = 8)

for t in range(d_frames):

    if t % 50 == 0:
        print(f"Step: {t} / {d_frames}")

    # --- Current date ---
    current_date = pd.to_datetime(full_index[t]) 
    date_str = current_date.strftime("%B %Y") 

    # --- Non gapfilled ---
    values_non_gapfilled = np.empty(n_masked_in_window, dtype=float)
    window_non_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)


    # assign values
    window_non_gapfilled[is_masked] = ndvi_chunk[:, t].astype(float)
    window_non_gapfilled = window_non_gapfilled.reshape((win_rows, win_cols))


    # --- Gapfilled L2 ---
    window_gapfilled_smoothed = np.full(win_rows * win_cols, np.nan, dtype=float)

    # reuse same indices
    window_gapfilled_smoothed[is_masked] = ndvi_chuck_smoothed[:, t].astype(float)
    window_gapfilled_smoothed = window_gapfilled_smoothed.reshape((win_rows, win_cols))


    # --- Plot Non gapfilled vs L2 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.087185 * win_rows / win_cols), constrained_layout=True) # so its divisible by 16
    im0 = axes[0].imshow(window_non_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_title("Raw data")
    im1 = axes[1].imshow(window_gapfilled_smoothed, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[1].set_title("Smoothed")

    fig.suptitle(f"{date_str}", fontsize=16)

    fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05, pad=0.02).set_label("NDVI")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].astype(np.uint8)

    # write the frame to the video
    writer1.append_data(frame)

    # close the figure and free memory
    plt.close(fig)
    del values_non_gapfilled, window_non_gapfilled, window_gapfilled_smoothed, frame, buf
    gc.collect()

writer1.close()

print("done")
