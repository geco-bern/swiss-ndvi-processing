# nohup python /home/francesco/data_scratch/swiss-ndvi-processing/demo/area_visulization/highland_broadleaf.py > output_high_broadleaf.log 2>&1 &

# find correct pixel
import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import rioxarray
import zarr
import pandas as pd

# ----- Config -----
zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"
out_gif = "ndvi_highland_broadleaf_non_gapfilled.gif"

z = zarr.open(zarr_path, mode = "r")


# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# Rectangle corners (UL and BR)
UL_x, UL_y = 2612000.000000, 1205000.000000
BR_x, BR_y = 2613000.000000, 1204000.000000

# ----- compute pixel window (row 0 = top) -----
x_min, x_max = min(UL_x, BR_x), max(UL_x, BR_x)
y_min, y_max = min(UL_y, BR_y), max(UL_y, BR_y)

col_min = int(math.floor((x_min - left) / px))
col_max = int(math.floor((x_max - left) / px))

row_min = int(math.floor((top - y_max) / px))
row_max = int(math.floor((top - y_min) / px))

# clip to bounds
col_min = max(0, min(width-1, col_min))
col_max = max(0, min(width-1, col_max))
row_min = max(0, min(height-1, row_min))
row_max = max(0, min(height-1, row_max))

win_cols = col_max - col_min + 1
win_rows = row_max - row_min + 1
print(f"Window cols {col_min}..{col_max} ({win_cols}), rows {row_min}..{row_max} ({win_rows})")

# ----- load mask -----
mask = np.load(mask_path)
assert mask.shape == (height, width), f"Mask shape {mask.shape} != raster {(height, width)}"

mask_flat = mask.ravel(order='C')
masked_positions = np.flatnonzero(mask_flat)
n_masked = masked_positions.size
print(f"Mask has {n_masked} True pixels.")

# build index map from full array -> masked array
idx_map = np.full(mask_flat.shape[0], -1, dtype=np.int64)
idx_map[masked_positions] = np.arange(n_masked, dtype=np.int64)

# ----- compute flat indices in window -----
rows = np.arange(row_min, row_max+1, dtype=np.int64)
cols = np.arange(col_min, col_max+1, dtype=np.int64)
rr, cc = np.meshgrid(rows, cols, indexing='ij')
full_flat_idx = (rr * width + cc).ravel()

masked_idx_in_window = idx_map[full_flat_idx]
is_masked = masked_idx_in_window >= 0
n_masked_in_window = is_masked.sum()
print(f"Pixels in window: {full_flat_idx.size}, masked pixels: {n_masked_in_window}")

if n_masked_in_window == 0:
    raise RuntimeError("No masked pixels in window!")

# ----- open Zarr -----
N, T = z.shape
assert N == n_masked, f"Zarr first-dim {N} != mask True count {n_masked}"

# ----- plotting extent -----
extent = (
    left + col_min * px,
    left + (col_max + 1) * px,
    top - (row_max + 1) * px,
    top - row_min * px
)


values = np.empty(n_masked_in_window, dtype=float)
batch = 1_000_000
start = 0
end = min(start + batch, n_masked_in_window)
sel = masked_idx_in_window[is_masked][start:end].tolist()


# Limit to first 500 timesteps or all
n_frames = min(100, T)

# ----- plotting extent -----
extent = (
    left + col_min * px,
    left + (col_max + 1) * px,
    top - (row_max + 1) * px,
    top - row_min * px
)

# ----- create GIF -----
frames = []

for t in np.arange(0,n_frames):
    # read NDVI for masked pixels in window
    values = np.empty(n_masked_in_window, dtype=float)
    batch = 1_000_000
    start = 0
    while start < n_masked_in_window:
        end = min(start + batch, n_masked_in_window)
        sel = masked_idx_in_window[is_masked][start:end].tolist()
        values_batch = z[sel, t].astype(float)

        # normalize
        values_batch /= 10000.0

        # mask clouds
        values_batch = np.where((values_batch > 1) | (values_batch < 0), np.nan, values_batch)

        values[start:end] = values_batch
        start = end

    # reconstruct 2D window
    window_arr = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_arr[is_masked] = values
    window_arr = window_arr.reshape((win_rows, win_cols))

    # plot frame
    fig, ax = plt.subplots(figsize=(6, 6 * win_rows / win_cols))
    im = ax.imshow(window_arr, origin='upper', extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_title(f"NDVI timestep {t}")
    ax.set_xlabel("EPSG:2056 (m)")
    ax.set_ylabel("EPSG:2056 (m)")
    plt.colorbar(im, ax=ax, label="NDVI")

    # convert to image
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    frames.append(image)
    plt.close(fig)

# save GIF
imageio.mimsave(out_gif, frames, fps=10)
print("Saved GIF:", out_gif)


ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/sample_seasonal_cycle_parameter_preds.zarr")
ndvi = ds['ndvi']
dates = ds['dates']

params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)

# convert dates to doy
dates_pd = pd.to_datetime(dates)

df = pd.DataFrame({
    'date': dates_pd
})

# Sort by date
df_sorted = df.sort_values(by='date')

# Extract the sorted arrays if needed
dates_sorted = df_sorted['date'].values
dates_pd_sorted = pd.to_datetime(dates_sorted)

doy = dates_pd_sorted.dayofyear.values

doy = torch.tensor(doy, dtype=torch.float32)
T_SCALE = 1.0 / 365.0
t = doy.unsqueeze(0).repeat(params_lower.shape[0], 1) * T_SCALE

# Define the double logistic function
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

lower = double_logistic_function(t[[0]], params_lower[[91]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[91]]).squeeze().cpu().numpy()

iqr = lower -upper

median_iqr = upper - iqr/2

param_iqr =1.5
bottom_iqr = 0.3
upper_iqr = 0.7

dates_pd = pd.to_datetime(dates)


z_out =  zarr.open("/home/francesco/data_scratch/swiss-ndvi-processing/highland_broadleaf.zarr/ndvi")

for pixel in np.arange(0,len(sel)-1):

    pixel_sel = sel[pixel]

    time_serie = z[pixel_sel, ].astype(float)
    time_serie /= 10000.0

    # mask clouds
    time_serie = np.where((time_serie > 1) | (time_serie < 0), np.nan, time_serie)

    # proper sorting

    df = pd.DataFrame({
        'date': dates_pd,
        'ndvi': time_serie
        })

    df_sorted = df.sort_values(by='date')

    dates_sorted = df_sorted['date'].values
    ndvi_sorted = df_sorted['ndvi'].values

    # initialize ndvi gapfilled
    ndvi_gapfilled = np.copy(ndvi_sorted)


    valid_idx = np.where(np.isfinite(ndvi_sorted))
    valid_ndvi = ndvi_sorted[valid_idx]
    valid_idx = np.array(valid_idx[0])
    valid_upper = upper[valid_idx]
    valid_lower = lower[valid_idx]

    valid_iqr = valid_upper - valid_lower

    median_valid = valid_upper - valid_iqr / 2

    # initialize outlier flag
    outlier_arr = np.repeat(False,len(dates))

    #######################
    ## outlier detection ##
    #######################

    # calculate threshold
    delta_threshold_upper = valid_upper + param_iqr * valid_iqr
    delta_threshold_lower = valid_lower - param_iqr * valid_iqr

    # outlier detection by threshold

    deltas = []
    is_outlier_threshold = []

    for i in range(0,len(valid_idx)):

        if valid_ndvi[i] > valid_upper[i]:

                delta = valid_ndvi[i] - valid_upper[i]

                if valid_ndvi[i] > delta_threshold_upper[i]:
                    
                    outlier = True
                else:
                    outlier = False
        
        elif (valid_ndvi[i] < valid_upper[i]) and (valid_ndvi[i] > valid_lower[i]):
            
            delta = valid_ndvi[i] - (valid_upper[i] - valid_iqr[i])
            outlier = False

        else:
                delta = valid_ndvi[i] - valid_lower[i]

                if valid_ndvi[i] < delta_threshold_lower[i]:
                    
                    outlier = True
                else:
                    outlier = False

        deltas.append(delta)
        is_outlier_threshold.append(outlier)
        
    is_outlier_threshold = np.array(is_outlier_threshold)
    deltas = np.array(deltas)

    # outlier detection by deltas neighbour
    # the deltas are calculated based on the neaster bound

    delta_delta_left = (deltas[1:] - deltas[:-1]) 
    delta_delta_right = (deltas[:-1] - deltas[1:]) 

    delta_delta_left = np.array(delta_delta_left)
    delta_delta_right = np.array(delta_delta_right)


    slope_is_outlier = np.logical_and(
            
            np.logical_or(delta_delta_left[1:] > np.quantile(delta_delta_left,upper_iqr), 
                        delta_delta_left[1:]  <  np.quantile(delta_delta_left,bottom_iqr)),

            np.logical_or(delta_delta_right[:-1]  > np.quantile(delta_delta_right,upper_iqr), 
                        delta_delta_right[:-1]  < np.quantile(delta_delta_right,bottom_iqr))
        )


    # to be an outlier, a point must met both conditions
    is_outlier = np.logical_and(is_outlier_threshold[1:-1],slope_is_outlier)

    # write the outlier in the new data
    outlier_arr[valid_idx[1:-1]] = is_outlier

    # outlier detection first and last
    if np.logical_and(
        
        np.logical_or(
            delta_delta_right[0] >= np.quantile(delta_delta_right,upper_iqr), 
            delta_delta_right[0] <= np.quantile(delta_delta_right,bottom_iqr)
            ),
            
            is_outlier_threshold[0]):

        outlier_arr[valid_idx[0]] = True
        ndvi_gapfilled[valid_idx[0]] = np.nan

    if np.logical_and(
                    
            np.logical_or(
                delta_delta_left[-1] >= np.quantile(delta_delta_left,upper_iqr), 
                delta_delta_left[-1] <= np.quantile(delta_delta_left,bottom_iqr)
                ),
                
                is_outlier_threshold[-1]) == True:


        outlier_arr[valid_idx[-1]] = True
        ndvi_gapfilled[valid_idx[-1]] = np.nan

    #######################
    ## linear gapfilling ##
    #######################

    to_remove = valid_idx[1:-1][is_outlier == True]
    ndvi_gapfilled[to_remove] = np.nan

    valid_idx = np.where(np.isfinite(ndvi_gapfilled))
    valid_idx = np.array(valid_idx[0])

    distances = valid_idx[1:] - valid_idx[:-1]  

    for i in range(0,len(valid_idx)-1):

        idx_to_gapfill = range(valid_idx[i]+1,valid_idx[i+1])
        idx_to_gapfill = np.array(idx_to_gapfill)

        if len(idx_to_gapfill) != 0: # if len == 0 means 2 contigous obs data

            multiplier = range(1,len(idx_to_gapfill)+1)
            multiplier = np.array(multiplier)

            # gapfill based on the median

            delta_1 = ndvi_gapfilled[valid_idx[i]] - median_iqr[valid_idx[i]] 
            delta_2 = ndvi_gapfilled[valid_idx[i+1]] - median_iqr[valid_idx[i+1]] 
            
            slope = (delta_2 - delta_1) / distances[i]

            values = (median_iqr[idx_to_gapfill] + delta_1 + slope * multiplier ) 


            ndvi_gapfilled[idx_to_gapfill] = values


    z_out[:, pixel] = ndvi_gapfilled

    if pixel % 100 == 0:
        print(pixel)



out_gif_gapfilled = "ndvi_highland_broadleaf_gapfilled.gif"

frames = []
n_frames = min(100, z_out.shape[0])  # time dimension

for t in range(n_frames):
    # read NDVI for all masked pixels in window
    values = z_out[t, :].astype(float)  # shape (len(sel),)

    # reconstruct 2D window
    window_arr = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_arr[is_masked] = values
    window_arr = window_arr.reshape((win_rows, win_cols))

    # plot frame
    fig, ax = plt.subplots(figsize=(6, 6 * win_rows / win_cols))
    im = ax.imshow(window_arr, origin='upper', extent=extent,
                   vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_title(f"Gapfilled NDVI timestep {t}")
    ax.set_xlabel("EPSG:2056 (m)")
    ax.set_ylabel("EPSG:2056 (m)")
    plt.colorbar(im, ax=ax, label="NDVI")

    # convert to image
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    frames.append(image)
    plt.close(fig)

# save GIF
imageio.mimsave(out_gif_gapfilled, frames, fps=10)
print("Saved GIF:", out_gif_gapfilled)
