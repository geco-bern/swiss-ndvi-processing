# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/script/area_extraction_gif_creation/highland_evergreen.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/output_high_evergreen.log 2>&1 &

import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import torch
import pandas as pd
from scipy.signal import savgol_filter
import gc
from functions import *

# ----- Config -----
zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

z = zarr.open(zarr_path, mode="r")

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# Rectangle corners (UL and BR)
UL_x, UL_y = 2782037.00, 1183475.00
BR_x, BR_y = 2783037.00, 1184475.00


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

# Limit to first timesteps
d_frames = min(1000, T)


# ----- seasonal cycle fitting -----
ds = xr.open_zarr("/data_2/scratch/sbiegel/processed/sample_seasonal_cycle_parameter_preds_updated.zarr")
ndvi = ds["ndvi"]
dates = ds["dates"]
params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)

# convert dates to doy
dates_pd = pd.to_datetime(dates)
df = pd.DataFrame({"date": dates_pd})
df_sorted = df.sort_values(by="date")
dates_sorted = df_sorted["date"].values
dates_pd_sorted = pd.to_datetime(dates_sorted)
doy = dates_pd_sorted.dayofyear.values
doy = torch.tensor(doy, dtype=torch.float32)
T_SCALE = 1.0 / 365.0
t = doy.unsqueeze(0).repeat(params_lower.shape[0], 1) * T_SCALE

# Define double logistic function
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

median_iqr = upper - (upper - lower) / 2

param_iqr = 1.5
bottom_iqr = 0.9
upper_iqr = 0.1
window_length = 14
polyorder = 2


random_pixels = sel[-1] #int(np.median(sel))  # pick one pixel index to check

ndvi_series = z[random_pixels, :]

ndvi_gapfilled, outlier_arr = gapfill_ndvi(ndvi_series, lower, upper)

smoothed_data = savgol_filter(ndvi_gapfilled, window_length=window_length, polyorder=polyorder) 


plot_results(
        title= "hihg_ever.png",
        pixel_idx=random_pixels,
        ndvi_series=ndvi_series / 10000.0,
        ndvi_gapfilled=ndvi_gapfilled,
        outlier_arr=outlier_arr,
        lower=lower,
        upper=upper,
        dates=dates_sorted,   # your precomputed time axis
        save_path=f"/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/figure/median_pixel_highland_evergreen_L1_gapfilled_evergreen.png"
    )

plot_results(
        title= "high_ever_1.png",
        pixel_idx=random_pixels,
        ndvi_series=ndvi_series / 10000.0, 
        ndvi_gapfilled=smoothed_data,
        outlier_arr=outlier_arr,
        lower=lower,
        upper=upper,
        dates=dates_sorted,   # your precomputed time axis
        save_path=f"/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/figure/median_pixel_highand_evergreen_L2_gapfilled_2.png"
    )

print("done")

# ----- open new zarr output -----
#z_out = zarr.open("/home/francesco/data_scratch/swiss-ndvi-processing/lowland_broadleaf.zarr/ndvi")
ndvi_chunk = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)
ndvi_chuck_smoothed = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)

# ----- GAPFILLING LOOP -----
ndvi_chunk = np.empty((n_masked_in_window, T), dtype=np.float32)
outlier_matrix = np.zeros((n_masked_in_window, T), dtype=bool)

ndvi_continous_L1 = np.empty((n_masked_in_window, T), dtype=np.float32)
ndvi_continous_L2 = np.empty((n_masked_in_window, T), dtype=np.float32)

for i, pixel_sel in enumerate(sel):
    if i % 1000 == 0:
        print(f"Gapfilling pixel {i}/{len(sel)}")

    ndvi_series = z[pixel_sel, :]   # raw time series

    ndvi_gapfilled, outlier_arr = gapfill_ndvi(ndvi_series, lower, upper)

    smoothed_data = np.copy(ndvi_gapfilled)

    smoothed_data = savgol_filter(smoothed_data, window_length=window_length, polyorder=polyorder) 

    ndvi_chuck_smoothed[i, :] = smoothed_data
    ndvi_chunk[i, :] = ndvi_gapfilled
    outlier_matrix[i, :] = outlier_arr


    ndvi_filled, outlier_mask, forecast_only, smoothed = gapfill_ndvi(ndvi_series, lower, upper, forecasting=True)

    ndvi_continous_L1[i, :] = forecast_only
    ndvi_continous_L2[i, :] = ndvi_filled


out_gif_combined_1 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/ndvi_highand_evergreen_combined_1.gif"
out_gif_combined_2 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/ndvi_highand_evergreen_combined_2.gif"
out_gif_combined_3 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/ndvi_continous_highand_evergreen_combined.gif"

# Prepare writers (stream to disk instead of keeping frames in memory)
writer1 = imageio.get_writer(out_gif_combined_1, fps=10)
writer2 = imageio.get_writer(out_gif_combined_2, fps=10)
#writer3 = imageio.get_writer(out_gif_combined_3, fps=10)

for t in range(d_frames):

    if t % 100 == 0:
        print("Step:", t)

    # --- Non gapfilled ---
    values_non_gapfilled = np.empty(n_masked_in_window, dtype=float)
    batch = 1_000_000
    start = 0
    while start < n_masked_in_window:
        end = min(start + batch, n_masked_in_window)
        sel = masked_idx_in_window[is_masked][start:end].tolist()
        values_batch = z[sel, t].astype(float) / 10000.0
        values_batch = np.where((values_batch > 1) | (values_batch < 0), np.nan, values_batch)
        values_non_gapfilled[start:end] = values_batch
        start = end

    window_non_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_non_gapfilled[is_masked] = values_non_gapfilled
    window_non_gapfilled = window_non_gapfilled.reshape((win_rows, win_cols))

    # --- Gapfilled L1 ---
    window_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_gapfilled[is_masked] = ndvi_chunk[:, t].astype(float)
    window_gapfilled = window_gapfilled.reshape((win_rows, win_cols))

    # --- Gapfilled L2 ---
    window_gapfilled_smoothed = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_gapfilled_smoothed[is_masked] = ndvi_chuck_smoothed[:, t].astype(float)
    window_gapfilled_smoothed = window_gapfilled_smoothed.reshape((win_rows, win_cols))

    # --- Plot Non gapfilled vs L1 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * win_rows / win_cols), constrained_layout=True)
    im0 = axes[0].imshow(window_non_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_title("Non Gapfilled")
    im1 = axes[1].imshow(window_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[1].set_title("Gapfilled L1 product")
    fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05, pad=0.02).set_label("NDVI")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    writer1.append_data(buf[:, :, :3].copy())
    plt.close(fig)

    # --- Plot L1 vs L2 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * win_rows / win_cols), constrained_layout=True)
    im0 = axes[0].imshow(window_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_title("Gapfilled L1 product")
    im1 = axes[1].imshow(window_gapfilled_smoothed, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[1].set_title("Gapfilled L2 product")
    fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05, pad=0.02).set_label("NDVI")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    writer2.append_data(buf[:, :, :3].copy())
    plt.close(fig)

    # --- Continuous ingestion GIF (only from step 14 onward) ---
    """if t >= 14:
        window_gapfilled_c1 = np.full(win_rows * win_cols, np.nan, dtype=float)
        window_gapfilled_c1[is_masked] = ndvi_continous_L1[:, t].astype(float)
        window_gapfilled_c1 = window_gapfilled_c1.reshape((win_rows, win_cols))

        window_gapfilled_c2 = np.full(win_rows * win_cols, np.nan, dtype=float)
        window_gapfilled_c2[is_masked] = ndvi_continous_L2[:, t].astype(float)
        window_gapfilled_c2 = window_gapfilled_c2.reshape((win_rows, win_cols))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6 * win_rows / win_cols), constrained_layout=True)
        im0 = axes[0].imshow(window_non_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
        axes[0].set_title("Gapfilled forecast")
        im1 = axes[1].imshow(window_gapfilled_c1, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
        axes[1].set_title("Gapfilled retroactive")
        fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05, pad=0.02).set_label("NDVI")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        writer3.append_data(buf[:, :, :3].copy())
        plt.close(fig)"""

    # --- free memory explicitly ---
    del values_non_gapfilled, window_non_gapfilled
    del window_gapfilled, window_gapfilled_smoothed
    gc.collect()

# Close writers
writer1.close()
writer2.close()
#writer3.close()

print("All GIFs saved.")


