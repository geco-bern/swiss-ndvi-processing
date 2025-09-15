# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/script/area_extraction_gif_creation/fire.py > fire.log 2>&1 &

import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import torch
import pandas as pd
from scipy.signal import savgol_filter
import geopandas as gpd
from rasterio import features
from affine import Affine
from shapely.geometry import Point
from functions import *
import gc


# ----- Config -----
zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

z = zarr.open(zarr_path, mode="r")

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# ----- load polygon from KML -----
gdf = gpd.read_file("/home/francesco/data_scratch/swiss-ndvi-processing/demo/area_visulization/polygon_fire.kml", driver="KML")
gdf = gdf.to_crs(epsg=2056)   # Swiss LV95
polygon = gdf.geometry.unary_union

# ----- raster info -----
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# ----- polygon bounds -> pixel window -----
x_min, y_min, x_max, y_max = polygon.bounds

col_min = int(math.floor((x_min - left) / px))
col_max = int(math.floor((x_max - left) / px))
row_min = int(math.floor((top - y_max) / px))
row_max = int(math.floor((top - y_min) / px))

# clip to raster extent
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

# ----- compute pixel centers in window -----
rows = np.arange(row_min, row_max + 1, dtype=np.int64)
cols = np.arange(col_min, col_max + 1, dtype=np.int64)
rr, cc = np.meshgrid(rows, cols, indexing="ij")

xx = left + (cc + 0.5) * px
yy = top - (rr + 0.5) * px

# ----- rasterize polygon to mask instead of looping -----
transform = Affine(px, 0, left, 0, -px, top)
poly_mask = features.rasterize(
    [(polygon, 1)],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# extract only window
inside_mask = poly_mask[row_min:row_max+1, col_min:col_max+1].astype(bool)

# flatten to keep same workflow
inside = inside_mask.ravel()

# ----- flat indices -----
full_flat_idx = (rr * width + cc).ravel()

# keep only pixels inside polygon
poly_flat_idx = full_flat_idx[inside]

masked_idx_in_window = idx_map[poly_flat_idx]
is_masked = masked_idx_in_window >= 0
n_masked_in_window = is_masked.sum()
print(f"Pixels in window: {full_flat_idx.size}, masked pixels in polygon: {n_masked_in_window}")

if n_masked_in_window == 0:
    raise RuntimeError("No masked pixels in polygon!")

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


lower = double_logistic_function(t[[0]], params_lower[[91]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[91]]).squeeze().cpu().numpy()

median_iqr = upper - (upper - lower) / 2

param_iqr = 1.5
bottom_iqr = 0.9
upper_iqr = 0.1
window_length = 14
polyorder = 2


random_pixels = int(np.median(sel))  # pick one pixel index to check

ndvi_series = z[random_pixels, :]

ndvi_gapfilled, outlier_arr = gapfill_ndvi(ndvi_series, lower, upper)

smoothed_data = savgol_filter(ndvi_gapfilled, window_length=window_length, polyorder=polyorder) 


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


    ndvi_filled, outlier_mask = gapfill_ndvi(ndvi_series, lower, upper, forecasting=False)

    ndvi_continous_L2[i, :] = ndvi_filled


out_gif_combined_1 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/fire.gif"
out_gif_combined_2 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/ndvi_highand_broadleaf_combined_2.gif"
out_gif_combined_3 = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/gif/ndvi_continous_highand_broadleaf_combined.gif"

# Prepare writers (stream to disk instead of keeping frames in memory)
writer1 = imageio.get_writer(out_gif_combined_1, fps=10)

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

    # --- Create window arrays ---
    window_non_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_gapfilled_smoothed = np.full(win_rows * win_cols, np.nan, dtype=float)

    # --- Compute flat positions of masked pixels in the window ---
    masked_window_positions = np.flatnonzero(is_masked)

    # --- Assign values ---
    window_non_gapfilled[masked_window_positions] = values_non_gapfilled
    window_gapfilled[masked_window_positions] = ndvi_chunk[:, t].astype(float)
    window_gapfilled_smoothed[masked_window_positions] = ndvi_chuck_smoothed[:, t].astype(float)

    # --- Reshape to 2D window ---
    window_non_gapfilled = window_non_gapfilled.reshape((win_rows, win_cols))
    window_gapfilled = window_gapfilled.reshape((win_rows, win_cols))
    window_gapfilled_smoothed = window_gapfilled_smoothed.reshape((win_rows, win_cols))

    # --- Plot Non gapfilled vs L1 ---
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6 * win_rows / win_cols), constrained_layout=True
    )
    im0 = axes[0].imshow(window_non_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_title("Non Gapfilled")

    im1 = axes[1].imshow(window_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[1].set_title("Gapfilled L1 product")

    fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05, pad=0.02).set_label("NDVI")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    writer1.append_data(buf[:, :, :3].copy())
    plt.close(fig)

    # --- free memory explicitly ---
    del values_non_gapfilled, window_non_gapfilled
    del window_gapfilled, window_gapfilled_smoothed
    gc.collect()

# Close writer
writer1.close()
print("All GIFs saved.")

