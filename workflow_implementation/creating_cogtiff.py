from datetime import datetime, date
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
import zarr
import math

INPUT_ZARR = "/data_3/scratch/francesco/ndvi_processed.zarr"
PIXEL_INPUT = "/data_3/scratch/francesco/new_zarr_bol.zarr"

ds_p = xr.open_zarr(PIXEL_INPUT)


ds = xr.open_zarr(INPUT_ZARR)

pixel = ds_p["pixel"].values

pixel = pixel[:1000000]

ndvi_layer =  ds["ndvi_processed"].isel(time = 100).values

zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

z = zarr.open(zarr_path, mode="r")

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# ----- center cooridnates  -----
center_x, center_y = 2694491.82, 1126023.20
# Rectangle corners (UL and BR)
UL_x, UL_y = center_x - 6500, center_y - 6500 
BR_x, BR_y = center_x + 6500, center_y + 6500


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

sel = masked_idx_in_window[is_masked].tolist()
sel = sel[:1000000]

values_non_gapfilled = np.empty(n_masked_in_window, dtype=float)
window_non_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)

is_masked

true_indices = np.flatnonzero(is_masked)

# Take the last 14 of those indices
last_14_true_indices = true_indices[-14:]

# Set them to False
is_masked[last_14_true_indices] = False

window_non_gapfilled[is_masked] = ndvi_layer
window_non_gapfilled = window_non_gapfilled.reshape((win_rows, win_cols))





import rasterio
from rasterio.transform import from_origin
import numpy as np

# Your existing array
arr = window_non_gapfilled.astype('float32')


x_min = 2694491 - 6500
y_max = 1126023 - 6500
pixel_width = 10
pixel_height = 10

transform = from_origin(x_min, y_max, pixel_width, pixel_height)

# Output COG filename
cog_tif = 'subset_example.tif'

# One-step creation of COG
with rasterio.open(
    cog_tif,
    'w',
    driver='COG',
    height=arr.shape[0],
    width=arr.shape[1],
    count=1,
    dtype=arr.dtype,
    crs='EPSG:2056',  # Swiss LV95 CRS
    transform=transform,
    nodata=np.nan,
    compress='deflate',
    tiled=True
) as dst:
    dst.write(arr, 1) 

import rasterio
import matplotlib.pyplot as plt


with rasterio.open(cog_tif) as src:
    print("CRS:", src.crs)
    print("Width, Height:", src.width, src.height)
    print("Transform:", src.transform)
    print("NoData value:", src.nodata)

    # Read the full array
    arr = src.read(1)

# Quick plot
plt.imshow(arr, cmap='viridis')
plt.colorbar()
plt.show()

