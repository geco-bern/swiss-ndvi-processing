# python /home/francesco/data_scratch/swiss-ndvi-processing/demo/script/data_extraction/extract_highland_evergreen.py

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

z = zarr.open(zarr_path, mode = "r")




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

compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

ndvi_ds = zarr.create_array(
    name="ndvi",
    store='/home/francesco/data_scratch/swiss-ndvi-processing/highland_evergreen.zarr',
    shape=(T, len(sel)),
    chunks=(1, len(sel)),
    dtype="float32",
    fill_value=np.nan,
    compressors=compressors,
    zarr_format=3,
)

values_batch = z[sel, :].astype(float).T
ndvi_ds[:, :] = values_batch




