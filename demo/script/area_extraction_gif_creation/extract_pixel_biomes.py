# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/script/area_extraction_gif_creation/extract_pixel_biomes.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/extraction.log 2>&1 &

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
import os
import shutil
import geopandas as gpd
from rasterio import features
from affine import Affine
from shapely.geometry import Point


zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

z = zarr.open(zarr_path, mode="r")

print("Zarr shape:", z.shape)


# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

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

    mid = len(sel) // 2

    start = max(0, mid - 12)
    end   = min(len(sel), mid + 13)   

    sel_window = sel[start:end]
    return(sel_window)

# lowland broadleaf
sel_1 = extract_pixel( UL_x = 2694027.49, UL_y = 1123239.84, BR_x = 2693027.49, BR_y = 1122239.84) # to fix
# hihgland broadleaf
sel_2 = extract_pixel( UL_x = 2694027.49, UL_y = 1123239.84, BR_x = 2693027.49, BR_y = 1122239.84)
# lowland evergreen
sel_3 = extract_pixel( UL_x =2613028.38, UL_y = 1127777.24, BR_x = 2612028.38, BR_y = 1126777.24)
# hihgland evergreen
sel_4 = extract_pixel( UL_x = 2782037.00, UL_y = 1183475.00, BR_x = 2783037.00, BR_y = 1184475.00)

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

sel_5 = masked_idx_in_window[is_masked].tolist()

gdf = gpd.read_file("/home/francesco/data_scratch/swiss-ndvi-processing/demo/area_visulization/polygon_drought.kml", driver="KML")
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

sel_6 = masked_idx_in_window[is_masked].tolist()

# ----- load polygon from SHP -----
gdf = gpd.read_file("/home/francesco/data_scratch/swiss-ndvi-processing/demo/area_visulization/storm_reference_burglind.shp")
gdf = gdf.to_crs(epsg=2056)   # Swiss LV95
gdf["geometry"] = gdf["geometry"].buffer(0)

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

# ----- rasterize polygon to mask (fast) -----
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

sel_7 = masked_idx_in_window[is_masked].tolist()


all_sel = sel_1 + sel_2 + sel_3 + sel_4 + sel_5 + sel_6 + sel_7

# Convert to numpy array
all_sel = np.array(all_sel, dtype=np.int64)

subset_path = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/pixel_biomes.zarr"
if os.path.exists(subset_path):
    shutil.rmtree(subset_path)

store = zarr.open(subset_path, mode="w")

# Create dataset: (100 pixels, all timesteps)
out = store.create(
    name="ndvi",
    shape=(len(all_sel), z.shape[1]),   # (100, time=1084)
    chunks=(len(all_sel), z.chunks[1]), # chunk over time
    dtype=z.dtype
)

# Copy data in manageable chunks over time axis
chunk_size = z.chunks[1]   # how many timesteps per chunk
for t in range(0, z.shape[1], chunk_size):
    t_end = min(t + chunk_size, z.shape[1])
    print(f"Copying timesteps {t}:{t_end} ...")
    out[:, t:t_end] = z[all_sel, t:t_end]

print("✅ Saved new Zarr with 175 pixels:", subset_path, "shape:", out.shape)