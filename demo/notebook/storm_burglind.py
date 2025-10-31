#  nohup python -u  /home/francesco/data_scratch/swiss-ndvi-processing/demo/notebook/storm_burglind.py >  /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/burglind.log 2>&1 &

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


# data loading and raster initialization
# ----- Config -----

# fitting and smoothing
# ----- seasonal cycle fitting -----
ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/sample_seasonal_cycle_parameter_preds.zarr")
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

param_iqr = 1.02
bottom_iqr = 0.2
upper_iqr = 0.8
window_length = 14
polyorder = 2

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

# ----- load polygon from Shapefile -----
shapefile_path = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/shapefiles/storm_reference_burglind.shp"
gdf = gpd.read_file(shapefile_path)

# Ensure it's in Swiss LV95 (EPSG:2056)
if gdf.crs is None:
    raise ValueError("Shapefile has no CRS defined! Please assign the correct CRS before reprojecting.")
if gdf.crs.to_epsg() != 2056:
    gdf = gdf.to_crs(epsg=2056)

# ---- Fix invalid geometries before union ----
gdf["geometry"] = gdf["geometry"].buffer(0)  # cleans self-intersections, etc.

# Optional: check for remaining invalid geometries
invalid = gdf[~gdf.is_valid]
if not invalid.empty:
    print(f"Warning: {len(invalid)} invalid geometries remain after repair!")

# Combine into one polygon
polygon = gdf.union_all()   # new method (Shapely 2.0+)



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

out_txt = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/burlgind_pixels.txt"

np.savetxt(out_txt, sel, fmt="%d")
print(sel)