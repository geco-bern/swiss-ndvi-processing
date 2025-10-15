# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/00_create_demo_zarr_folder.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_creation.log 

import numpy as np
import math
import zarr
import pandas as pd
import torch
import os

import time

start_time = time.time()

# =====================================================
#  Load Source Dataset
# =====================================================
ds = zarr.open_group("/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr", mode="r")
params = ds["params"]
params_lower = params["params_lower"]
params_upper = params["params_upper"]
ndvi = ds["ndvi"]
dates = pd.to_datetime([d.decode("utf-8") for d in ds["dates"][:]])

T_SCALE = 1.0 / 365.0
t = torch.tensor(dates.dayofyear * T_SCALE, dtype=torch.float32)

# =====================================================
#  Raster Info
# =====================================================
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

# =====================================================
#  Pixel Selection
# =====================================================
def extract_pixel(UL_x, UL_y, BR_x, BR_y):
    x_min, x_max = min(UL_x, BR_x), max(UL_x, BR_x)
    y_min, y_max = min(UL_y, BR_y), max(UL_y, BR_y)
    col_min = int(math.floor((x_min - left) / px))
    col_max = int(math.floor((x_max - left) / px))
    row_min = int(math.floor((top - y_max) / px))
    row_max = int(math.floor((top - y_min) / px))
    col_min = max(0, min(width - 1, col_min))
    col_max = max(0, min(width - 1, col_max))
    row_min = max(0, min(height - 1, row_min))
    row_max = max(0, min(height - 1, row_max))

    print(f"Window cols {col_min}..{col_max}, rows {row_min}..{row_max}")

    mask = np.load(mask_path)
    mask_flat = mask.ravel(order="C")
    masked_positions = np.flatnonzero(mask_flat)
    idx_map = np.full(mask_flat.shape[0], -1, dtype=np.int64)
    idx_map[masked_positions] = np.arange(masked_positions.size, dtype=np.int64)

    rows = np.arange(row_min, row_max + 1, dtype=np.int64)
    cols = np.arange(col_min, col_max + 1, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    full_flat_idx = (rr * width + cc).ravel()
    masked_idx_in_window = idx_map[full_flat_idx]
    sel = masked_idx_in_window[masked_idx_in_window >= 0].tolist()
    print(f"Selected {len(sel)} masked pixels")
    return sel

# =====================================================
#  Extract subset
# =====================================================
center_x, center_y = 2694491.82, 1126023.20
sel_1 = extract_pixel(center_x - 6500, center_y - 6500,
                      center_x + 6500, center_y + 6500)

subset_save_path = "/data_2/scratch/francesco/zarr_demo/"
os.makedirs(subset_save_path, exist_ok=True)

n_dates = len(dates)
n_pixels = len(sel_1)
print(f"Subset has {n_pixels} pixels and {n_dates} dates.")

# =====================================================
#  Pull NDVI subset
# =====================================================
shape = ndvi.shape
print(f"NDVI array shape: {shape}")
if shape[0] == n_dates:
    print("Detected (time, pixel)")
    ndvi_subset = ndvi.get_orthogonal_selection((slice(None), sel_1))
else:
    print("Detected (pixel, time)")
    ndvi_subset = ndvi.get_orthogonal_selection((sel_1, slice(None))).T

counter_subset = np.zeros(n_dates, dtype=np.int16)

# =====================================================
# Save subset (Zarr v2 API)
# =====================================================
subset_group = zarr.open_group(subset_save_path, mode="w")

# --- Dates ---
dates_data = np.array([d.strftime("%Y-%m-%d").encode("utf-8") for d in dates], dtype="S10")
subset_group.create_dataset(
    "dates",
    data=dates_data,
    shape=dates_data.shape,
    dtype="S10"
)

# --- NDVI subset ---
subset_group.create_dataset(
    "ndvi",
    data=ndvi_subset,
    shape=ndvi_subset.shape,
    chunks=(1, ndvi_subset.shape[1]),
    dtype=np.int16,
)

# --- Counter array ---
subset_group.create_dataset(
    "counter",
    data=counter_subset,
    shape=counter_subset.shape,
    dtype=np.int16,
)

# --- Params ---
param_group = subset_group.create_group("params")

params_lower_subset = params_lower.get_orthogonal_selection((sel_1, slice(None)))
params_upper_subset = params_upper.get_orthogonal_selection((sel_1, slice(None)))

param_group.create_dataset(
    "params_lower",
    data=params_lower_subset,
    shape=params_lower_subset.shape,
    dtype=np.float32,
)
param_group.create_dataset(
    "params_upper",
    data=params_upper_subset,
    shape=params_upper_subset.shape,
    dtype=np.float32,
)

# --- Last dates (8 × n_pixels, filled with "1900-01-01") ---
last_dates_str = b"1900-01-01"
last_dates_data = np.full((8, n_pixels), last_dates_str, dtype="S10")

subset_group.create_dataset(
    "last_dates",
    data=last_dates_data,
    shape=last_dates_data.shape,
    dtype="S10",
    chunks=(1, n_pixels),
)


print(f"✅ Subset saved to: {subset_save_path}")

end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")
