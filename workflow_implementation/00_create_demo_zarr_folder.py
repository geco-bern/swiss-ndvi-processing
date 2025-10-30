# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/00_create_demo_zarr_folder.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/zarr_daily_creation.log &

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
SRC_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
OUT_ZARR = "/data_2/scratch/francesco/zarr_demo_daily/"  # NEW daily Zarr folder

ds = zarr.open_group(SRC_ZARR, mode="r")
params = ds["params"]
params_lower = params["params_lower"]
params_upper = params["params_upper"]
ndvi = ds["ndvi"]
dates = pd.to_datetime([d.decode("utf-8") for d in ds["dates"][:]])

print(f"Loaded NDVI with shape {ndvi.shape}, {len(dates)} unique dates.")

# =====================================================
#  Generate Daily Date Range
# =====================================================
daily_dates = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
print(f"Generated {len(daily_dates)} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

# Map original dates to indices for lookup
date_to_index = {d.date(): i for i, d in enumerate(dates)}

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

n_pixels = len(sel_1)
print(f"Subset has {n_pixels} pixels.")

# =====================================================
#  Split into 1000-pixel chunks and save each as a separate Zarr file
# =====================================================
import math

CHUNK_SIZE = 1000
os.makedirs(OUT_ZARR, exist_ok=True)

# Split pixel indices
chunks = [sel_1[i:i + CHUNK_SIZE] for i in range(0, len(sel_1), CHUNK_SIZE)]
print(f"Total chunks to create: {len(chunks)}")

for idx, chunk_pixels in enumerate(chunks):
    print(f"\n--- Processing chunk {idx+1}/{len(chunks)} with {len(chunk_pixels)} pixels ---")

    # =====================================================
    # Extract NDVI subset for this chunk
    # =====================================================
    if ndvi.shape[0] == len(dates):
        ndvi_subset = ndvi.get_orthogonal_selection((slice(None), chunk_pixels))
    else:
        ndvi_subset = ndvi.get_orthogonal_selection((chunk_pixels, slice(None))).T

    # --- Allocate daily NDVI ---
    daily_ndvi = np.full((len(daily_dates), ndvi_subset.shape[1]), 32767, dtype=np.int16)

    # Fill available days
    for i, day in enumerate(daily_dates):
        d = day.date()
        if d in date_to_index:
            daily_ndvi[i, :] = ndvi_subset[date_to_index[d], :]

    # =====================================================
    # Save subset (Zarr v2 API)
    # =====================================================
    subset_path = os.path.join(OUT_ZARR, f"chunk_{idx:04d}.zarr")
    subset_group = zarr.open_group(subset_path, mode="w")

    # --- Dates ---
    dates_data = np.array([d.strftime("%Y-%m-%d").encode("utf-8") for d in daily_dates], dtype="S10")
    subset_group.create_dataset("dates", data=dates_data, shape=dates_data.shape, dtype="S10")

    # --- NDVI subset ---
    subset_group.create_dataset(
        "ndvi",
        data=daily_ndvi,
        shape=daily_ndvi.shape,
        chunks=(1, daily_ndvi.shape[1]),
        dtype=np.int16,
    )

    # --- Counter array ---
    subset_group.create_dataset(
        "counter",
        data=np.zeros(len(daily_dates), dtype=np.int16),
        shape=(len(daily_dates),),
        dtype=np.int16,
    )

    # --- Params ---
    param_group = subset_group.create_group("params")
    params_lower_subset = params_lower.get_orthogonal_selection((chunk_pixels, slice(None)))
    params_upper_subset = params_upper.get_orthogonal_selection((chunk_pixels, slice(None)))

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

    # --- Last dates (dummy placeholder) ---
    last_dates_str = b"1900-01-01"
    last_dates_data = np.full((8, len(chunk_pixels)), last_dates_str, dtype="S10")

    subset_group.create_dataset(
        "last_dates",
        data=last_dates_data,
        shape=last_dates_data.shape,  
        dtype="S10",
        chunks=(1, len(chunk_pixels)),
    )

    print(f"✅ Created {subset_path}")

print(f"Finished creating {len(chunks)} Zarr files at {OUT_ZARR}")
print(f"Total time: {time.time() - start_time:.1f} s")
