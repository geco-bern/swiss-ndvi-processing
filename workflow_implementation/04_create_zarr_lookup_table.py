# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/04_create_zarr_lookup_table.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/output/log/zarr_daily_creation_extended.log &

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import zarr
from zarr.codecs import BloscCodec

start_time = time.time()

# ================= CONFIG =================
SRC_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
OUT_ZARR = "/data_3/scratch/francesco/zarr_demo_daily.zarr"  # single top-level zarr store
MASK_PATH = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

MAX_PIXELS = 1_000_000   # limit to 1 million pixels
CHUNK_COLS = 5000        # write in blocks of this many pixels
# ==========================================

print("Opening source Zarr:", SRC_ZARR)
ds = zarr.open_group(SRC_ZARR, mode="r")
params = ds["params"]
params_lower = params["params_lower"]
params_upper = params["params_upper"]
ndvi = ds["ndvi"]
# original dates in the source (assume stored as bytes strings)
dates = pd.to_datetime([d.decode("utf-8") for d in ds["dates"][:]])
print(f"Loaded NDVI with shape {ndvi.shape}, {len(dates)} unique dates.")

# generate daily date range
daily_dates = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
n_times = len(daily_dates)
print(f"Generated {n_times} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

# map original dates to indices
date_to_index = {d.date(): i for i, d in enumerate(dates)}

# raster info (kept from your script)
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# double logistic function (unchanged)
def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    t = t[:, None]
    sos = sos.T; mat_minus_sos = mat_minus_sos.T; sen = sen.T
    eos_minus_sen = eos_minus_sen.T; M = M.T; m = m.T
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10))
    ndvi_curve = (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m
    return ndvi_curve

# pixel selection (kept from your script)
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

    mask = np.load(MASK_PATH)
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
    sel = sel[:MAX_PIXELS]
    print(f"Selected {len(sel)} masked pixels")
    return sel

# choose your window (same as before)
center_x, center_y = 2694491.82, 1126023.20
sel_1 = extract_pixel(center_x - 6500, center_y - 6500,
                      center_x + 6500, center_y + 6500)

# actual pixels to store
n_pixels_available = len(sel_1)
n_pixels = min(n_pixels_available, MAX_PIXELS)
sel_1 = sel_1[:n_pixels]
print(f"Using {n_pixels} pixels (available {n_pixels_available})")

# ---------- Create one Zarr v3 store ----------
os.makedirs(os.path.dirname(OUT_ZARR), exist_ok=True)
root = zarr.open_group(OUT_ZARR, mode="w")

# Dates stored as YYYYMMDD int32
dates_int = np.array([int(d.strftime("%Y%m%d")) for d in daily_dates], dtype=np.int32)
dates_arr = root.create_array("dates", shape=dates_int.shape, dtype=np.int32)
dates_arr[:] = dates_int

# compressor: zarr v3 codec (BloscCodec), pass single codec instance via 'compressors'
blosc_codec = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

# main datasets (use create_array; assign later in slices)
chunk_cols = CHUNK_COLS
ndvi_ds = root.create_array("ndvi",
                            shape=(n_times, n_pixels),
                            chunks=(1, min(chunk_cols, n_pixels)),
                            dtype=np.int16,
                            compressors=blosc_codec)

median_ndvi_ds = root.create_array("median_ndvi",
                                   shape=(n_times, n_pixels),
                                   chunks=(1, min(chunk_cols, n_pixels)),
                                   dtype=np.int16,
                                   compressors=blosc_codec)

# last_dates stored as YYYYMMDD int32 (19000101 placeholder)
last_dates_ds = root.create_array("last_dates",
                                  shape=(8, n_pixels),
                                  chunks=(1, min(chunk_cols, n_pixels)),
                                  dtype=np.int32,
                                  compressors=blosc_codec)

# params group
params_group = root.create_group("params")
params_lower_ds = params_group.create_array("params_lower",
                                           shape=(n_pixels, 6),
                                           chunks=(min(chunk_cols, n_pixels), 6),
                                           dtype=np.float32,
                                           compressors=blosc_codec)
params_upper_ds = params_group.create_array("params_upper",
                                           shape=(n_pixels, 6),
                                           chunks=(min(chunk_cols, n_pixels), 6),
                                           dtype=np.float32,
                                           compressors=blosc_codec)

print("Created Zarr store and datasets. Beginning chunked write.")

# process in manageable chunks so memory stays sane
chunk_size = CHUNK_COLS
chunks = [sel_1[i:i + chunk_size] for i in range(0, n_pixels, chunk_size)]
print(f"Total chunks to write: {len(chunks)} (chunk size {chunk_size})")

# precompute DOY t for median calculation
doy = np.array([d.timetuple().tm_yday for d in daily_dates], dtype=np.int16)
doy[doy == 366] = 365
T_SCALE = 1.0 / 365.0
t = torch.tensor(doy * T_SCALE, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for idx, chunk_pixels in enumerate(chunks):
    print(f"\n--- chunk {idx+1}/{len(chunks)}: writing {len(chunk_pixels)} pixels ---")
    col_start = idx * chunk_size
    col_end = col_start + len(chunk_pixels)

    # extract NDVI subset from source (safe selection)
    if ndvi.shape[0] == len(dates):
        ndvi_subset = ndvi.get_orthogonal_selection((slice(None), chunk_pixels))
    else:
        ndvi_subset = ndvi.get_orthogonal_selection((chunk_pixels, slice(None))).T

    # fill daily_ndvi with missing sentinel 32767 then overwrite available dates
    daily_ndvi = np.full((n_times, ndvi_subset.shape[1]), 32767, dtype=np.int16)
    for i, day in enumerate(daily_dates):
        d = day.date()
        if d in date_to_index:
            daily_ndvi[i, :] = ndvi_subset[date_to_index[d], :]

    # compute median NDVI curve for this chunk
    params_lower_subset = params_lower.get_orthogonal_selection((chunk_pixels, slice(None)))
    params_upper_subset = params_upper.get_orthogonal_selection((chunk_pixels, slice(None)))

    # convert to torch and compute curves on device
    params_lower_t = torch.tensor(params_lower_subset, dtype=torch.float32, device=t.device)
    params_upper_t = torch.tensor(params_upper_subset, dtype=torch.float32, device=t.device)

    upper_curve = double_logistic_function(t, params_upper_t)
    lower_curve = double_logistic_function(t, params_lower_t)
    median_curve = 0.5 * (upper_curve + lower_curve)
    median_curve_scaled = torch.clamp(median_curve * 10000.0, -32768, 32767).short().cpu().numpy()

    # write slices into the big arrays
    ndvi_ds[:, col_start:col_end] = daily_ndvi
    median_ndvi_ds[:, col_start:col_end] = median_curve_scaled
    params_lower_ds[col_start:col_end, :] = params_lower_subset
    params_upper_ds[col_start:col_end, :] = params_upper_subset

    # write last_dates placeholder as YYYYMMDD int
    last_dates_ds[:, col_start:col_end] = np.full((8, len(chunk_pixels)), 19000101, dtype=np.int32)

    print(f"Written cols {col_start}..{col_end-1} (pixels {len(chunk_pixels)})")

total_time = time.time() - start_time
print(f"\n✅ Finished writing {n_pixels} pixels into {OUT_ZARR} in {total_time:.1f} s")
