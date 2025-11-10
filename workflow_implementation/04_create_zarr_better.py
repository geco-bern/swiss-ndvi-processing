# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/04_create_zarr_better.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/output/log/zarr_daily_creation_better_3.log &

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import zarr
from numcodecs import Blosc  # compression

start_time = time.time()

# ================= CONFIG =================
SRC_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
OUT_ZARR = "/data_3/scratch/francesco/zarr_demo_pixel_chunked_300000.zarr"
MASK_PATH = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

MAX_PIXELS = 105715396
CHUNK_PIXELS = 300_000 
# ==========================================

print("Opening source Zarr:", SRC_ZARR)
ds = zarr.open_group(SRC_ZARR, mode="r")
params = ds["params"]
params_lower = params["params_lower"]
params_upper = params["params_upper"]
ndvi = ds["ndvi"]

dates = pd.to_datetime([d.decode("utf-8") for d in ds["dates"][:]])
print(f"Loaded NDVI with shape {ndvi.shape}, {len(dates)} unique dates.")

daily_dates = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
n_times = len(daily_dates)
print(f"Generated {n_times} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

date_to_index = {d.date(): i for i, d in enumerate(dates)}

sel_1 = np.linspace(0,MAX_PIXELS,MAX_PIXELS, dtype = int)
print(f"Using {MAX_PIXELS} pixels (available {MAX_PIXELS})")

# ---------- Create Zarr v2 store ----------
os.makedirs(os.path.dirname(OUT_ZARR), exist_ok=True)
root = zarr.open_group(OUT_ZARR, mode="w", zarr_format=2)

# Dates stored as YYYYMMDD int32
dates_int = np.array([int(d.strftime("%Y%m%d")) for d in daily_dates], dtype=np.int32)
root.create_dataset("dates", data=dates_int, shape=dates_int.shape, dtype=np.int32, chunks=(n_times,))

# compressor
blosc = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

# main datasets (1 pixel × full time series chunks)
ndvi_ds = root.create_dataset("ndvi",
                              shape=(n_times, MAX_PIXELS),
                              chunks=(n_times, CHUNK_PIXELS),
                              dtype=np.int16,
                              compressor=blosc)

last_dates_ds = root.create_dataset("last_dates",
                                    shape=(8, MAX_PIXELS),
                                    chunks=(8, CHUNK_PIXELS),
                                    dtype=np.int32,
                                    compressor=blosc)

params_group = root.create_group("params")
params_lower_ds = params_group.create_dataset("params_lower",
                                              shape=(MAX_PIXELS, 6),
                                              chunks=(CHUNK_PIXELS, 6),
                                              dtype=np.float32,
                                              compressor=blosc)
params_upper_ds = params_group.create_dataset("params_upper",
                                              shape=(MAX_PIXELS, 6),
                                              chunks=(CHUNK_PIXELS, 6),
                                              dtype=np.float32,
                                              compressor=blosc)

print("Created Zarr store and datasets. Beginning chunked write.")

# Each chunk = one pixel (or more if CHUNK_PIXELS > 1)
chunks = [sel_1[i:i + CHUNK_PIXELS] for i in range(0, MAX_PIXELS, CHUNK_PIXELS)]
print(f"Total chunks to write: {len(chunks)} (chunk size {CHUNK_PIXELS})")


# ---------- Write loop ----------
for idx, chunk_pixels in enumerate(chunks):
    print(f"\n--- chunk {idx+1}/{len(chunks)}: writing {len(chunk_pixels)} pixel(s) ---")
    col_start = idx * CHUNK_PIXELS
    col_end = col_start + len(chunk_pixels)

    print(f"writing pixels from {col_start} to {col_end}")

    if ndvi.shape[0] == len(dates):

        # shape is (time, pixel)

        ndvi_subset = ndvi.get_orthogonal_selection((slice(None), chunk_pixels))
    else:

        # shape is (pixel, time)

        ndvi_subset = ndvi.get_orthogonal_selection((chunk_pixels, slice(None))).T
        
    ndvi_subset = ndvi_subset.astype(np.int16)


    daily_ndvi = np.full((n_times, len(chunk_pixels)), 32767, dtype=np.int16)
    for i, day in enumerate(daily_dates):
        d = day.date()
        if d in date_to_index:
            daily_ndvi[i, :] = ndvi_subset[date_to_index[d], :]

    params_lower_subset = params_lower.get_orthogonal_selection((chunk_pixels, slice(None)))
    params_upper_subset = params_upper.get_orthogonal_selection((chunk_pixels, slice(None)))

    ndvi_ds[:, col_start:col_end] = daily_ndvi.astype(np.int16)
    params_lower_ds[col_start:col_end, :] = params_lower_subset
    params_upper_ds[col_start:col_end, :] = params_upper_subset
    last_dates_ds[:, col_start:col_end] = np.full((8, len(chunk_pixels)), 19000101, dtype=np.int32)

    print(f"Written pixel(s) {idx}")

# ---------- Add coordinate metadata ----------
pixel_coord = root.create_array(
    name="pixel",
    shape=(MAX_PIXELS,),
    chunks=(min(CHUNK_PIXELS, MAX_PIXELS),),
    dtype=np.int64
)
pixel_coord[:] = np.arange(MAX_PIXELS, dtype=np.int64)

root["dates"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
root["pixel"].attrs["_ARRAY_DIMENSIONS"] = ["pixel"]
root["ndvi"].attrs["_ARRAY_DIMENSIONS"] = ["time", "pixel"]
root["last_dates"].attrs["_ARRAY_DIMENSIONS"] = ["band", "pixel"]
root["params/params_lower"].attrs["_ARRAY_DIMENSIONS"] = ["pixel", "param"]
root["params/params_upper"].attrs["_ARRAY_DIMENSIONS"] = ["pixel", "param"]

root.attrs.update({
    "title": "Daily NDVI lookup table (Zarr v2, pixel-chunked)",
    "institution": "Your Lab or Organization",
    "source": SRC_ZARR,
    "history": f"Created on {pd.Timestamp.now()}",
    "description": "Daily NDVI data, median logistic curves, and model parameters for selected pixels (chunked per pixel)."
})

# ---------- Consolidate metadata ----------
zarr.consolidate_metadata(OUT_ZARR)

total_time = time.time() - start_time
print(f"\n✅ Finished writing {MAX_PIXELS} pixels into {OUT_ZARR} in {total_time:.1f} s")

# ---------- Quick verification ----------
import xarray as xr
ds = xr.open_zarr(OUT_ZARR, consolidated=True)
print(ds)
