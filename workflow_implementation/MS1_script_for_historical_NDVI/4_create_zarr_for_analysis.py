import numpy as np
import math
import zarr
import pandas as pd
import torch
import os
import time
from dask.distributed import Client, LocalCluster
import xarray as xr
import dask.array as da

# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/4_create_zarr_for_analysis.py >  /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/create_historical_zarr.log &


# !!! IMPORTANT for some unkown reason, there are no data in swisstopo for 2021-02-05, 
# the NDVI will not be downloaded but the date is still considered (see fabian log)
#'/vsicurl/https://data.geo.admin.ch/ch.swisstopo.swisseo_s2-sr_v100/2021-02-05t102221/ch.swisstopo.swisseo_s2-sr_v100_mosaic_2021-02-05t102221_masks-10m.tif' 
# does not exist in the file system, and is not recognized as a supported dataset name.

# NOT RUN THIS COMMENTED PART (already fix it)
"""
import zarr
import numpy as np
import pandas as pd

ZARR_PATH = "/data_3/scratch/francesco/processed/all_ndvi_dataset_temporal.zarr"

# Open Zarr in append/edit mode
root = zarr.open_group(ZARR_PATH, mode="a")

# Load existing date array
dates_raw = root["date"][:]              # dtype S10
dates = pd.to_datetime(dates_raw.astype(str))

print("Original length:", len(dates))

# Identify all indices equal to 2021-02-05
mask = dates != pd.Timestamp("2021-02-05")
dates_filtered = dates_raw[mask]         # still dtype S10

print("New length:", len(dates_filtered))

# Remove and recreate date array
del root["date"]

root.create_array(
    name="date",
    dtype="S10",
    shape=(len(dates_filtered),),
    chunks=(len(dates_filtered),),
)

root["date"][:] = dates_filtered

print("Removed date 2021-02-05 successfully.")"""


SRC_ZARR = "/data_3/scratch/francesco/processed/all_ndvi_dataset_temporal.zarr"
OUT_ZARR  = "/data_3/scratch/francesco/zarr_ready_all_pixels.zarr"

# !!! before must run workflow_implementation/demo up to update date


# SETUP PARALLELIZATION CLUSTER
client = Client(
    n_workers=50,
    threads_per_worker=1,
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':2231'
)  # start distributed scheduler locally.
client.dashboard_link

ds0 = zarr.open_group(SRC_ZARR, mode="r")
ndvi_z = ds0["ndvi"]
ndvi_da = da.from_zarr(ndvi_z) 

ndsi_z= ds0["ndsi"]
ndsi_da = da.from_zarr(ndsi_z) 


# Decode dates into a small in-memory coordinate
dates = pd.to_datetime([d.decode("utf-8") for d in ds0["date"][:]])

# =====================================================
#  Generate Daily Date Range (in-memory index only)
# =====================================================
daily_dates = pd.date_range(start=dates.min(), end=max(np.datetime64("2025-11-30"),dates.max()), freq="D")
print(f"Generated {len(daily_dates)} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

obs_dates = daily_dates.isin(dates)


ndvi_xr = xr.DataArray(ndvi_da, dims=("pixel", "date"),  coords={"date": dates, "pixel": np.arange(ndvi_da.shape[0])})
ndsi_xr = xr.DataArray(ndsi_da, dims=("pixel", "date"),  coords={"date": dates, "pixel": np.arange(ndsi_da.shape[0])})

# =====================================================
# Filter NDVI using NDSI > 0.43
# Set NDVI = 32767 where NDSI > 0.43
# =====================================================

MASK_VALUE = np.int16(32767)

# Apply mask lazily with Dask
ndvi_xr_filtered = ndvi_xr.where(ndsi_xr <= 0.43, other=MASK_VALUE)

# Replace main NDVI with filtered version
ndvi_xr = ndvi_xr_filtered



# =====================================================
#  Reindex NDVI to daily (lazy), fill with 32767 (int16)
# =====================================================

def _dedup_date_coord(da: xr.DataArray, how: str = "first") -> xr.DataArray:
    """
    Ensure da's date coordinate is unique.
    how: 'first' | 'last' | 'mean' | 'median' | 'max'
    - 'first'/'last': keep first/last occurrence (fast, no compute)
    - reductions: aggregate duplicates lazily with Dask
    """
    idx = pd.Index(da["date"].values)
    if not idx.has_duplicates:
        return da
    if how in ("first", "last"):
        keep_mask = ~idx.duplicated(keep=how)
        return da.isel(date=np.nonzero(keep_mask)[0])
    if how == "mean":
        return da.groupby("date").mean("date")
    if how == "median":
        return da.groupby("date").median("date")
    if how == "max":
        return da.groupby("date").max("date")
    raise ValueError(f"Unsupported how={how}")

# Deduplicate time first, then reindex to daily
# Change how='first' to 'mean'/'median'/'max' as needed
ndvi_sel_nodup = _dedup_date_coord(ndvi_xr, how="first")
ndvi_daily = ndvi_sel_nodup.astype(np.int16).reindex(date=daily_dates, method=None, fill_value=np.int16(32767))


obs_dates_xr = xr.DataArray(
    obs_dates,
    dims=("date",),
    coords={"date": daily_dates},
    name="obs_dates"
)


lookuptable_src = "/data_3/francesco/lookup_table_median_ndvi.zarr"

lookuptable = xr.open_zarr(lookuptable_src)

lookuptable_arr = lookuptable["median_ndvi"]

doy = daily_dates.dayofyear

# remove leap year if encountered
doy_array_fixed = np.where(doy.values == 366, 365, doy.values)

median_ndvi = lookuptable.sel(doy=xr.DataArray(doy_array_fixed, dims="date"))


DATE_CHUNKS = min(len(ndvi_daily.date), 4000)
PIXEL_CHUNKS = min(len(ndvi_daily.pixel), 4000)


ndvi_stack = ndvi_daily.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
median_ndvi = median_ndvi.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
obs_date_stack = obs_dates_xr.chunk({"date": DATE_CHUNKS})

out_ds = xr.Dataset({
    "ndvi": ndvi_stack,
    "median_ndvi": median_ndvi["median_ndvi"],
    "obs_date": obs_date_stack
},
coords={
    "pixel": ndvi_stack.pixel,
    "date": daily_dates
})

for v in out_ds.data_vars:
    out_ds[v].encoding.pop("compressor", None)
    out_ds[v].encoding.setdefault("chunks", None)
for c in out_ds.coords:
    out_ds[c].encoding.pop("compressor", None)
    out_ds[c].encoding.setdefault("chunks", None)

OUT_ZARR = "/data_3/scratch/francesco/zarr_to_historical_all_pixels.zarr"
out_ds.to_zarr(OUT_ZARR, mode="w", consolidated=True)

print("✅ Done")

