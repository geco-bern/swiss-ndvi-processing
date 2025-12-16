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
from datetime import datetime, date



SRC_ZARR = "/data_3/scratch/francesco/processed/ndvi_dataset_temporal.zarr"
OUT_ZARR_TMP  = "/data_3/scratch/francesco/demo.zarr"
historical_ndvi_src = "data_for_demo/historic_ndvi.zarr"
OUT_ZARR = "data_for_demo/merged_ndvi.zarr"
lookuptable_src = "data_for_demo/lookup_table.zarr"


# SETUP PARALLELIZATION CLUSTER
client10 = Client(
    n_workers=10,
    threads_per_worker=1,
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':2231'
)  # start distributed scheduler locally.
client10.dashboard_link

# =====================================================
#  Load Source Dataset lazily (no xarray metadata needed)
# =====================================================

ds0 = zarr.open_group(SRC_ZARR, mode="r")
ndvi_z = ds0["ndvi"]
ndvi_da = da.from_zarr(ndvi_z)     # lazy
         # equivalently sel() but much slower

dates = da.from_zarr(ds0["date"]).astype("datetime64[D]")

dates = dates.compute()

start_dates = np.datetime64("2018-06-01", "D")
end_dates = np.datetime64("2018-06-05", "D")

daily_dates = pd.date_range(start=start_dates, end=end_dates, freq="D")
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

# =====================================================
#  Assemble Dataset and write
# =====================================================
out_ds = xr.Dataset(
    {
        "ndvi": ndvi_daily,
        "obs" : obs_dates_xr
    }
)


DATE_CHUNKS = len(daily_dates)
PIXEL_CHUNKS = 4000
out_ds = out_ds.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})


# Write to Zarr
os.makedirs(OUT_ZARR_TMP, exist_ok=True)
print(f"Writing lazily computed Dataset to {OUT_ZARR_TMP} with Dask...")
out_ds.to_zarr(OUT_ZARR_TMP, mode="w", consolidated=True)
print("✅ Done")


historical_ndvi = xr.open_zarr(historical_ndvi_src)

# historical filtered
ndvi_historic = historical_ndvi["ndvi_processed"].sel(
    date= slice(None, "2018-05-31")
).rename("ndvi")

obs_date_historical = historical_ndvi["obs_date"].sel( date= slice(None, "2018-05-31"))


# new data
ds_to_stack = xr.open_zarr(OUT_ZARR_TMP)
ndvi_new = ds_to_stack["ndvi"].rename("ndvi") 
obs_date_new = ds_to_stack["obs"].rename("obs_date")

# stack along time
ndvi_stack = xr.concat([ndvi_historic, ndvi_new], dim="date").sortby("date")

obs_date_stack = xr.concat([obs_date_historical, obs_date_new], dim="date").sortby("date")


date_stack =  xr.concat([historical_ndvi["date"].sel(date= slice(None, "2018-05-31")), ds_to_stack["date"]], dim="date").sortby("date")

date_stack = date_stack.astype("datetime64[D]")

# extract the mean of lower and upper bands


lookuptable = xr.open_zarr(lookuptable_src)

lookuptable_arr = lookuptable["median_ndvi"]

doy = date_stack.dt.dayofyear

# remove leap year if encountered
doy_array_fixed = np.where(doy.values == 366, 365, doy.values)

median_ndvi = lookuptable.sel(doy=xr.DataArray(doy_array_fixed, dims="date"))


DATE_CHUNKS = len(ndvi_stack.date)
PIXEL_CHUNKS = 4000


ndvi_stack = ndvi_stack.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
median_ndvi = median_ndvi.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
obs_date_stack = obs_date_stack.chunk({"date": DATE_CHUNKS})

out_ds = xr.Dataset({
    "ndvi": ndvi_stack,
    "median_ndvi": median_ndvi["median_ndvi"],
    "obs_date": obs_date_stack
},
coords={
    "pixel": ndvi_stack.pixel,
    "date": date_stack
})

for v in out_ds.data_vars:
    out_ds[v].encoding.pop("compressor", None)
    out_ds[v].encoding.setdefault("chunks", None)
for c in out_ds.coords:
    out_ds[c].encoding.pop("compressor", None)
    out_ds[c].encoding.setdefault("chunks", None)

OUT_ZARR = "data_for_demo/merged_ndvi.zarr"
out_ds.to_zarr(OUT_ZARR, mode="w", consolidated=True)

print("✅ Done")

