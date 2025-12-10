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

SRC_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
OUT_ZARR  = "/data_3/scratch/francesco/zarr_ready_all_pixels.zarr"


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

# Lazy dask arrays from zarr
ndvi_z = ds0["ndvi"]
pl_z   = ds0["params"]["params_lower"]
pu_z   = ds0["params"]["params_upper"]

ndvi_da = da.from_zarr(ndvi_z)     # lazy
pl_da   = da.from_zarr(pl_z)       # lazy
pu_da   = da.from_zarr(pu_z)       # lazy

# Decode dates into a small in-memory coordinate
dates = pd.to_datetime([d.decode("utf-8") for d in ds0["dates"][:]])

# Build xarray objects with explicit dims/coords (still lazy)
param_labels = ['par0', 'par1', 'par2', 'par3', 'par4', 'par5'] # TODO: what are param names from Samanthas model?
ndvi_xr         = xr.DataArray(ndvi_da, dims=("pixel", "date"),  coords={"date": dates, "pixel": np.arange(pl_da.shape[0])})
params_lower_xr = xr.DataArray(pl_da,   dims=("pixel", "param"), coords={               "pixel": np.arange(pl_da.shape[0]), "param": param_labels})
params_upper_xr = xr.DataArray(pu_da,   dims=("pixel", "param"), coords={               "pixel": np.arange(pl_da.shape[0]), "param": param_labels})

# dims/coords are used as following:
# params_upper_xr.sel(param = 'par1') # sel uses label
# params_upper_xr.isel(param = 0)     # isel uses integer

print(f"Loaded NDVI lazily with shape {tuple(ndvi_xr.shape)}, {len(dates)} unique dates.")

# =====================================================
#  Generate Daily Date Range (in-memory index only)
# =====================================================
daily_dates = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
print(f"Generated {len(daily_dates)} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

obs_dates = daily_dates.isin(dates)


# =====================================================
#  Reindex NDVI to daily (lazy), fill with 32767 (int16)
# =====================================================
# NOTE: issue that there are duplicate dates:
dates.values
np.unique(dates)
# TODO: understand why we have this issue of repeated indices
#       in the output of Samanthas code
#       and then check if below workaround treats this correctly (this is currently just an unverified AI solution)

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
ndvi_sel_nodup = _dedup_date_coord(ndvi_sel, how="first")
ndvi_daily = ndvi_sel_nodup.astype(np.int16).reindex(date=daily_dates, method=None, fill_value=np.int16(32767))



# Build auxiliary lazy arrays
counter_da = xr.DataArray(
    da.zeros(ndvi_daily.sizes["date"], dtype=np.int16),
    dims=("date",),
    coords={"date": ndvi_daily["date"]},
)


# =====================================================
#  Compute median_ndvi (lazy, same shape as ndvi_daily)
# =====================================================

# get numpy array of scaled time (same as before)
doy = np.array([d.timetuple().tm_yday for d in daily_dates], dtype=np.float32)
doy[doy == 366] = 365
t_scaled = doy / 365.0  # shape (date,)



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
        "median_ndvi": median_ndvi_xr,
        "counter": counter_da,
        "params_lower": pl_sel,
        "params_upper": pu_sel,
        "obs" : obs_dates_xr
    }
)

# Optional encodings (Zarr chunks and dtypes)

# Optional IO-friendly chunking: write one day per task
# ndvi_daily = ndvi_daily.chunk({"date": 1})
# Optional IO-friendly chunking
# ndvi_daily = ndvi_daily.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
# Optional IO-friendly chunking
DATE_CHUNKS = 1600
PIXEL_CHUNKS = 4000
out_ds = out_ds.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

# Ensure correct encoding (is written to the metadata???)
# out_ds["ndvi"].encoding.update({"dtype": "int16"})
# out_ds["counter"].encoding.update({"dtype": "int16"})
# out_ds["params_lower"].encoding.update({"dtype": "float32"})
# out_ds["params_upper"].encoding.update({"dtype": "float32"})
# out_ds["last_dates"].encoding.update({"dtype": "S10"})


# Write to Zarr
os.makedirs(OUT_ZARR, exist_ok=True)
print(f"Writing lazily computed Dataset to {OUT_ZARR} with Dask...")
out_ds.to_zarr(OUT_ZARR, mode="w", consolidated=True)
print("✅ Done")


"""
# How to use?
# a) with integers:
out_ds['ndvi'][13,13]
out_ds['ndvi'][{'pixel':13, 'date':13}]
out_ds['ndvi'][{'date':13, 'pixel':13}]
out_ds['ndvi'].isel(pixel=13, date=13)
# a) with coordinate values (i.e. labels):
out_ds['ndvi'].sel(pixel=85668629, date='2017-04-16')
out_ds['params_lower'].sel(pixel=85668629, param='par0').load()

out_ds['params_lower'].isel(pixel=slice(0,10)).load()

# zarr_written = xr.open_zarr(OUT_ZARR, chunks='auto', mask_and_scale=True)"""