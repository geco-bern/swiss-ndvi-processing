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
OUT_ZARR  = "/data_3/scratch/francesco/demo.zarr"


# SETUP PARALLELIZATION CLUSTER
client10 = Client(
    n_workers=10,
    threads_per_worker=1,
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':2231'
)  # start distributed scheduler locally.
client10.dashboard_link

# =====================================================
#  Load Forest Mask for Pixel Selection
# =====================================================

def extract_pixel_index(UL_x, UL_y, BR_x, BR_y):
    height, width = 24542, 37728
    left, bottom = 2474090.0, 1065110.0
    px = 10.0
    top = bottom + height * px
    mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

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
#  Extract subset (pixel indices) and subset lazily
# =====================================================
center_x, center_y = 2694491.82, 1126023.20
sel_1 = extract_pixel_index(
    center_x - 300, center_y - 300, # TODO: if x and y are Swiss coordinates they increas north and eastward.
    center_x + 300, center_y + 300) #       Thus the provided coordinates are lower-left (SW) and upper-right (NE).

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


# =====================================================
#  Forest Pixel Selection
# =====================================================
n_pixels = len(sel_1)
print(f"Subset has {n_pixels} pixels.")
ndvi_xr = xr.DataArray(ndvi_da, dims=("pixel", "date"),  coords={"date": dates, "pixel": np.arange(ndvi_da.shape[0])})
ndvi_sel = ndvi_xr.isel(pixel=sel_1)

# =====================================================
#  Reindex NDVI to daily (lazy), fill with 32767 (int16)
# =====================================================
# NOTE: issue that there are duplicate dates:
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


DATE_CHUNKS = min(len(daily_dates),365)
PIXEL_CHUNKS = min( len(sel_1),4000)
out_ds = out_ds.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})


# Write to Zarr
os.makedirs(OUT_ZARR, exist_ok=True)
print(f"Writing lazily computed Dataset to {OUT_ZARR} with Dask...")
out_ds.to_zarr(OUT_ZARR, mode="w", consolidated=True)
print("✅ Done")

# merge with historical ndvi (TODO)

# slice the historical ndvi data (not to do in future)


historical_ndvi_src = "/data_3/scratch/francesco/ndvi_processed2.zarr"
historical_ndvi = xr.open_zarr(historical_ndvi_src)

# historical filtered
ndvi_historic = historical_ndvi["ndvi_processed"].sel(
    pixel=sel_1,
    date= slice(None, "2018-05-31")
).rename("ndvi")

obs_date_historical = historical_ndvi["obs_date"].sel( date= slice(None, "2018-05-31"))


# new data
ds_to_stack = xr.open_zarr(OUT_ZARR)
ndvi_new = ds_to_stack["ndvi"].sel(pixel=sel_1).rename("ndvi")  # <── match pixel subset
obs_date_new = ds_to_stack["obs"].rename("obs_date")

# stack along time
ndvi_stack = xr.concat([ndvi_historic, ndvi_new], dim="date").sortby("date")

obs_date_stack = xr.concat([obs_date_historical, obs_date_new], dim="date").sortby("date")


date_stack =  xr.concat([historical_ndvi["date"].sel(date= slice(None, "2018-05-31")), ds_to_stack["date"]], dim="date").sortby("date")

date_stack = date_stack.astype("datetime64[D]")

# extract the mean of lower and upper bands

lookuptable_src = "/data_3/francesco/lookup_table_median_ndvi.zarr"

lookuptable = xr.open_zarr(lookuptable_src)

lookuptable_arr = lookuptable["median_ndvi"].sel(pixel = sel_1)

doy = date_stack.dt.dayofyear

# remove leap year if encountered
doy_array_fixed = np.where(doy.values == 366, 365, doy.values)

median_ndvi = lookuptable.sel(doy=xr.DataArray(doy_array_fixed, dims="date"), pixel = sel_1)


DATE_CHUNKS = min(len(ndvi_stack.date), 2000)
PIXEL_CHUNKS = min(len(ndvi_stack.pixel), 4000)


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

OUT_ZARR = "/data_3/scratch/francesco/demo_stacked.zarr"
out_ds.to_zarr(OUT_ZARR, mode="w", consolidated=True)

print("✅ Done")

