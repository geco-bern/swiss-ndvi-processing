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
#  Forest Pixel Selection
# =====================================================
n_pixels = len(sel_1)
print(f"Subset has {n_pixels} pixels.")

# Lazy subset for selected pixels
ndvi_sel = ndvi_xr.isel(pixel=sel_1)         # equivalently sel() but much slower
pl_sel   = params_lower_xr.isel(pixel=sel_1) # equivalently sel() but much slower
pu_sel   = params_upper_xr.isel(pixel=sel_1) # equivalently sel() but much slower

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

def double_logistic(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = np.split(params, 6, axis=-1)
    mat_minus_sos = np.log1p(np.exp(mat_minus_sos))
    eos_minus_sen = np.log1p(np.exp(eos_minus_sen))
    t = t[None, :]  # shape (1, date)
    sigmoid_sos_mat = 1 / (1 + np.exp(2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10)))
    sigmoid_sen_eos = 1 / (1 + np.exp(2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10)))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

# get numpy array of scaled time (same as before)
doy = np.array([d.timetuple().tm_yday for d in daily_dates], dtype=np.float32)
doy[doy == 366] = 365
t_scaled = doy / 365.0  # shape (date,)

# Compute median lazily pixel-wise
def build_median_ndvi_block(pl_block, pu_block):
    ndvi_lower = double_logistic(t_scaled, pl_block)
    ndvi_upper = double_logistic(t_scaled, pu_block)
    return ((ndvi_lower + ndvi_upper) / 2.0 * 10000).astype(np.int16)

# dask.map_blocks over first axis (pixel)
median_da = da.map_blocks(
    build_median_ndvi_block,
    pl_sel.data,
    pu_sel.data,
    dtype=np.int16,
    chunks=(pl_sel.chunks[0], (len(daily_dates),))
)

median_ndvi_xr = xr.DataArray(
    median_da,
    dims=("pixel", "date"),
    coords={"pixel": pl_sel["pixel"], "date": ndvi_daily["date"]},
    name="median_ndvi",
)

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