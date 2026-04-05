"""
Merge downloaded Swisstopo Sentinel-2 datasets for Switzerland.
"""
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import pystac_client
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import numpy as np
import zarr
from tqdm import tqdm
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import argparse
import os, shutil
import sys
from datetime import datetime
from affine import Affine
import pandas as pd
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec
import time

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

# CONFIGURE:
files_to_merge = [
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2017-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2018-01-01_2018-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2019-01-01_2019-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2020-01-01_2020-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2021-01-01_2021-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2022-01-01_2022-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2023-01-01_2023-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2024-01-01_2024-12-31.zarr",
    "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2025-01-01_2025-12-31.zarr"
]
# derive from above files_to_merge
start_date = "2017-01-01"
end_date = "2025-12-31"
today = "2026-04-04_18h16" # datetime.today().strftime("%Y-%m-%d_%Hh%M")


OUTPUT_ZARR      = f"/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_{today}_ndvi_01_downloaded_{start_date}_{end_date}.zarr"
# OUTPUT_ZARR      = f"/data_3/scratch/fabian/UniBe-swiss-ndvi/data/tmp_{today}_ndvi_01_downloaded_{start_date}_{end_date}.zarr"
# ==============================================================================


# check structure (it appears we do have some NaT present in the download)
def preprocess(ds):
    # remove times that are NaT
    valid_times = ds['datetime'].notnull()
    print(f"Removed {sum(~valid_times.values)} NaT times "+
          f"(at locations {[i for i, x in enumerate(valid_times.values) if ~x]}) "+
          f"     in file: {os.path.basename(ds.encoding["source"])}")
    return ds.isel(datetime=valid_times)

# for path in files_to_merge:
#     #ds = xr.open_dataset(path, engine="zarr")
#     ds = preprocess(xr.open_dataset(path, engine="zarr"))
#     t = pd.to_datetime(ds["datetime"].values)
#     print(
#         path,
#         "| sorted:", t.is_monotonic_increasing or t.is_monotonic_decreasing,
#         "| duplicates:", pd.Index(t).has_duplicates,
#         "| first:", t[0] if len(t) else None,
#         "| last:", t[-1] if len(t) else None,
#     )

N_WORKERS = 50
MEMORY_PER_WORKER = "60GB"
cluster = LocalCluster(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    processes=True,
    memory_limit=MEMORY_PER_WORKER,
    dashboard_address=":1235"
)
client = Client(cluster)
print(client, flush = True)
print(client.dashboard_link, flush = True) # use this dashboard to follow progress

ds = xr.open_mfdataset(files_to_merge, preprocess=preprocess, parallel=True)

PIXEL_CHUNKS = 40000
out_ds = ds.chunk({"pixel": PIXEL_CHUNKS, "datetime": -1})


# ==========================================================================
# Write out compressed zarr
compressors = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)

# Explicit encoding: simple compressor for each data var
encoding_compr = {v: {"compressors": compressors} for v in out_ds.data_vars}
t0=time.perf_counter()
out_ds.to_zarr(
        OUTPUT_ZARR,
        mode="w",
        # consolidated=True,
        compute=True,
        encoding=encoding_compr,
        zarr_format=3)
print(f"Elapsed time for full dataset: {time.perf_counter()-t0:.3f}s")

# FOR DEVELOPMENT
# test load this dataset:
# ds_test = xr.open_dataset(OUTPUT_ZARR)
# # ds_test = xr.open_dataset("/data_3/scratch/fabian/UniBe-swiss-ndvi/data/tmp_2026-03-23_22h26_ndvi_01_downloaded_2026-01-01_2026-01-15.zarr")
# # ds_test = xr.open_dataset("/data_3/scratch/fabian/UniBe-swiss-ndvi/data/tmp_2026-03-23_22h26_ndvi_01_downloaded_2026-01-01_2026-01-15.zarr")
# # ds_test = xr.open_dataset("/data_3/scratch/fabian/UniBe-swiss-ndvi/data/tmp_2026-03-23_12h50_ndvi_01_downloaded_2025-11-30_2026-03-22.zarr")
# # ds_test = xr.open_dataset("/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr")
# # test plot this dataset    
# xmin, xmax = 2650000, 2750000 # focus on Ticino
# ymin, ymax = 1070000, 1160000 # focus on Ticino
# pixels_subset_mask = (
#     (ds_test.x.values >= xmin) &
#     (ds_test.x.values <= xmax) &
#     (ds_test.y.values >= ymin) &
#     (ds_test.y.values <= ymax)
# )
# ds_test_subset = ds_test["ndvi"].isel(pixel=pixels_subset_mask.nonzero()[0])
# plot_da_map(ds_test_subset.isel(datetime = 0), png_fname = 'foo5.png')
# plot_da_map(ds_test_subset.isel(datetime = 18), png_fname = 'foo5ter.png')
# pixels_subset_mask2 = (
#     (ds_test.ndvi.values != 32767) &
#     (ds_test.ndvi.values != -32768) &
#     (ds_test.y.values >= ymin) &
#     (ds_test.y.values <= ymax)
# )
# plot_da_map(ds_test_subset.isel(datetime = 0), reduction_factor = 1, png_fname = 'foo1.png')
