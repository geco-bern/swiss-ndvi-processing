import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dask.distributed import Client
from dask import visualize
import dask.array as da
import xarray as xr
import argparse
import os, shutil, sys
import time
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

NO_COVERAGE = 32767
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step
INVALID     = -32768
INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
    
if __name__ == "__main__":

    COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)

    def write_zarr(ds, outfile):
        # Explicit encoding: simple compressor for each data var
        encoding = {v: {"compressors": COMPRESSOR} for v in ds.data_vars}

        # drop any coord/data var chunk encodings that conflict   # TODO: is this needed?
        for name in list(ds.coords) + list(ds.data_vars): # TODO: remove this again if possilbe
            ds[name].encoding.pop("chunks", None)                           # TODO: remove this again if possilbe
            ds[name].encoding.pop("compressor", None)                       # TODO: remove this again if possilbe
            ds[name].encoding.pop("compressors", None)                      # TODO: remove this again if possilbe

        # overwrite (mode="w")
        ds.to_zarr(
            outfile, 
            mode="w", 
            compute=True,
            encoding=encoding, 
            zarr_format=3
        )

    N_WORKERS = 30
    DATE_CHUNKS = -1
    DATE_CHUNKS_OUT = 365
    PIXEL_CHUNKS    = 40000
    MEMORY_PER_WORKER = '120GB'
    N_THREADS_PER_WORKER = 1

    # N_WORKERS = 60        # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    
    t0=time.perf_counter()
    DASK_TEMP_DIR = "/mnt/data2/UniBe-swiss-ndvi/tmp_data6/"
    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=N_THREADS_PER_WORKER,
        memory_limit=MEMORY_PER_WORKER,
        local_directory= DASK_TEMP_DIR,
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':8344')
    print(client, flush = True)
    print(client.dashboard_link, flush = True) # use this dashboard to follow progress


    # run interactively
    # INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-04-29_07h16_ndvi_01_downloaded_2026-01-03_2026-01-03_processed.zarr"
    # HISTO_ZARR_INPUT = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c.zarr"
    
    historic_ds  = xr.open_zarr(HISTO_ZARR_INPUT, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    ###  INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr" # TODO: move to data2
    ###  new_ds       = xr.open_zarr(INPUT_ZARR, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": -1})
    ###  lookuptable  = xr.open_zarr(INPUT_LOOKUPTABLE).chunk({"pixel": PIXEL_CHUNKS})

    # write subset:
    write_zarr(
        historic_ds.sel(pixel = slice(10401, 10500)),
        "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-pixels10401-to-10500.zarr_bkp")

    subset_pixels = xr.DataArray([2694495,2692025,2761095,2781535,2644035,2644325,2690025,2689565], dims="pixel")
    write_zarr(
        historic_ds.sel(pixel = subset_pixels),
        "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-focus-sites.zarr_bkp")
