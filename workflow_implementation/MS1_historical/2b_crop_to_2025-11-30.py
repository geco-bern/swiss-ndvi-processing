from datetime import datetime, date, timedelta
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import dask
import xarray as xr
import os, sys
import shutil
import pandas as pd
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec
import time
from math import ceil

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

if __name__ == "__main__":

    # Paths
    OUTPUT_TIFF_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7final"

    os.makedirs(OUTPUT_TIFF_BASE, exist_ok=True)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("INPUT_HISTORIC",  help="Full path to Zarr folder with historic NDVI data")
    # parser.add_argument("date",            help="Start date in YYYY-MM-DD or then 'all_dates'")
    # args = parser.parse_args()

    # args = parser.parse_args()
    # INPUT_HISTORIC = args.INPUT_HISTORIC
    INPUT_HISTORIC = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr"
    OUT_PATH       = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7_2025-11.zarr"

    CROP_TO_DATE = "2025-11-30"
    
    COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)
    N_WORKERS = 20
    with Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit='80GB',
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':1245') as client:

        NDVI_historic = xr.open_zarr(INPUT_HISTORIC)
        out_ds = NDVI_historic.sel(date = slice(None, pd.to_datetime(CROP_TO_DATE)))
        
        encoding = {v: {"compressors": COMPRESSOR} for v in out_ds.data_vars}

        # drop any coord/data var chunk encodings that conflict
        for name in list(out_ds.coords) + list(out_ds.data_vars):
            out_ds[name].encoding.pop("chunks", None)
            out_ds[name].encoding.pop("compressor", None)
            out_ds[name].encoding.pop("compressors", None)
            
        out_ds.to_zarr(OUT_PATH, mode="w", encoding=encoding, zarr_format=3)
    
    sys.exit(0)

# from GECO-Workstation-02:
# rsync -ahz --info=progress2 -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr /data_3/scratch
# rsync -ahz --info=progress2 -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr /data_3/scratch; mv /data_3/scratch/historical_2026-04-04_18h16_historical_v7.zarr /data_3/scratch/historical_2026-04-04_18h16_historical_v7_2026-04-09.zarr