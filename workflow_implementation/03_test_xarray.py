# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/03_test_parallel_with_dask.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/output/log/zarr_parallel_continous_ndvi_gpu_ssd_3.log &
# import os
# import sys
# import time
# import math
# import gc
# import zarr
# import shutil
# import hashlib
# import traceback
# import concurrent.futures
# import torch

# from datetime import datetime, date
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import dask.array as da
# from dask import delayed

from dask.distributed import Client
from matplotlib import pyplot as plt
import xarray as xr
import timeit

INPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"
# N_WORKERS = 1
N_WORKERS = 10

# ds = zarr.open_group(INPUT_DIR, mode='r')
# ndvi_z = da.from_zarr(ds["ndvi"])
# last_dates_z = da.from_zarr(ds["last_dates"])
# params_lower_z = da.from_zarr(ds["params"]["params_lower"])
# params_upper_z = da.from_zarr(ds["params"]["params_upper"])

# create client (i.e. cluster of workers)
client = Client(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    memory_limit='24GB',
    processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
    dashboard_address=':1234'
)  # start distributed scheduler locally.
# Total cores used: 10 workers × 2 threads = 20 cores

client.gather
client.ncores
client.dashboard_link # provides the link for the dask dashboard: most likely: http://127.0.0.1:8787/status
# on UBELIX in RSTUDIO: switch console to R and in R do: getOption("viewer")('http://127.0.0.1:8787/status') # to open dask dashboard in Viewer: https://support.posit.co/hc/en-us/articles/202133558-Extending-the-RStudio-IDE-with-the-Viewer-Pane

# Open main Zarr dataset as an xarray Dataset or DataArray with chunks (Dask arrays inside)
ds_main = xr.open_zarr(INPUT_DIR, chunks='auto')

# Check input visually:
ds_main["ndvi"].isel(pixel = 1).plot()
ds_main["ndvi"].isel(pixel = 100).plot()

ds_main["median_ndvi"].isel(pixel = 10000).plot()
ds_main["ndvi"].isel(pixel = 10000).plot()
ds_main["last_dates"].isel(pixel = 10000).plot()
ds_main["dates"].plot()

ds_main["median_ndvi"][:,0:3].plot()

ds_main["median_ndvi"].isel(pixel = 0).plot()
ds_main["median_ndvi"].isel(pixel = 1).plot()
ds_main["median_ndvi"].isel(pixel = slice(0,1)).plot()
(ds_main["median_ndvi"]
    .isel(pixel = slice(0,3))
    .plot(x = 'time', hue ='pixel'))

ds_main["ndvi"].isel(pixel = slice(0,3), time = slice(0,365))
ds_main["ndvi_filtered"] = ds_main["ndvi"].where(ds_main["ndvi"] >= 20000)

(ds_main["ndvi"]
    .isel(pixel = slice(0,3), time = slice(0,365))
    .plot(x = 'time', hue ='pixel'))

(ds_main["ndvi_filtered"]
    .isel(pixel = slice(0,3), time = slice(0,365))
    .plot(x = 'time', hue ='pixel'))


# do some timings:
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = 1)[0:2].plot(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = 1)[0:10].plot(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = 1)[0:100].plot(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = 1)[0:1000].plot(), 
    number = 1)


timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = 1)[0:100].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(0,10))[0:100].load(), 
    number = 1)

timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(10,20))[0:100].load(), 
    number = 1)

timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(10,20))[0:1000].load(), 
    number = 1)

timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,1))[:].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,50))[0:1].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,5000))[0:1].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,5001))[0:1].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,10000))[0:1].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,20000))[2:3].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,100000))[2:3].load(), 
    number = 1)
timeit.timeit(
    lambda: ds_main["ndvi"].isel(pixel = slice(1,1000000))[2:3].load(), 
    number = 1)

# timeit.timeit(
#     # lambda: ds_main["ndvi"][0:100,:].load(),
#     lambda: ds_main["ndvi"][:,:].load(),
#     number = 1
# )
# 


## %%time 
# a=12






# Workflow with xr.apply_ufunc:

# # Open the interval bounds file (assuming similar dimensions)
# ds_intervals = xr.open_zarr('path_to_intervals.zarr', chunks='auto')
# upper_bounds = ds_intervals['upper']  # variable name example
# lower_bounds = ds_intervals['lower']  # variable name example

# Define function that uses rolling window data and interval bounds
def funcX(window_data, upper, lower):
    # example logic: Use window_data 7-day window with bounds to produce scalar output
    # window_data shape expected: (window size, ...)
    # upper and lower could be broadcasted if needed
    # Here just a dummy example:
    return ((window_data > lower) & (window_data < upper)).mean()

# Apply rolling window over time dimension with window size 7 (adjust dimension name as needed)
rolling_obj = ds_main.rolling(time=7, center=True)

# Construct rolling windows to pass full windows to apply_ufunc
rolling_window = rolling_obj.construct("window_dim")

# Apply funcX using xarray's apply_ufunc with dask parallelization
result = xr.apply_ufunc(
    funcX,
    rolling_window,
    upper_bounds,
    lower_bounds,
    input_core_dims=[["window_dim"], [], []],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
)

# result is an xarray object with the rolling window function applied

