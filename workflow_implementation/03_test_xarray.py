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

from dask.distributed import Client, LocalCluster
from matplotlib import pyplot as plt
import xarray as xr
import timeit

INPUT_DIR = "/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"
N_WORKERS = 5

# SETUP PARALLELIZATION CLUSTER
# create a scheduler client (i.e. cluster of workers) 
client.close()
client = Client(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    # memory_limit='12GB', if commented out, uses all available the memory
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':1234'
)  # start distributed scheduler locally.
# Total cores used: 5 workers × 2 threads = 10 cores
                # FURTHER INFO:
                # the above implictly creates a local cluster:  cluster = LocalCluster()
                # and then uses this to initialize a scheduler: client = Client(cluster)
                # cluster = LocalCluster(
                #     dashboard_address=':1234',
                #     n_workers=N_WORKERS, threads_per_worker=1)
                # client = Client(cluster)
                # 
                # Alternatively the approach can be modified to distribute the 
                # workes in a cluster across different compute nodes, even across
                # SLURM-controlled compute nodes. Thus this approach easily 
                # scales to HPC infrastructures.
    

client.gather
client.ncores
client.dashboard_link # provides the link for the dask dashboard: most likely: http://127.0.0.1:8787/status
# on UBELIX in RSTUDIO: switch console to R and in R do: getOption("viewer")('http://127.0.0.1:8787/status') # to open dask dashboard in Viewer: https://support.posit.co/hc/en-us/articles/202133558-Extending-the-RStudio-IDE-with-the-Viewer-Pane
# on WORKSTATION in SSH-connected VSCode, next to the terminal tab, ensure the PORTS are forwarded to you local computer and click on the globe symbol next to the Forwarded Address

# Open main Zarr dataset as an xarray Dataset or DataArray with chunks (Dask arrays inside)
ds_main = xr.open_zarr(INPUT_DIR, chunks='auto', mask_and_scale=True)
ds_main
ds_main["ndvi"].encoding["_FillValue"]

# ds_main2 = xr.open_zarr("v3_.zarr", chunks='auto') # here, you could load an alterantively chunked data set to get an idea on the performance penality of bad chunking

# computations are always done layzli (unless they are triggered by a compute(), plot(), load(), write_to_disk(), ... )
ds_main["ndvi"].isel(pixel = slice(0,1))[1:5].mean()           # lazy

# thus our workflow would open data lazily, define the task list of TODOs, and then only do them when data is outputted to the final data set
# result = ds_main["ndvi"].isel(pixel = slice(0,1))[1:5].mean() # lazy
# result.write_to_zarr("output.zarr", )                         # not lazy, but actually writing the result (chunk-wise) to disk

# Check input visually:
ds_main["ndvi"].isel(pixel = 1).plot()    # This is slow because we request all time steps (we need to load 3000 chunks of 5000 pixels, but only use 1 pixel)
ds_main["ndvi"].isel(pixel = 100).plot()  # This is slow because we request all time steps (we need to load 3000 chunks of 5000 pixels, but only use 1 pixel)
ds_main["ndvi"].isel(pixel = 100)[90:100].plot()  # This is faster because we request only 10 time steps (we need to load 10 chunks of 5000 pixels, but only use 1 pixel)

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


# Perform some timings:
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
    lambda:  v, 
    number = 1)

timeit.timeit(
    ds_main["ndvi"].isel(pixel = slice(0,1))[1:5].mean().compute(),
    number = 1)

timeit.timeit(
    lambda: ds_main["ndvi"][0:10,:].load(),
    number = 1
)
# This is the real test:
# timeit.timeit(
#     # lambda: ds_main["ndvi"][0:100,:].load(),
#     lambda: ds_main["ndvi"][:,:].load(),
#     number = 1
# )
# 
client


# Questions:
# 1) Input zarr file:
#       Total data: 12GB (ndvi) + 12GB (median ndvi)
#       But file size is only 7.5 GB, so it appears to be compressed?
#       Moreover it is not at all optimized for the access pattern:
#           we need to be able to do
#           ds_main["ndvi"][:,:].load()
#           and dont care about
#           ds_main["ndvi"][:,:].load()

## %%time 
# a=12






# xarray opeartions can look something like dplyr operations in R.
# our goal is to define the application of our function(s) in a similar
# way to dplyr::summarise from R.
#
#
# datafame |>
#     group_by(week_number) |>
#     summarise(weekly_mean = funcX(NDVI, upper))


# # Workflow with xr.apply_ufunc:

# # # Open the interval bounds file (assuming similar dimensions)
# # ds_intervals = xr.open_zarr('path_to_intervals.zarr', chunks='auto')
# # upper_bounds = ds_intervals['upper']  # variable name example
# # lower_bounds = ds_intervals['lower']  # variable name example

# # Define function that uses rolling window data and interval bounds
# def funcX(window_data, upper, lower):
#     # example logic: Use window_data 7-day window with bounds to produce scalar output
#     # window_data shape expected: (window size, ...)
#     # upper and lower could be broadcasted if needed
#     # Here just a dummy example:
#     return ((window_data > lower) & (window_data < upper)).mean()

# # Apply rolling window over time dimension with window size 7 (adjust dimension name as needed)
# rolling_obj = ds_main.rolling(time=7, center=True)

# # Construct rolling windows to pass full windows to apply_ufunc
# rolling_window = rolling_obj.construct("window_dim")

# # Apply funcX using xarray's apply_ufunc with dask parallelization
# result = xr.apply_ufunc(
#     funcX,
#     rolling_window,
#     upper_bounds,
#     lower_bounds,
#     input_core_dims=[["window_dim"], [], []],
#     vectorize=True,
#     dask='parallelized',
#     output_dtypes=[float]
# )

# # # result is an xarray object with the rolling window function applied

