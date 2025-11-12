################################################################################
################################################################################
################################################################################

################################################################################
# Setup:
from dask.distributed import Client, LocalCluster
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import timeit

INPUT_DIR1 = "/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"
INPUT_DIR2 = "/data_3/scratch/francesco/zarr_demo_pixel_chunked_10000.zarr"
INPUT_DIR3 = "/data_3/scratch/francesco/zarr_demo_pixel_chunked_small.zarr"

# SETUP PARALLELIZATION CLUSTER
# create a scheduler client10 (i.e. cluster of workers) 
# client10.close()
client10 = Client(
    n_workers=10,
    threads_per_worker=1,
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':2234'
)  # start distributed scheduler locally.

client20 = Client(
    n_workers=20,
    threads_per_worker=1,
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':2334'
)  # start distributed scheduler locally.

client01 = Client(
    n_workers=1,
    threads_per_worker=1,
    processes=True,  # Use separate processes (not threads, this appears to be much faster (even though using non-shared memory))
    dashboard_address=':1134'
)  # start distributed scheduler locally.


client10.gather
client10.ncores
client10.dashboard_link # provides the link for the dask dashboard: most likely: http://127.0.0.1:8787/status
# on UBELIX in RSTUDIO: switch console to R and in R do: getOption("viewer")('http://127.0.0.1:8787/status') # to open dask dashboard in Viewer: https://support.posit.co/hc/en-us/articles/202133558-Extending-the-RStudio-IDE-with-the-Viewer-Pane
# on WORKSTATION in SSH-connected VSCode, next to the terminal tab, ensure the PORTS are forwarded to you local computer and click on the globe symbol next to the Forwarded Address



################################################################################
################################################################################
################################################################################

################################################################################
# Open data

# Open main Zarr dataset as an xarray Dataset or DataArray with chunks (Dask arrays inside)
ds_main_chunk_1_5000     = xr.open_zarr(INPUT_DIR1, chunks='auto', mask_and_scale=True)
ds_main_chunk_3073_10000 = xr.open_zarr(INPUT_DIR2, chunks='auto', mask_and_scale=True)
ds_main_chunk_3073_1     = xr.open_zarr(INPUT_DIR3, chunks='auto', mask_and_scale=True)



################################################################################
################################################################################
################################################################################

################################################################################
# Test which chunk structure is faster:

# timeit.timeit(
#     lambda: ds_main_chunk_1_5000["ndvi"].isel(pixel = slice(0,3)).groupby('pixel').mean().compute(), 
#     number = 1)
# timeit.timeit(
#     lambda: ds_main_chunk_3073_10000["ndvi"].isel(pixel = slice(0,3)).groupby('pixel').mean().compute(), 
#     number = 1)
timeit.timeit(
    lambda: ds_main_chunk_3073_1["ndvi"].isel(pixel = slice(0,2)).groupby('pixel').mean().compute(), 
    number = 1)


# RESULT: as expected the one chunking pixels separately is optimal


# Test how many cores result in what performance
with client10.as_current(): # Starts using client10
    result10 = timeit.timeit(
        lambda: ds_main_chunk_3073_1["ndvi"].groupby('pixel').mean().compute(),
        number = 1
    )

with client20.as_current(): # Starts using client20
    result20 = timeit.timeit(
        lambda: ds_main_chunk_3073_1["ndvi"].groupby('pixel').mean().compute(),
        number = 1
    )

result10 # 9.66 seconds for 1000 pixels (all time steps)
result20 # 9.44 seconds for 1000 pixels (all time steps)


with client20.as_current(): # Starts using client20
    result21 = timeit.timeit(
        lambda: ds_main_chunk_3073_1["ndvi"][0:10].groupby('pixel').mean().compute(),
        number = 1
    )

result21 # this is only for 10 time steps 
         # (for the mean this does not change the timing)

client20.close()    # only keep client10
client01.close()    # only keep client10
client10.dashboard_link

################################################################################
################################################################################
################################################################################



# DEVELOP THE xr.apply_ufunc() approach

################################################################################
# Approach A) Pixel-by-pixel: (treat single pixel in a single function call, 
#                              but all 3073 values of the whole time series)
# Alternative: Manually handle 2D arrays in funcX_2d

ds_main_chunk_3073_1["dates"][50:60].plot()
ds_main_chunk_3073_1["ndvi"].isel(pixel = 100)[50:60].plot()
ds_main_chunk_1_5000["median_ndvi"].isel(pixel = 100)[50:60].plot()

ds_main_chunk_1_5000.isel(pixel = 100)
ds_main_chunk_1_5000.isel(pixel = 100).load()

# Define function that uses grouped data and interval bounds
def funcX(ndvi_data, median):
    # example logic: Use ndvi_data time series with bounds median to produce time series output
    # ndvi_data: time series (3073 days)
    # median:    time series (3073 days)
    
    # Here just a dummy example:
    diff = abs(ndvi_data - median)
    result = np.cumsum(diff, axis=-1) # note that we need axis arguments, else it flattens array

    return result

# Test the function:

# Test on its own:
# in-memory
ndvi_arg   = ds_main_chunk_3073_1["ndvi"       ].isel(time = slice(50,60), pixel = 100).load() 
median_arg = ds_main_chunk_1_5000["median_ndvi"].isel(time = slice(50,60), pixel = 100).load() 

# lazy
ndvi_arg2   = ds_main_chunk_3073_1["ndvi"       ].isel(time = slice(50,60), pixel = 100)
median_arg2 = ds_main_chunk_1_5000["median_ndvi"].isel(time = slice(50,60), pixel = 100)

funcX(ndvi_arg.to_numpy()     , median_arg.to_numpy())        # in-memory
funcX(ndvi_arg.to_numpy()[0:2], median_arg.to_numpy()[0:2])   # in-memory
funcX(ndvi_arg2.to_numpy()     , median_arg2.to_numpy())      # lazy
funcX(ndvi_arg2.to_numpy()[0:2], median_arg2.to_numpy()[0:2]) # lazy


# Test with xr.apply_ufunc():
# Test single pixel
xr.apply_ufunc(              # in-memory
    funcX,
    ndvi_arg,
    median_arg
)
result_one_pixel = xr.apply_ufunc(     # lazy
    funcX,
    ndvi_arg2,
    median_arg2,
    dask = "parallelized", # otherwise: ValueError: apply_ufunc encountered a chunked array on an argument, but handling for chunked arrays has not been enabled. Either set the ``dask`` argument or load your data into memory first with ``.load()`` or ``.compute()``
    output_dtypes=[np.float32]
)
result_one_pixel                       # lazy
result_one_pixel.compute()             # now compute

# Test multiple pixels grouped by 'pixel'
result_multiple_pixels = xr.apply_ufunc(
    funcX,

    # subset some time steps:
    # ds_main_chunk_3073_1["ndvi"       ].isel(time = slice(50,60), pixel = slice(0, 1000)).chunk(dict(time=-1)).groupby('pixel'), # TODO: ideally we would not need to rechunk here...
    # ds_main_chunk_1_5000["median_ndvi"].isel(time = slice(50,60), pixel = slice(0, 1000)).chunk(dict(time=-1)).groupby('pixel'), # TODO: ideally we would not need to rechunk here...

    # without subsetting time steps
    # ds_main_chunk_3073_1["ndvi"       ].isel(pixel = slice(0, 1000)).chunk(dict(time=-1)).groupby('pixel'), # TODO: ideally we would not need to rechunk here...
    # ds_main_chunk_1_5000["median_ndvi"].isel(pixel = slice(0, 1000)).chunk(dict(time=-1)).groupby('pixel'), # TODO: ideally we would not need to rechunk here...
    ds_main_chunk_3073_1["ndvi"       ].isel(pixel = slice(0, 100)).chunk(dict(time=-1)).groupby('pixel'), # TODO: ideally we would not need to rechunk here...
    ds_main_chunk_1_5000["median_ndvi"].isel(pixel = slice(0, 100)).chunk(dict(time=-1)).groupby('pixel'), # TODO: ideally we would not need to rechunk here...

    # other needed options
    input_core_dims=[['time'], ['time']],  # Tell it which dims to iterate over
    output_core_dims=[['time']],           # Output has time dimension
    vectorize=True,                        # Apply function to each pixel separately
    dask = "parallelized", # otherwise: ValueError: apply_ufunc encountered a chunked array on an argument, but handling for chunked arrays has not been enabled. Either set the ``dask`` argument or load your data into memory first with ``.load()`` or ``.compute()``
    output_dtypes=[np.float32]
).to_dataset(name="cumsum_ndvi_diff")

timeit.timeit(
    lambda: result_multiple_pixels.compute(),
    number = 1
) # 1000 pixels take 9.9 seconds (for 10 time steps and 1000 pixels) and 16 seconds (for 3073 time steps and 1000 pixels)

result_multiple_pixels_memory = result_multiple_pixels.compute()
result_multiple_pixels["cumsum_ndvi_diff"].to_numpy()

# Store output
# result_multiple_pixels.to_zarr("output.zarr", zarr_format=2, mode="w", consolidated=False)
result_multiple_pixels.to_zarr("output.zarr", zarr_format=2, mode="w", consolidated=True)

# stored_results = xr.open_zarr("output.zarr", chunks='auto')
stored_results = xr.open_zarr("output.zarr", chunks='auto', consolidated=False)

# plot the stored results to check
stored_results["cumsum_ndvi_diff"].plot(x='time', hue='pixel', add_legend=True)


################################################################################
# Approach B) Vectorization: (treat multiple pixels in a single function call)
# Alternative: Manually handle 2D arrays in funcX_2d
# def funcX_2d(ndvi_data, median):
#     """Version that handles 2D input (time, pixel)"""
#     # Apply along axis 0 (time dimension)
#     diff = np.abs(ndvi_data - median)
#     result = np.cumsum(diff, axis=0)  # Cumsum along time dimension
#     return result



################################################################################
################################################################################
################################################################################
