from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr

def historical_ndvi(ndvi_arr, medians,dates):

        days_diff = (dates- dates[0])  / np.timedelta64(1, 'D')
     
        ndvi_arr = ndvi_arr / 10000
        medians  = medians  / 10000
        mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

        ndvi_valid = ndvi_arr[mask_valid_ndvi]
        median_valid = medians[mask_valid_ndvi]
        days_diff_2 = days_diff[mask_valid_ndvi]
        
        # outlier detection

        delta_ndvi = ndvi_valid - median_valid
        delta_delta_left = delta_ndvi[2:]
        delta_delta_rigth = delta_ndvi[:-2]
        outlier_mask = ((abs(delta_ndvi[1:-1]) > 0.05) & (abs(delta_delta_left) > 0.05) & (abs(delta_delta_rigth) > 0.05))
        ndvi_valid = ndvi_valid[1:-1][~outlier_mask]
        delta_ndvi = delta_ndvi[1:-1][~outlier_mask]
        days_diff_2 = days_diff_2[1:-1][~outlier_mask]

        # some sites do not have any observation or very few
        if len(delta_ndvi) > 6:
        
            # L2 smoothing

            idx = np.arange(len(delta_ndvi))
            loess =  sm.nonparametric.lowess(delta_ndvi, idx, frac= 7 / len(delta_ndvi), it=3, return_sorted=False)

            # combine smoothed value with values yet to smooth, after that linearly interpolate everything

            ndvi_to_interpolate = np.concatenate([np.array([0]),loess[:-4],delta_ndvi[-4:],np.array([0])]) 
            dates_to_interpolate = np.concatenate([np.array([0]),days_diff_2,np.array([3072])]) # hardcoded, should be days_diff[-1]

            interpolated_values = np.interp(days_diff,dates_to_interpolate,ndvi_to_interpolate)

            final_ndvi_value = interpolated_values + medians


            return final_ndvi_value
        
        else:

            return ndvi_arr


N_WORKERS = 50

client = Client(
n_workers=N_WORKERS,
threads_per_worker=1,
#memory_limit='24GB',
processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
dashboard_address=':12345')  

# already having medians computed

INPUT_ZARR = "/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"
ds = xr.open_zarr(INPUT_ZARR, chunks={"time": -1, "pixel": 5000})
ndvi_array = ds["ndvi"].isel(pixel=slice(0, 999999))            # dims ("time","pixel")
median_array = ds["median_ndvi"].isel(pixel=slice(0, 999999))    # dims ("time","pixel") 
dates_int = ds["dates"].values.astype(np.int32)
dates_array = np.array([datetime.strptime(str(d), "%Y%m%d").date() for d in dates_int], dtype="datetime64[D]")

# call gufunc where core dim is "time" (1D arrays per pixel)
result = xr.apply_ufunc(
    historical_ndvi,
    ndvi_array,
    median_array,
    input_core_dims=[["time"], ["time"]],    # each call gets 1D time arrays
    output_core_dims=[["time"]],
    vectorize=True, 
    dask="parallelized",
    kwargs={"dates": dates_array},
    output_dtypes=[ndvi_array.dtype],
    dask_gufunc_kwargs={"allow_rechunk": True},
)

client.dashboard_link

# create the dataset to write 

out_ds = xr.Dataset({"ndvi_processed": result}, coords={"time": ds["dates"], "pixel": ds["pixel"]})
out_ds = out_ds.chunk({"pixel": 5000, "time": -1})

# Remove any incompatible 'compressor' metadata left over from the source dataset
for v in list(out_ds.data_vars):
    out_ds[v].encoding.pop("compressor", None)
    # ensure chunks entry exists to avoid surprises
    out_ds[v].encoding.setdefault("chunks", None)

for c in list(out_ds.coords):
    out_ds[c].encoding.pop("compressor", None)
    out_ds[c].encoding.setdefault("chunks", None)

# Explicit encoding: no compressor for each data var
encoding = {v: {"compressor": None} for v in out_ds.data_vars}

# Write using zarr version 2 to avoid new v3 codec/BytesBytesCodec mismatch
out_ds.to_zarr("ndvi_processed.zarr", mode="w", consolidated=True, compute=True, encoding=encoding, zarr_version=2)

client.close