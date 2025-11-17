from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import math

def continous_ndvi(pixel,current_date,last_dates_array,first_date):

    idx_current_date = (current_date - first_date)  / np.timedelta64(1, 'D')
    idx_last_date =  (last_dates_array[6] - first_date)  / np.timedelta64(1, 'D')

    current_ndvi_value = ds["ndvi"].isel(pixel=pixel, time =idx_current_date ).load()
    last_ndvi_value =  ds["ndvi"].isel(pixel=pixel, time =idx_last_date).load()

    current_ndvi_value = current_ndvi_value / 10000

    median_current_value =  ds["median_ndvi"].isel(pixel=pixel, time =idx_current_date).load()
    median_last_value =  ds["median_ndvi"].isel(pixel=pixel, time =idx_last_date).load()
    last_delta = last_ndvi_value - median_current_value

    # check if the current ndvi value is an obs. or not

    if (current_ndvi_value > 0) & (current_ndvi_value < 1):
             
        # check if it is a potential outlier or not
        current_delta = current_ndvi_value - median_current_value
        delta_delta = current_delta - last_delta

        if ((abs(current_delta) > 0.05) & (abs(delta_delta) > 0.05)):
                 
            # the value is a potential outlier that will be checked as soon a new observation will be avaible
            last_dates_array[7] = current_date

            return last_dates_array
            
        else:
                 
            # check if a potential outlier is pending or not. 
            # If there is no potential outlier the last dates will have a placehodler value of 1900-01-01

            if last_dates_array[7] != date(1900,1,1):
                    
                potential_idx = (last_dates_array[7] - first_date)  / np.timedelta64(1, 'D')
                potential_ndvi_value = ds["ndvi"].isel(pixel=pixel, time = potential_idx).load()
                potential_median_value = ds["median_ndvi"].isel(pixel=pixel, time = potential_idx).load()

                potential_delta = potential_ndvi_value - potential_median_value
                potential_delta_delta = potential_delta - current_delta

                if abs(potential_delta_delta) < 0.05:

                    # the potential outlier is NOT a true outlier but it is an observation
                    # since it is an obseravtion was not process before, the L1 and L2 must be done twice
                    days_diff_array = ((np.arange(last_dates_array[6],current_date + 1) - first_date)  / np.timedelta64(1, 'D'))
                    median_array =  ds["median_ndvi"].isel(pixel=pixel)[idx_last_date:].load()

                    idx_to_interpolate = np.concatenate(last_dates_array[6],last_dates_array[7],current_date)
                    delta_to_interpolate = np.concatenate(last_delta,potential_delta,current_delta)

                    interpolation = np.interp(days_diff_array,idx_to_interpolate,delta_to_interpolate)
                    L1_interpolated = median_array + interpolation
                    # write it back to ndvi array

                    # load the past 6 values to perform the smoothing (since we already loaded the last value)
                    idx_last_date_arr = (last_dates_array[:6] - first_date)  / np.timedelta64(1, 'D')
                    last_ndvi_arr = ds["ndvi"].isel(pixel=pixel, time = idx_last_date_arr).load()
                    last_median_arr =  ds["median_ndvi"].isel(pixel=pixel, time = idx_last_date_arr).load()

                    last_delta_arr = last_ndvi_arr - last_median_arr

                    # concatenate the apst value with last delta and current delta
                        
                    delta_to_smooth = np.concatenate(last_delta_arr, last_delta,potential_delta,current_delta)
                    idx = np.arange(1,9) # having the potential outlier as obs gives one more value
                    loess =  sm.nonparametric.lowess(delta_to_smooth, idx, frac= 1, it=3, return_sorted=False)

                    # the smoothed value to write are the third and FIFTH values instead of fourth (since we have an extra value)
                    idx_to_smooth = ((np.arange(last_dates_array[2],last_dates_array[4] + 1) - first_date)  / np.timedelta64(1, 'D')) 

                    median_array_to_smooth =  ds["median_ndvi"].isel(pixel=pixel, time = idx_to_smooth)
                    idx_to_interp = ((np.concatenate(last_dates_array[2],last_dates_array[3],last_dates_array[4] + 1)) - first_date)  / np.timedelta64(1, 'D')
                        
                    interpolated_smoothed_values = np.interp(idx_to_smooth,idx_to_interp,loess[2:5])
                    ndvi_smoothed_value = interpolated_smoothed_values + median_array_to_smooth

                    # write it back to ndvi array

                    # update last date array with the new dates as shown above, INCLUDING the potential date (at 8th position)
                    last_dates_array = np.concatenate(last_dates_array[2:],current_date,date(1900,1,1))

                    return last_dates_array 


                else:

                    # the potential outlier is a true outlier but it is an observation
                    # this block of code and the following one are identical, maybe wrap in a function in the future

                    # perform the L1 linear interpolation between current date and last date
                    days_diff_array = ((np.arange(last_dates_array[6],current_date + 1) - first_date)  / np.timedelta64(1, 'D')) 

                    median_array =  ds["median_ndvi"].isel(pixel=pixel)[idx_last_date:].load()
                    interpolation = np.linspace(last_delta, current_delta,num = len(median_array))
                    L1_interpolated = median_array + interpolation
                    # write it back to ndvi array

                    # load the past 6 values to perform the smoothing (since we already loaded the last value)
                    idx_last_date_arr = (last_dates_array[:6] - first_date)  / np.timedelta64(1, 'D')
                    last_ndvi_arr = ds["ndvi"].isel(pixel=pixel, time = idx_last_date_arr).load()
                    last_median_arr =  ds["median_ndvi"].isel(pixel=pixel, time = idx_last_date_arr).load()

                    last_delta_arr = last_ndvi_arr - last_median_arr

                    # concatenate the apst value with last delta and current delta
                        
                    delta_to_smooth = np.concatenate(last_delta_arr, last_delta,current_delta)
                    idx = np.arange(1,8) # having the potential outlier as obs gives one more value
                    loess =  sm.nonparametric.lowess(delta_to_smooth, idx, frac= 1, it=3, return_sorted=False)

                    # the smoothed value to write are the third and fourth values
                    idx_to_smooth = ((np.arange(last_dates_array[2],last_dates_array[3] + 1) - first_date)  / np.timedelta64(1, 'D')) 

                    median_array_to_smooth =  ds["median_ndvi"].isel(pixel=pixel, time = idx_to_smooth).load()

                        
                    interpolated_smoothed_values = np.linspace(loess[2], loess[3], num = len(idx_to_smooth))
                    ndvi_smoothed_value = interpolated_smoothed_values + median_array_to_smooth

                    # write it back to ndvi array

                    # update the last date array
                    # after an observation is met, it is not possible to have a potential outlier so it must se to the placehodler value

                    last_dates_array = np.concatenate(last_dates_array[1:7],current_date,date(1900,1,1))

                    return last_dates_array 

            else:

                # perform the L1 linear interpolation between current date and last date
                days_diff_array = ((np.arange(last_dates_array[6],current_date + 1) - first_date)  / np.timedelta64(1, 'D')) 

                median_array =  ds["median_ndvi"].isel(pixel=pixel)[idx_last_date:].load()
                interpolation = np.linspace(last_delta, current_delta,num = len(median_array))
                L1_interpolated = median_array + interpolation
                # write it back to ndvi array

                # load the past 6 values to perform the smoothing (since we already loaded the last value)
                idx_last_date_arr = (last_dates_array[:6] - first_date)  / np.timedelta64(1, 'D')
                last_ndvi_arr = ds["ndvi"].isel(pixel=pixel, time = idx_last_date_arr).load()
                last_median_arr =  ds["median_ndvi"].isel(pixel=pixel, time = idx_last_date_arr).load()

                last_delta_arr = last_ndvi_arr - last_median_arr

                # concatenate the apst value with last delta and current delta
                    
                delta_to_smooth = np.concatenate(last_delta_arr, last_delta,current_delta)
                idx = np.arange(1,8)
                loess =  sm.nonparametric.lowess(delta_to_smooth, idx, frac= 1, it=3, return_sorted=False)

                # the smoothed value to write are the third and fourth values
                idx_to_smooth = ((np.arange(last_dates_array[2],last_dates_array[3] + 1) - first_date)  / np.timedelta64(1, 'D')) 

                median_array_to_smooth =  ds["median_ndvi"].isel(pixel=pixel, time = idx_to_smooth).load()
                interpolated_smoothed_values = np.linspace(loess[2], loess[3], num = len(idx_to_smooth))
                ndvi_smoothed_value = interpolated_smoothed_values + median_array_to_smooth

                # write it back to ndvi array
                    
                # update last date array with the new dates as shown above
                last_dates_array = np.concatenate(last_dates_array[1:7],current_date,date(1900,1,1))

                return last_dates_array 

    else:
             
        # perform the L0 estimation based on the last delta avaible
        days_diff = (last_dates_array[6] - current_date)  / np.timedelta64(1, 'D')

        decrease_factor = math.exp(-math.log(2) * (days_diff / 15.0))
        ndvi_estimated = median_last_value + last_delta * decrease_factor
        # write it back to ndvi array

        # here no need to update the last dates array

        return last_dates_array







# this part after is not changed from historic ndvi so it won't work




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