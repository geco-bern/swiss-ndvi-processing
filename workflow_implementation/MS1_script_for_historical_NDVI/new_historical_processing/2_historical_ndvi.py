from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import os
import shutil
import pandas as pd

#  nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_folder/2_historical_ndvi_short.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_folder/2_historical_ndvi_short_3.log 2>&1 &

def historical_ndvi(ndvi_arr_original,full_median_array_original,obs_date):

    # create evenly spaced ndvi
    ndvi_full = np.full(obs_date.shape, 2**15 -1, dtype= np.int16)
    ndvi_full[obs_date] = ndvi_arr_original


    # Ensure mask_array is writable
    mask_array = np.zeros(full_median_array_original.shape, dtype=np.int8)

    days_diff = np.arange(0, len(obs_date)) 

    delta_ndvi = np.array([0])

    # renaming is necessary otherwise won't work
     
    ndvi_arr = ndvi_full / 10000
    full_median_array = full_median_array_original / 10000

    mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1) & obs_date

    ndvi_valid = ndvi_arr[mask_valid_ndvi]
    median_valid = full_median_array[mask_valid_ndvi]
    days_diff_2 = days_diff[mask_valid_ndvi]

    original_idx = np.arange(len(ndvi_full)) # used to keep track of delta ndvi position and the outlier position
    original_idx = original_idx[mask_valid_ndvi]
        
    # outlier detection

    delta_threshold = 0.1
    delta_delta_threshold = 0.1

    delta_ndvi = ndvi_valid - median_valid
    delta_delta_left = delta_ndvi[2:]
    delta_delta_rigth = delta_ndvi[:-2]
    outlier_mask = ((abs(delta_ndvi[1:-1]) > delta_threshold) & (abs(delta_delta_left) > delta_delta_threshold) & (abs(delta_delta_rigth) > delta_delta_threshold))
    ndvi_valid = ndvi_valid[1:-1][~outlier_mask]
    delta_ndvi = delta_ndvi[1:-1][~outlier_mask]
    days_diff_2 = days_diff_2[1:-1][~outlier_mask]

    original_idx_2 = original_idx[1:-1][~outlier_mask]
        

    # some sites do not have any observation or very few
    if len(delta_ndvi) > 6:
        
        # L2 smoothing
        # loop over the 7 rolling deltas. If the deltas are too large (extreme events as fire) 
        # or the original values too close to the boundaries condition (0.9 and 0.1) we do linear fit

        delta_ndvi_to_interpolate = np.empty(len(delta_ndvi) -6, dtype=float)

        idx = np.arange(0,7)

        for i in np.arange(len(delta_ndvi)-6):

            delta_window_to_smooth = delta_ndvi[i:i+7] # window to smooth, the center value will be appended
            ndvi_valid_to_check = ndvi_valid[i:i+7] # this will be used to check if the absolute value is close to the boundaries condition

            if (np.any((ndvi_valid_to_check < 0.05) | (ndvi_valid_to_check > 0.95))or (np.sum(delta_window_to_smooth < -0.2) >= 5)): 
                        
                # here, check for the NDVI close to the boundaries or extreme negative NDVI values (fire but not drought)                       
                # in case this conditions are met, skip the smoothing and keep the non-smoothed delta
                delta_ndvi_to_interpolate[i] = delta_window_to_smooth[3]

            else:
                    
                # smooth the 7 rolling window
                loess =  sm.nonparametric.lowess(delta_window_to_smooth, idx, frac= 1, it=3, return_sorted=False)
                delta_ndvi_to_interpolate[i] = loess[3]

            

        # combine smoothed value with values yet to smooth, after that linearly interpolate everything

        delta_ndvi_to_interpolate = np.concatenate([delta_ndvi_to_interpolate,delta_ndvi[-6:]]) 

        interpolated_values = np.interp(days_diff,days_diff_2,delta_ndvi_to_interpolate)

        ndvi_smoothed = np.array(10000 * (interpolated_values + full_median_array),  dtype=np.int16)


        # mask_array 
        mask_array[mask_valid_ndvi] = 2
        before = np.arange(len(mask_array)) <= original_idx_2[-3]

        outlier_idx = original_idx[1:-1][outlier_mask]
        valid_outlier_idx = outlier_idx[obs_date[outlier_idx] == 1]

        mask_array[ before & mask_valid_ndvi ] = 3
        mask_array[ before & (~mask_valid_ndvi) ] = 1

        mask_array[valid_outlier_idx] = 4

        return ndvi_smoothed, mask_array
        
    else:

        return ndvi_full , mask_array

if __name__ == "__main__":

    N_WORKERS = 30

    client = Client(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    memory_limit='200GB',
    processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
    dashboard_address=':1234')  
    print(client.dashboard_link)

    # already having medians computed

    #INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small.zarr" 
    INPUT_ZARR = "/data_3/scratch/francesco/processed/new_ndvi_dataset_spatial_short.zarr"
    INPUT_ZARR_LOOKUPTABLE = "/data_3/francesco/lookup_table_median_ndvi_v6.zarr"
    OUT_PATH = "/data_3/scratch/francesco/processed/short_historical.zarr"


    ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 10000})
    ds_median = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE, chunks={"date": -1, "pixel": 10000})

    # retrieve doy from date and use it for extract the median values
    doy = ds["datetime"].dt.dayofyear
    median_array_original = ds_median["median_ndvi"].sel(doy=doy)   # dims ("datetime","pixel")

    # create dataset of medians 
    full_dates = pd.date_range(
        start=pd.to_datetime(ds["datetime"].min().values),
        end=pd.to_datetime(ds["datetime"].max().values),
        freq="D"
    )

    full_dates_array = full_dates.values.astype("datetime64[D]") 

    ds_dates = ds["datetime"].values
    obs_date = np.isin(full_dates.values, ds_dates).astype(bool)

    full_dates_d = full_dates.values.astype("datetime64[D]")
    ds_dates_d = ds["datetime"].values.astype("datetime64[D]")

    obs_date = np.isin(full_dates_d, ds_dates_d)

    # create a zarr with doy and date, date will be used as coordinate for xr.apply
    full_doy = xr.DataArray(
        full_dates.dayofyear, 
        dims="date",
        name="doy")


    # build the full daily median array for that exact span
    full_median_array_original = ds_median["median_ndvi"].sel(doy=full_doy)

    dates_array = ds["date"].values.astype("datetime64[D]").ravel()   #.values.astype(np.int32)


    ndvi_array_original = ds["ndvi"]           # dims ("datetime","pixel")


    # call gufunc where core dim is "time" (1D arrays per pixel)
    ndvi_processed, mask_array = xr.apply_ufunc(
        historical_ndvi,
        ndvi_array_original,
        full_median_array_original,
        input_core_dims=[["datetime"],["date"]],    # each call gets 1D time arrays
        output_core_dims=[["date"],["date"]],
        vectorize=True, 
        dask="parallelized",
        kwargs={"obs_date" : obs_date},
        output_dtypes=[np.int16, np.int8],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # create the dataset to write 

    out_ds = xr.Dataset(
    {
        "ndvi_processed": ndvi_processed,
        "mask_array": mask_array
    })
    
    out_ds = out_ds.chunk({"date": -1, "pixel": 10000})


    # Clean everything
    for var in out_ds.variables:
        out_ds[var].attrs.pop('_FillValue', None)
        out_ds[var].encoding.clear()


    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)

    out_ds.to_zarr(OUT_PATH, mode="w", consolidated=True, compute=True)

    print("done")
    client.close()

