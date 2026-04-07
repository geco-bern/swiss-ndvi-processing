from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import os
import shutil
import pandas as pd

#  nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_folder/2_test_historical_ndvi.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_folder/2_historical_ndvi_short_4.log 2>&1 &


# !!! Important
# some date have duplicate, if that the case (1148 unique enttry out of 1180)
# If that is the case, I take the average for value within the range

def historical_ndvi(ndvi_arr_original,full_median_array_original,obs_date):

    NO_COVERAGE = 32767

    # create evenly spaced ndvi and add obs to the rigth location
    ndvi_full = np.full(full_median_array_original.shape, NO_COVERAGE, dtype = np.int16)
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
        
    print(len(delta_ndvi))

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

        #print(f"len delta_ndvi_to_interpolate", len(delta_ndvi_to_interpolate))
        #print(f"len days_diff_2", len(days_diff_2))

        interpolated_values = np.interp(days_diff,days_diff_2,delta_ndvi_to_interpolate)

        #print(f"interpolated_value: ",interpolated_values)

        ndvi_smoothed = np.array((10000 * (interpolated_values + full_median_array)),  dtype=np.int16)

        ndvi_smoothed = np.clip(ndvi_smoothed, 0, 10000)

        #print(f"ndvi_smoothed: ",ndvi_smoothed)
        print(ndvi_smoothed)



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

    N_WORKERS = 20

    client = Client(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    memory_limit='1GB',
    processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
    dashboard_address=':1234')  
    print(client.dashboard_link)

    # already having medians computed



    import zarr
    import numpy as np

    # path al tuo tmp
    OUT = "/data_3/scratch/francesco/processed/new_ndvi_dataset_spatial_tmp_short.zarr"

    # apri solo l'array NDVI dal tmp
    ndvi_arr = zarr.open_array(store=f"{OUT}/ndvi", mode="r")

    # stampa shape per vedere se è ok
    mask = (ndvi_arr[3:103, 200:255] > 0) & (ndvi_arr[3:103, 200:255] < 10000)
    count = np.sum(mask)


    #INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small.zarr" 
    INPUT_ZARR = "/data_3/scratch/francesco/processed/new_ndvi_dataset_spatial_short_2.zarr" # for whole set "/data_3/scratch/francesco/processed/new_ndvi_dataset_spatial.zarr"
    INPUT_ZARR_LOOKUPTABLE = "/data_3/francesco/lookup_table_median_ndvi.zarr"
    OUT_PATH = "/data_3/scratch/francesco/processed/short_historical.zarr"

    NO_COVERAGE = 32767
    INVALID     = -32768


    ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 10000}, mask_and_scale= False)
    ds_median = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE, chunks={"date": -1, "pixel": 10000}, mask_and_scale= False).isel(pixel = 0)


    # retrieve doy from date and use it for extract the median values
    dayofyear_pos = (ds['datetime'].dt.dayofyear - 1).astype('int64').clip(0, 364)


    # this is only used with a short dataset to createa dummy data
    median_array_original = ds_median["median_ndvi"].isel(doy=dayofyear_pos)   # dims ("datetime","pixel")

    # create dataset of medians 
    datetime_values = ds["datetime"].compute().values

    # check number of entries
    print(len(datetime_values))
    
    # remove missing date
    datetime_clean = datetime_values[~np.isnat(datetime_values)]
    datetime_clean_days = datetime_clean.astype("datetime64[D]")

    # check number of clean entries
    print(len(datetime_clean_days))

    # check number of unique clean entries
    print(len(np.unique(datetime_clean_days)))


    # clip to first and last obs date
    dt_min = np.min(datetime_clean_days)
    dt_max = np.max(datetime_clean_days)
    full_dates = pd.date_range(start=dt_min, end=dt_max, freq="D")

    # generate evenly spaced data
    full_dates_array = full_dates.values.astype("datetime64[D]") 

    # boolean array of where the obs. is located
    obs_date = np.isin(full_dates_array, datetime_clean_days).astype(bool)


    # check, should the equal to print(len(np.unique(datetime_clean_days)))
    print(np.sum(obs_date))


    ds_dates_d = ds["datetime"].values.astype("datetime64[D]")

    obs_date = np.isin(full_dates_array, ds_dates_d)

    

    # create a zarr with doy and date, date will be used as coordinate for xr.apply
    doy_for_full_dates = np.minimum(pd.to_datetime(full_dates_array).dayofyear.values, 365)



    full_doy = xr.DataArray(doy_for_full_dates, dims="date")


    # build the full daily median array for that exact span
    full_median_array_original = ds_median["median_ndvi"].sel(doy=full_doy)



    ndvi_array_original = ds["ndvi"].isel(pixel = 0)
    ndvi_array_original = ndvi_array_original.where(
    (ndvi_array_original != NO_COVERAGE) & (ndvi_array_original != INVALID),
    other=NO_COVERAGE).astype(np.int16)


    # remove wrong date date
    valid_time_mask = ~np.isnat(ds["datetime"].values)
    ds_filtered = ds.isel(datetime = valid_time_mask)


    valid = ds_filtered["ndvi"].isel(pixel=(slice(0,1000)))
    ndvi_avg = (valid.groupby(valid.datetime.dt.date).mean(dim="datetime", skipna=False))

    # jsut to print the len
    ndvi_test = valid.load().to_numpy()

    print(valid)

    # check if all data are zeros or not
    print(np.sum(ndvi_test))


    # call gufunc where core dim is "time" (1D arrays per pixel)
    ndvi_processed, mask_array = xr.apply_ufunc(
        historical_ndvi,
        ndvi_avg,
        full_median_array_original,
        input_core_dims=[["datetime"],["date"]],    # each call gets 1D time arrays
        output_core_dims=[["date"],["date"]],
        vectorize=True, 
        dask="parallelized",
        kwargs={"obs_date" : obs_date},
        output_dtypes=[np.int16, np.int8],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # force int16
    ndvi_processed = ndvi_processed.astype(np.int16, copy=True)
    mask_array = mask_array.astype(np.int8, copy=True)
    
    # create the dataset to write 
    out_ds = xr.Dataset(
        {
            "ndvi_processed": ndvi_processed,
            "mask_array": mask_array
        },
        coords={
            "date": full_dates_array,
            "pixel": ds["pixel"]
        }
    )

    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)

    # 1. Re‑chunk only the data variables (big arrays)
    out_ds = out_ds.chunk({
        "date": -1,
        "pixel": 10000
    })

    # 2. PREPARE ENCODING to force int16 and _FillValue, not attributes on out_ds
    encoding = {}
    encoding["ndvi_processed"] = {
        "dtype": np.int16,
        "_FillValue": np.int16(32767),   # your NO_COVERAGE
        "compressor": None,
    }
    encoding["mask_array"] = {
        "dtype": np.int8,
        "_FillValue": np.int8(0),
        "compressor": None,
    }

    # 3. Do NOT add _FillValue as an attribute on the dataset variables
    #    xarray will pick it up from `encoding` in to_zarr
    for coord_name in out_ds.coords:
        out_ds[coord_name].encoding.pop("chunks", None)
        out_ds[coord_name].encoding.pop("compressor", None)

    out_ds.to_zarr(
        OUT_PATH,
        mode="w",
        consolidated=True,
        compute=True,
        encoding=encoding,
        zarr_format=3
    )

    test = xr.open_zarr(OUT_PATH, chunks={"date": -1, "pixel": 10000}, mask_and_scale= False)

    print(test)

    ndvi_2 = test["ndvi_processed"].load().to_numpy()

    print(ndvi_2)

    
    client.close()