from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import os
import shutil
import time

#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_francesco.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_2.log 2>&1 &

def historical_ndvi(ndvi_arr_original, medians,mask_array_original, obs_dates,dates,starting_date):
        
        start_idx = np.searchsorted(dates, starting_date) 
        obs_prior = np.nonzero(obs_dates[:start_idx])[0]

        # Ensure mask_array is writable
        mask_array_original = np.array(mask_array_original, copy=True)


        if len(obs_prior) < 3:

            return ndvi_arr_original, mask_array_original

        crop_start = obs_prior[-3]  # Start at 3th prior obs, use to smooth
        ndvi_arr = ndvi_arr_original[crop_start:]
        medians = medians[crop_start:]
        obs_dates = obs_dates[crop_start:]
        dates = dates[crop_start:]
        mask_array = mask_array_original[crop_start:] 

        ndvi_not_analyzed =  ndvi_arr_original[:crop_start] 
        mask_array_not_analyzed = mask_array_original[:crop_start] 

        days_diff = (dates- dates[0])  / np.timedelta64(1, 'D')

        delta_ndvi = np.array([0])
     
        ndvi_arr = ndvi_arr / 10000
        medians  = medians  / 10000

        mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

        ndvi_valid = ndvi_arr[mask_valid_ndvi]
        median_valid = medians[mask_valid_ndvi]
        days_diff_2 = days_diff[mask_valid_ndvi]

        original_idx = np.arange(len(ndvi_arr)) # used to keep track of delta ndvi position and the outlier position
        original_idx = original_idx[mask_valid_ndvi]

        obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & obs_dates
        
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

            ndvi_smoothed = 10000 * (interpolated_values + medians)


            # indexing of array mask
            mask_array[obs_mask] = 2
            before = np.arange(len(mask_array)) < original_idx_2[-4]

            outlier_idx = original_idx[1:-1][outlier_mask]
            valid_outlier_idx = outlier_idx[obs_dates[outlier_idx] == 1]

            mask_array[ before & obs_mask ] = 3
            mask_array[ before & (~obs_mask) ] = 1

            mask_array[valid_outlier_idx] = 4


            mask_array_final =  np.concatenate([mask_array_not_analyzed, mask_array])
            final_ndvi_value =  np.concatenate([ndvi_not_analyzed, ndvi_smoothed])

            return final_ndvi_value, mask_array_final
        
        else:

            return ndvi_arr_original, mask_array_original

# used with nohup (ni idea why)

if __name__ == "__main__":

    t0 = time.perf_counter()

    N_WORKERS = 10

    client = Client(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    memory_limit='200GB',
    processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
    dashboard_address=':1234')  
    print(client.dashboard_link)

    # already having medians computed

    #INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small.zarr" 
    INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small_FB2026-03-17.zarr" 
    
    ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 500})

    ds_sub = ds.isel(pixel=slice(0, 1000))

    ndvi_array = ds_sub["ndvi"]           # dims ("time","pixel")
    median_array = ds_sub["median_ndvi"]    # dims ("time","pixel") 
    dates_array = ds_sub["date"].values.astype("datetime64[D]").ravel()   #.values.astype(np.int32)
    obs_dates = ds_sub["obs_date"]
    mask_array = ds_sub["mask_array"]

    starting_date = dates_array[365]

    # call gufunc where core dim is "time" (1D arrays per pixel)
    ndvi_processed, mask_array = xr.apply_ufunc(
        historical_ndvi,
        ndvi_array,
        median_array,
        mask_array,
        obs_dates,
        input_core_dims=[["date"], ["date"],["date"], ["date"]],    # each call gets 1D time arrays
        output_core_dims=[["date"],["date"]],
        vectorize=True, 
        dask="parallelized",
        kwargs={"dates": dates_array, "starting_date": starting_date},
        output_dtypes=[ndvi_array.dtype, np.int8],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


    # create the dataset to write 

    out_ds = xr.Dataset(
    {
        "ndvi_processed": ndvi_processed,
        "mask_array": mask_array
    },
    coords={
        "date": ds_sub["date"],
        "pixel": ds_sub["pixel"]
    }
    )
    out_ds = out_ds.chunk({"pixel": 5000, "date": -1})

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

    OUT_PATH = "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/data_for_demo_2"

    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)

    # Write using zarr version 2 to avoid new v3 codec/BytesBytesCodec mismatch
    out_ds.to_zarr(OUT_PATH, mode="w", consolidated=True, compute=True, encoding=encoding, zarr_version=3)

    print("done")
    client.close()
    

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds")
