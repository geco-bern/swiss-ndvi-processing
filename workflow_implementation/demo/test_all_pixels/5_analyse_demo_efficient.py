from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import os
import shutil
import time

#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient_FB_2026-03-19.log 2>&1 &

def historical_ndvi(ndvi_arr_original, medians,mask_array_original, obs_dates,dates,starting_date):
        
        start_idx = np.searchsorted(dates, starting_date) 
        obs_prior = np.nonzero(obs_dates[:start_idx])

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

            idx = np.arange(len(delta_ndvi))

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

            delta_ndvi_to_interpolate = np.concatenate([np.array([0]),loess[:-3],delta_ndvi[-3:],np.array([0])]) 
            dates_to_interpolate = np.concatenate([np.array([0]),days_diff_2,np.array([days_diff[-1]])]) 

            interpolated_values = np.interp(days_diff,dates_to_interpolate,delta_ndvi_to_interpolate)

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

    #INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small.zarr" 
    #INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small_FB2026-03-17.zarr" 
    #ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 500})
                # <xarray.Dataset> Size: 164MB
                # Dimensions:      (pixel: 10000, date: 3265)
                # Coordinates:
                #   * pixel        (pixel) int64 80kB 0 1 2 3 4 5 ... 9995 9996 9997 9998 9999
                #   * date         (date) datetime64[ns] 26kB 2017-04-03 2017-04-04 ... 2026-03-10
                #     doy          (date) int64 26kB dask.array<chunksize=(3265,), meta=np.ndarray>
                #     y            (pixel) int64 80kB dask.array<chunksize=(500,), meta=np.ndarray>
                #     x            (pixel) int64 80kB dask.array<chunksize=(500,), meta=np.ndarray>
                # Data variables:
                #     obs_date     (date) bool 3kB dask.array<chunksize=(3265,), meta=np.ndarray>
                #     mask_array   (pixel, date) int8 33MB dask.array<chunksize=(500, 3265), meta=np.ndarray>
                #     median_ndvi  (pixel, date) int16 65MB dask.array<chunksize=(500, 3265), meta=np.ndarray>
                #     ndvi         (pixel, date) int16 65MB dask.array<chunksize=(500, 3265), meta=np.ndarray>
    INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr"
    HISTO_ZARR_IN_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m.zarr"
    HISTO_ZARR_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m_extended.zarr" # TODO: remove this and instea do it circular
    INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_1000mX1000m_4th.zarr"
    historic_ds = xr.open_zarr(HISTO_ZARR_IN_OUTPUT, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    new_ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 500})
    
    # --- concatenate full datasets along time ----------------------------------
    # Add median NDVI from model
    lookuptable  = xr.open_zarr(INPUT_LOOKUPTABLE, consolidated=False)
    
    # to new_ds:
    doy_noLeap = xr.where(new_ds.doy == 366, 365, new_ds.doy) # remove leap year if encountered
    new_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_noLeap,
            pixel=new_ds.pixel) # this is to join by pixels and doy
    # to historic_ds:
    doy_noLeap = xr.where(historic_ds.doy == 366, 365, historic_ds.doy) # remove leap year if encountered
    historic_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_noLeap,
            pixel=historic_ds.pixel) # this is to join by pixels and doy

    # Add mask_array to new ds (filled with 0 or 2):
        # mask_array == 0: the data is not an observation and is yet to be smoothed
        # mask_array == 1: the data is not an observation and is smoothed
        # mask_array == 2: the data is an observation and is yet to be smoothed
        # mask_array == 3: the data is an observation and is smoothed
        # mask_array == 4: the data is an observation and is an outlier
    mask_0or2_1D = xr.where(new_ds["obs_date"], 2, 0).astype(np.int8)   # dims: date
    mask_0or2_2D = mask_0or2_1D.expand_dims({"pixel": new_ds.pixel})
    new_ds = new_ds.assign(mask_array=mask_0or2_2D)

    # --- concatenate full datasets along time ----------------------------------
    # Bind together with historic:
    new_ds = new_ds.rename(
        {'ndvi_obs':'ndvi_processed',
            'ndsi_obs':'ndsi_processed'}
    ).drop_vars('ndsi_processed')
    
    merged_ds = (
        xr.concat(
            [historic_ds, new_ds], 
            dim="date")
        .sortby("date")
    )
    merged_ds = merged_ds.chunk(
        {"pixel": PIXEL_CHUNKS, 
            "date": DATE_CHUNKS})
    ds = merged_ds

    # --- apply gapfilling and outlier detection function: historical_ndvi() ----------------------------------

    # prepare arguments spanning historic and new data: all lazy
    ndvi_array   = ds["ndvi_processed"]           # dims ("date","pixel")
    median_array = ds["median_ndvi"]              # dims ("date","pixel") 
    dates_array  = ds["date"].values.astype("datetime64[D]").ravel()   #.values.astype(np.int32)
    obs_dates    = ds["obs_date"]
    mask_array   = ds["mask_array"]

    # specifying where new data starts
    starting_date = dates_array[365] # TODO: fix this

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
        output_dtypes=[ndvi_array.dtype, obs_dates.dtype], # TODO: this shoudl be mask_array.dtype
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # create the dataset to write 
    out_ds = xr.Dataset(
    {
        "ndvi_processed": ndvi_processed,
        "mask_array": mask_array
    },
    coords={
        "date": ds["date"],
        "pixel": ds["pixel"]
    }
    )
    out_ds = out_ds.chunk({"pixel": 5000, "date": -1})
    out_ds.compute()

    # Remove any incompatible 'compressor' metadata left over from the source dataset
    for v in list(out_ds.data_vars):
        out_ds[v].encoding.pop("compressor", None)
        # ensure chunks entry exists to avoid surprises
        out_ds[v].encoding.setdefault("chunks", None)

    for c in list(out_ds.coords):
        out_ds[c].encoding.pop("compressor", None)
        out_ds[c].encoding.setdefault("chunks", None)

    # Explicit encoding: no compressor for each data var
    encoding = {v: {"compressor": None} for v in out_ds.data_vars} # TODO: why not? this should be following what was done to create v4 of historic

    if os.path.exists(HISTO_ZARR_OUTPUT): # TODO: remove this when going circular
        shutil.rmtree(HISTO_ZARR_OUTPUT)  # TODO: remove this when going circular

    out_ds.to_zarr(
          HISTO_ZARR_OUTPUT, 
          mode="w", 
          consolidated=True, 
          compute=True, 
          encoding=encoding, 
          zarr_version=3)

    print("done")
    client.close()
    

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds")
