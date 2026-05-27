from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import dask as dask
import xarray as xr
import os
import shutil

# OLD:  nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/benchamrk_historic_ndvi_parallel.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/historic_analysis.log &
# OLD: nohup python -u "/home/fabian/GitHub/geco-bern/swiss-ndvi-processing/workflow_implementation/ignore this/benchamrk_historic_ndvi_parallel.py" > /data_3/scratch/fabian/2026-04-10_ndvi_benchamrk_historic_ndvi_parallel_2026-04-10_ndvi_benchamrk_historic_ndvi_parallel_2e22cb0b5c71b5f8e7ecc4b9634059926957930c.zarr.log &
# nohup python -u "/home/fabian/GitHub/geco-bern/swiss-ndvi-processing/workflow_implementation/MS1_historical/new_historical_processing/0_0_benchmark_historic_processing.py" > /data_3/scratch/fabian/2026-04-10_ndvi_benchamrk_historic_ndvi_parallel_2026-04-10_ndvi_benchamrk_historic_ndvi_parallel_05d66be11177ff198a7f741b340e7a4fb1de642a.zarr.log &
# python -u "/home/fabian/GitHub/geco-bern/swiss-ndvi-processing/workflow_implementation/MS1_historical/new_historical_processing/0_0_benchmark_historic_processing.py"

# TODO: this one to modify:
def historical_ndvi_to_optimize(ndvi_array, median_array, mask_array, is_observation_date, dates_array, starting_date):
    """
    # ndvi_array          # this is the observed/gapfilled/processed NDVI value
    # median_array        # this is the modelled median NDVI for the corresponding DOY
    # mask_array          # this is the integer processing status
    # is_observation_date # this is the True-False boolean if a date contains satellite images (is_observation_date?)
    # kwargs={"dates_array":   dates_array,  # this contains all daily dates
    #         "starting_date": start_date}   # this contains the starting date when to start ??

    # in continuous case this function receives arrays that 
    # start with already processed (daily) historic values
    # followed by a range of daily values, containing empty or observed NDVI
    # ndvi_array =>  [hist, hist, hist, hist,   empty, obs, empty, empty, obs, empty, obs]
    #            =>  [hist, hist, hist, hist,   32767, obs, 32767, 32767, obs, 32767, obs]
    # mask_array  => [hist, hist, hist, hist,   0  ,   2  , 0  ,   0  ,   2  , 0  ,   2  ]
    # is_obs_date => [T   , T   , F   , T   ,   F  ,   T  , F  ,   F  ,   T  , F  ,   T  ]
    # median_array=> [hist, hist, hist, hist,   new,   new, new,   new,   new, new,   new]
    # dates_array => [20170403,..,.., 20170406, 20170407, 20170408, 20170409, ...]
    # starting_date => 20170407
    """

    is_historic_case = True # this can be set to False for continuous case

    if not len(ndvi_array.shape)          == 1: raise Exception( "Expected 1D array as ndvi_array" )
    if not len(median_array.shape)        == 1: raise Exception( "Expected 1D array as median_array" )
    if not len(mask_array.shape)          == 1: raise Exception( "Expected 1D array as mask_array" )
    if not len(is_observation_date.shape) == 1: raise Exception( "Expected 1D array as is_observation_date" )
    

    # Ensure mask_array is writable
    mask_array = np.array(mask_array, copy=True) # TODO: check if this is needed
    # # in the historic case, this function receives arrays that are lacking a previously generated historic part
    # if is_historic_case:
        # crop_start = 0   # NOTE when crop_start is set to 0, 
                          # this would lead to empty "_already_processed"-arrays

    # subset data into new and old (already processed) data in the continuous case:
    if not is_historic_case: # TODO: in the historic case this is deactivated
        start_idx = np.searchsorted(dates_array, starting_date)                   
        obs_prior = np.nonzero(is_observation_date[:start_idx])
        if len(obs_prior) < 3:
                return ndvi_arr_original, mask_array_original

        crop_start = obs_prior[-3]  # Start at 3th prior obs, use to smooth 

        ndvi_array          = ndvi_array[crop_start:]
        median_array        = median_array[crop_start:]
        is_observation_date = is_observation_date[crop_start:]
        dates_array         = dates_array[crop_start:]
        mask_array          = mask_array[crop_start:]

        ndvi_array_already_processed = ndvi_array[:crop_start]
        mask_array_already_processed = mask_array[:crop_start]

    if is_historic_case:
        days_diff = (dates_array- dates_array[0])  / np.timedelta64(1, 'D') # in continuous_case
        
        # days_diff = np.arange(0, len(is_observation_date)) # TODO # TODO 5_analyse_demo_efficient defines days_diff differently
        #TODO: Francesco, why was this different from dates_array - dates_array[0]. Was it supposed to ignore the actual dates for the historic 
        #                 processing and assume all observations are spaced by one day exactly?

    if not is_historic_case:
        days_diff = (dates_array- dates_array[0])  / np.timedelta64(1, 'D') # in continuous_case

    # renaming is necessary otherwise won't work 
    ndvi_arr        = ndvi_array   / 10000
    median_arr      = median_array / 10000
    mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

    ndvi_valid   = ndvi_arr[  mask_valid_ndvi & is_observation_date] # TODO 5_analyse_demo_efficient is not using is_observation_date here
    median_valid = median_arr[mask_valid_ndvi & is_observation_date] # TODO 5_analyse_demo_efficient is not using is_observation_date here
    days_diff_2  = days_diff[ mask_valid_ndvi & is_observation_date] # TODO 5_analyse_demo_efficient is not using is_observation_date here

    original_idx = np.arange(len(ndvi_arr)) # used to keep track of delta ndvi position and the outlier position
    original_idx = original_idx[mask_valid_ndvi & is_observation_date]      # TODO 5_analyse_demo_efficient is not using is_observation_date here

    obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & is_observation_date
        
    # outlier detection

    delta_threshold = 0.1
    delta_delta_threshold = 0.1

    delta_ndvi = ndvi_valid - median_valid
    delta_delta_left = delta_ndvi[2:]   # TODO: shouldnt this be a difference of deltas?
    delta_delta_rigth = delta_ndvi[:-2] # TODO: shouldnt this be a difference of deltas?
    outlier_mask = ((abs(delta_ndvi[1:-1]) > delta_threshold) &       # TODO: shouldn't this be a OR
                    (abs(delta_delta_left) > delta_delta_threshold) & # TODO: shouldn't this be a OR
                    (abs(delta_delta_rigth) > delta_delta_threshold))
    ndvi_valid = ndvi_valid[1:-1][~outlier_mask]
    delta_ndvi = delta_ndvi[1:-1][~outlier_mask]
    days_diff_2 = days_diff_2[1:-1][~outlier_mask]

    original_idx_2 = original_idx[1:-1][~outlier_mask]
        

    # L2 smoothing of all observations except the last 6 observations
    # some sites do not have any observation or very few
    if len(delta_ndvi) > 6:
        
        # L2 smoothing
        # loop over the 7 rolling deltas. If the deltas are too large (extreme events as fire) 
        # or the original values too close to the boundaries condition (0.9 and 0.1) we do linear fit

        delta_ndvi_to_interpolate = np.full(len(delta_ndvi)-6, np.nan)

        idx = np.arange(0,7) # NOTE: this uses indices just from a 7 day window
        # TODO: TODO 5_analyse_demo_efficient is using idx = np.arange(len(delta_ndvi)).  # This uses all the indices
        #            Thereby, 5_analyse_demo_efficient did smoothing the full data set in a single window from start to almost end

        for i in np.arange(0, len(delta_ndvi)-6): # loop over each indidivual day with 
                                                  # a rolling window (width of 7)
                                                  # from 0 to 7th last

            delta_window_to_smooth = delta_ndvi[i:i+7] # window to smooth, the center value will be appended
            ndvi_window_to_check   = ndvi_valid[i:i+7] # window to check the absolute NDVI value 

            # check if absolute value is close to the boundaries condition
            if (np.any((ndvi_window_to_check < 0.05) | (ndvi_window_to_check > 0.95)) 
                or (np.sum(delta_window_to_smooth < -0.2) >= 5)): 
                        
                # here, check for the NDVI close to the boundaries or extreme negative NDVI values (fire but not drought)                       
                # in case this conditions are met, skip the smoothing and keep the non-smoothed delta
                delta_ndvi_to_interpolate[i] = delta_window_to_smooth[3]

            else:
                
                # smooth the 7 rolling window
                loess =  sm.nonparametric.lowess(
                    delta_window_to_smooth, 
                    idx, 
                    frac = 1, 
                    it=3, 
                    return_sorted=False)
                delta_ndvi_to_interpolate[i] = loess[3]

            

        # combine smoothed value with values yet to smooth, after that linearly interpolate everything

        delta_ndvi_to_interpolate = np.concatenate([
            delta_ndvi_to_interpolate, 
            delta_ndvi[-6:]
        ])
        dates_to_interpolate = np.concatenate([
            days_diff_2
        ]) # TODO: in 5_analyse_demo_efficient we have: dates_to_interpolate = np.concatenate([np.array([0]),days_diff_2,np.array([days_diff[-1]])]) 

        interpolated_values = np.interp(
            days_diff,
            dates_to_interpolate,
            delta_ndvi_to_interpolate
        )

        ndvi_smoothed = np.array(
            (10000 * (interpolated_values + median_arr)),  
            dtype=np.int16)
        ndvi_smoothed = np.clip(ndvi_smoothed, 0, 10000)
        # TODO: ndvi_smoothed is generated differently in 5_analyse_demo_efficient
        #       there it was not clipped to 0, 10000 and not specifically turned into an np.array

        # indexing of array mask 
        mask_array[obs_mask] = 2
        before = np.arange(len(mask_array)) <= original_idx_2[-3]
        # TODO: 5_analyse_demo_efficient uses: <= original_idx_2[-4] here.

        outlier_idx = original_idx[1:-1][outlier_mask]
        valid_outlier_idx = outlier_idx[is_observation_date[outlier_idx] == 1]

        mask_array[ before & obs_mask ] = 3
        mask_array[ before & (~obs_mask) ] = 1

        mask_array[valid_outlier_idx] = 4

        if is_historic_case:
            return ndvi_smoothed, mask_array
        
        if not is_historic_case: # continuous case
            mask_array_final =  np.concatenate([mask_array_already_processed, mask_array])
            ndvi_value_final =  np.concatenate([ndvi_already_processed, ndvi_smoothed])
            return ndvi_value_final, mask_array_final

    else:

        return ndvi_array , mask_array

def historical_ndvi_2nd_slow(ndvi_array, median_array, mask_array, is_observation_date, dates_array, starting_date):
    """
    # ndvi_array          # this is the observed/gapfilled/processed NDVI value
    # median_array        # this is the modelled median NDVI for the corresponding DOY
    # mask_array          # this is the integer processing status
    # is_observation_date # this is the True-False boolean if a date contains satellite images (is_observation_date?)
    # kwargs={"dates_array":   dates_array,  # this contains all daily dates
    #         "starting_date": start_date}   # this contains the starting date when to start ??

    # in continuous case this function receives arrays that 
    # start with already processed (daily) historic values
    # followed by a range of daily values, containing empty or observed NDVI
    # ndvi_array =>  [hist, hist, hist, hist,   empty, obs, empty, empty, obs, empty, obs]
    #            =>  [hist, hist, hist, hist,   32767, obs, 32767, 32767, obs, 32767, obs]
    # mask_array  => [hist, hist, hist, hist,   0  ,   2  , 0  ,   0  ,   2  , 0  ,   2  ]
    # is_obs_date => [T   , T   , F   , T   ,   F  ,   T  , F  ,   F  ,   T  , F  ,   T  ]
    # median_array=> [hist, hist, hist, hist,   new,   new, new,   new,   new, new,   new]
    # dates_array => [20170403,..,.., 20170406, 20170407, 20170408, 20170409, ...]
    # starting_date => 20170407
    """

    is_historic_case = True # this can be set to False for continuous case

    if not len(ndvi_array.shape)          == 1: raise Exception( "Expected 1D array as ndvi_array" )
    if not len(median_array.shape)        == 1: raise Exception( "Expected 1D array as median_array" )
    if not len(mask_array.shape)          == 1: raise Exception( "Expected 1D array as mask_array" )
    if not len(is_observation_date.shape) == 1: raise Exception( "Expected 1D array as is_observation_date" )
    

    # Ensure mask_array is writable
    mask_array = np.array(mask_array, copy=True) # TODO: check if this is needed
    # in the historic case this function receives arrays that are lacking a previously generated historic part

    # split the input arguments in two: (only needed for continuous case)


    if is_historic_case:
        crop_start = 0   # NOTE when crop_start is set to 0, this would lead to empty "_not_processed"-arrays

    if not is_historic_case: # TODO: in the historic case this is deactivated
        start_idx = np.searchsorted(dates_array, starting_date)                   
        obs_prior = np.nonzero(is_observation_date[:start_idx])             # TODO: in the historic case this is deactivated
        if len(obs_prior) < 3:                                              # TODO: in the historic case this is deactivated
                return ndvi_arr_original, mask_array_original               # TODO: in the historic case this is deactivated
        crop_start = obs_prior[-3]  # Start at 3th prior obs, use to smooth # TODO: in the historic case this is deactivated

        ndvi_array_not_processed = ndvi_array[:crop_start]
        mask_array_not_processed = mask_array[:crop_start]

    ndvi_array          = ndvi_array[crop_start:]
    median_array        = median_array[crop_start:]
    is_observation_date = is_observation_date[crop_start:]
    dates_array         = dates_array[crop_start:]
    mask_array          = mask_array[crop_start:]


    if is_historic_case:
        days_diff = (dates_array- dates_array[0])  / np.timedelta64(1, 'D') # in continuous_case
        
        # days_diff = np.arange(0, len(is_observation_date)) # TODO # TODO 5_analyse_demo_efficient defines days_diff differently
        #TODO: Francesco, why was this different from dates_array - dates_array[0]. Was it supposed to ignore the actual dates for the historic 
        #                 processing and assume all observations are spaced by one day exactly?

    if not is_historic_case:
        days_diff = (dates_array- dates_array[0])  / np.timedelta64(1, 'D') # in continuous_case

    # renaming is necessary otherwise won't work 
    ndvi_arr = ndvi_array / 10000
    full_median_array = median_array / 10000

    mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

    ndvi_valid = ndvi_arr[mask_valid_ndvi & is_observation_date]            # TODO 5_analyse_demo_efficient is not using is_observation_date here
    median_valid = full_median_array[mask_valid_ndvi & is_observation_date] # TODO 5_analyse_demo_efficient is not using is_observation_date here
    days_diff_2 = days_diff[mask_valid_ndvi & is_observation_date]          # TODO 5_analyse_demo_efficient is not using is_observation_date here

    original_idx = np.arange(len(ndvi_array)) # used to keep track of delta ndvi position and the outlier position
    original_idx = original_idx[mask_valid_ndvi & is_observation_date]      # TODO 5_analyse_demo_efficient is not using is_observation_date here

    obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & is_observation_date
        
    # outlier detection

    delta_threshold = 0.1
    delta_delta_threshold = 0.1

    delta_ndvi = ndvi_valid - median_valid
    delta_delta_left = delta_ndvi[2:]   # TODO: shouldnt this be a difference of deltas?
    delta_delta_rigth = delta_ndvi[:-2] # TODO: shouldnt this be a difference of deltas?
    outlier_mask = ((abs(delta_ndvi[1:-1]) > delta_threshold) &       # TODO: shouldn't this be a OR
                    (abs(delta_delta_left) > delta_delta_threshold) & # TODO: shouldn't this be a OR
                    (abs(delta_delta_rigth) > delta_delta_threshold))
    ndvi_valid = ndvi_valid[1:-1][~outlier_mask]
    delta_ndvi = delta_ndvi[1:-1][~outlier_mask]
    days_diff_2 = days_diff_2[1:-1][~outlier_mask]

    original_idx_2 = original_idx[1:-1][~outlier_mask]
        

    # L2 smoothing of all observations except the last 6 observations
    # some sites do not have any observation or very few
    if len(delta_ndvi) > 6:
        
        # L2 smoothing
        # loop over the 7 rolling deltas. If the deltas are too large (extreme events as fire) 
        # or the original values too close to the boundaries condition (0.9 and 0.1) we do linear fit

        delta_ndvi_to_interpolate = np.full(len(delta_ndvi)-6, np.nan)

        idx = np.arange(0,7) # TODO: TODO 5_analyse_demo_efficient is using idx = np.arange(len(delta_ndvi))

        for i in np.arange(0, len(delta_ndvi)-6): # loop from 0 to 7th last

            delta_window_to_smooth = delta_ndvi[i:i+7] # window to smooth, the center value will be appended
            ndvi_valid_to_check    = ndvi_valid[i:i+7] # this will be used to check if the absolute value is close to the boundaries condition

            if (np.any((ndvi_valid_to_check < 0.05) | (ndvi_valid_to_check > 0.95)) 
                or (np.sum(delta_window_to_smooth < -0.2) >= 5)): 
                        
                # here, check for the NDVI close to the boundaries or extreme negative NDVI values (fire but not drought)                       
                # in case this conditions are met, skip the smoothing and keep the non-smoothed delta
                delta_ndvi_to_interpolate[i] = delta_window_to_smooth[3]

            else:
                
                # smooth the 7 rolling window
                loess =  sm.nonparametric.lowess(delta_window_to_smooth, idx, frac= 1, it=3, return_sorted=False)
                delta_ndvi_to_interpolate[i] = loess[3]

            

        # combine smoothed value with values yet to smooth, after that linearly interpolate everything

        delta_ndvi_to_interpolate = np.concatenate([delta_ndvi_to_interpolate, delta_ndvi[-6:]]) 
        # TODO: in 5_analyse_demo_efficient we have: dates_to_interpolate = np.concatenate([np.array([0]),days_diff_2,np.array([days_diff[-1]])]) 

        interpolated_values = np.interp(days_diff,days_diff_2,delta_ndvi_to_interpolate)

        ndvi_smoothed = np.array((10000 * (interpolated_values + full_median_array)),  dtype=np.int16)
        ndvi_smoothed = np.clip(ndvi_smoothed, 0, 10000)
        # TODO: ndvi_smoothed is generated differently in 5_analyse_demo_efficient
        # simply one statement: ndvi_smoothed = 10000 * (interpolated_values + medians)

        # indexing of array mask 
        mask_array[obs_mask] = 2
        before = np.arange(len(mask_array)) <= original_idx_2[-3] # TODO: 5_analyse_demo_efficient uses: < -4 here.

        outlier_idx = original_idx[1:-1][outlier_mask]
        valid_outlier_idx = outlier_idx[is_observation_date[outlier_idx] == 1]

        mask_array[ before & obs_mask ] = 3
        mask_array[ before & (~obs_mask) ] = 1

        mask_array[valid_outlier_idx] = 4

        if is_historic_case:
            return ndvi_smoothed, mask_array
        
        if not is_historic_case: # continuous case
            mask_array_final =  np.concatenate([mask_array_not_processed, mask_array])
            final_ndvi_value =  np.concatenate([ndvi_not_processed, ndvi_smoothed])
            return final_ndvi_value, mask_array_final

    else:

        return ndvi_array , mask_array

def historical_ndvi_1st_fast_5000_120s(ndvi_arr, median_arr,is_observation_date,dates):
        
        # initialize mask array
        mask_array  = np.empty(len(is_observation_date), dtype=object)
        mask_array.fill(0)

        days_diff = (dates- dates[0])  / np.timedelta64(1, 'D')
     
        ndvi_arr = ndvi_arr / 10000
        median_arr  = median_arr  / 10000
        mask_valid_ndvi = (ndvi_arr > 0) & (ndvi_arr < 1)

        ndvi_valid   = ndvi_arr[  mask_valid_ndvi]
        median_valid = median_arr[mask_valid_ndvi]
        days_diff_2  = days_diff[ mask_valid_ndvi]

        original_idx = np.arange(len(ndvi_arr)) # used to keep track of delta ndvi position and the outlier position
        original_idx = original_idx[mask_valid_ndvi]

        obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & is_observation_date
        
        # outlier detection

        delta_threshold = 0.1
        delta_delta_threshold = 0.1

        delta_ndvi = ndvi_valid - median_valid
        delta_delta_left = delta_ndvi[2:]
        delta_delta_rigth = delta_ndvi[:-2]
        outlier_mask = ((abs(delta_ndvi[1:-1]) > delta_threshold) & 
                        (abs(delta_delta_left) > delta_delta_threshold) & 
                        (abs(delta_delta_rigth) > delta_delta_threshold))
        ndvi_valid = ndvi_valid[1:-1][~outlier_mask]
        delta_ndvi = delta_ndvi[1:-1][~outlier_mask]
        days_diff_2 = days_diff_2[1:-1][~outlier_mask]

        original_idx_2 = original_idx[1:-1][~outlier_mask]
        

        # some sites do not have any observation or very few
        if len(delta_ndvi) > 6:
        
            # L2 smoothing
            # smooth the full data set in a single window from start to almost end
            idx = np.arange(len(delta_ndvi)) # This uses all the indices
            loess =  sm.nonparametric.lowess(delta_ndvi, idx, frac= 7 / len(delta_ndvi), it=3, return_sorted=False)

            # combine smoothed value with values yet to smooth, after that linearly interpolate everything

            delta_ndvi_to_interpolate = np.concatenate([
                np.array([0]),
                loess[:-4],
                delta_ndvi[-4:],
                np.array([0])
            ]) 
            dates_to_interpolate = np.concatenate([
                np.array([0]),
                days_diff_2,
                np.array([days_diff[-1]])
            ]) 

            interpolated_values = np.interp(
                days_diff,
                dates_to_interpolate,
                delta_ndvi_to_interpolate
            )

            ndvi_smoothed = 10000 * (interpolated_values + median_arr)

            # indexing of array mask
            mask_array[obs_mask] = 2
            before = np.arange(len(mask_array)) < original_idx_2[-4]

            outlier_idx = original_idx[1:-1][outlier_mask]
            valid_outlier_idx = outlier_idx[is_observation_date[outlier_idx] == 1]

            mask_array[ before & obs_mask ] = 3
            mask_array[ before & (~obs_mask) ] = 1

            mask_array[valid_outlier_idx] = 4

            return ndvi_smoothed, mask_array
        
        else:

            return 10000 * ndvi_arr, mask_array

# used with nohup (ni idea why)

if __name__ == "__main__":

    # N_WORKERS = 50

    # client = Client(
    # n_workers=N_WORKERS,
    # threads_per_worker=1,
    # memory_limit='200GB',
    # processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
    # dashboard_address=':1234')  
    # print(client.dashboard_link)

    INPUT_ZARR = "/data_3/scratch/francesco/zarr_to_historical_all_pixels.zarr" #"/data_3/scratch/francesco/zarr_demo_daily_v2.zarr/"
    ds = xr.open_zarr(INPUT_ZARR, chunks={"date": -1, "pixel": 5000})
    ds = ds.isel(pixel = slice(0,10000)) # TODO: for development

    ndvi_array     = ds["ndvi"]           # dims ("time","pixel")
    median_array   = ds["median_ndvi"]    # dims ("time","pixel") 
    dates_array    = ds["date"].values.astype("datetime64[D]").ravel()   #.values.astype(np.int32)
    obs_dates      = ds["obs_date"]
    start_date_arg = dates_array[0]

    arg1 = ndvi_array.isel(pixel=0).values
    arg2 = median_array.isel(pixel=0).values
    arg3 = np.zeros(len(obs_dates), dtype=np.int8)
    arg4 = obs_dates.values
    Nruns = 10
    t0 = datetime.now()
    for it in range(Nruns):
        historical_ndvi_1st_fast_5000_120s(arg1,arg2,arg4,dates = dates_array)
    t1 = datetime.now(); print(t1-t0, flush = True)    # with 120s for 5000, we should see 10 for 120s/500 = 0.24s
    for it in range(Nruns):
        historical_ndvi_2nd_slow(arg1,arg2,arg3,arg4,dates_array = dates_array,starting_date = start_date_arg)
    t2 = datetime.now(); print(t2-t1, flush = True)
    for it in range(Nruns):
        historical_ndvi_to_optimize(arg1,arg2,arg3,arg4,dates_array = dates_array,starting_date = start_date_arg)
    t3 = datetime.now(); print(t3-t2, flush = True)


    # # call gufunc where core dim is "time" (1D arrays per pixel)
    # ndvi_processed, mask_array = xr.apply_ufunc(
    #     # variant 1:
    #     #historical_ndvi_1st_fast_5000_120s,
    #     #ndvi_array,
    #     #median_array,
    #     #obs_dates,
    #     #kwargs={"dates": dates_array},
    #     #input_core_dims=[["date"], ["date"],["date"]],    # each call gets 1D time arrays
    #     # variant 2:
    #     historical_ndvi,
    #     ndvi_array,
    #     median_array,
    #     np.zeros(len(obs_dates), dtype=np.int8), # TODO: mask_array
    #     obs_dates,
    #     kwargs={"dates_array": dates_array,
    #             "starting_date": start_date_arg},
    #     input_core_dims=[["date"],["date"],["date"],["date"]],    # each call gets 1D time arrays
        
    #     # for both variants:
    #     output_core_dims=[["date"],["date"]],
    #     vectorize=True, 
    #     dask="parallelized",
    #     output_dtypes=[ndvi_array.dtype, obs_dates.dtype],
    #     dask_gufunc_kwargs={"allow_rechunk": True},
    # )


    # # create the dataset to write 

    # out_ds = xr.Dataset(
    # {
    #     "ndvi_processed": ndvi_processed,
    #     "mask_array": mask_array
    # },
    # coords={
    #     "date": ds["date"],
    #     "pixel": ds["pixel"]
    # }
    # )
    # out_ds = out_ds.chunk({"pixel": 5000, "date": -1})
    # g = out_ds.__dask_graph__()
    # print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
    # out_ds.persist()
    # dask.distributed.wait(out_ds)
    # out_ds.compute()

    # # Remove any incompatible 'compressor' metadata left over from the source dataset
    # for v in list(out_ds.data_vars):
    #     out_ds[v].encoding.pop("compressor", None)
    #     # ensure chunks entry exists to avoid surprises
    #     out_ds[v].encoding.setdefault("chunks", None)

    # for c in list(out_ds.coords):
    #     out_ds[c].encoding.pop("compressor", None)
    #     out_ds[c].encoding.setdefault("chunks", None)

    # # Explicit encoding: no compressor for each data var
    # encoding = {v: {"compressor": None} for v in out_ds.data_vars}

    # OUT_PATH = "/data_3/scratch/fabian/2026-04-10_ndvi_benchamrk_historic_ndvi_parallel_2e22cb0b5c71b5f8e7ecc4b9634059926957930c.zarr"

    # if os.path.exists(OUT_PATH):
    #     shutil.rmtree(OUT_PATH)

    # # Write using zarr version 2 to avoid new v3 codec/BytesBytesCodec mismatch
    # out_ds.to_zarr(OUT_PATH, mode="w", consolidated=True, compute=True, encoding=encoding, zarr_version=3)

    # """# add the array of obs dates
    # ds2 = xr.open_zarr(OUT_PATH, chunks={"date": -1, "pixel": 5000})

    # arr_to_insert = ds["obs_date"].values

    # obs_da = xr.DataArray(
    #     arr_to_insert,
    #     dims=("date",),
    #     coords={"date": ds2["date"]},
    #     name="obs_date"
    # )

    # # add to dataset
    # ds2["obs_date"] = obs_da

    # # write back in r+ mode (modify existing store)
    # ds2.to_zarr(OUT_PATH,
    #             mode="a",
    #             consolidated=True)"""

    # print("done")
    # client.close