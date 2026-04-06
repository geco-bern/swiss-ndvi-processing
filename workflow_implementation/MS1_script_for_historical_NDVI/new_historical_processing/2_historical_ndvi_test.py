from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import dask
import xarray as xr
import os, sys
import shutil
import pandas as pd
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec
import time

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)


NO_COVERAGE = 32767
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step
INVALID     = -32768
INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows

COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)

# FOR DEVELOPMENT:
# ndvi_array = ndvi_array_arg.isel(pixel=1).values
# median_array = median_array_arg.isel(pixel=1).values
# mask_array = mask_array_arg.isel(pixel=1).values
# is_observation_date =  is_obs_date_array_arg.values
# dates_array = dates_array_arg.values
# starting_date = dates_array_arg.values[0]
def historical_ndvi(ndvi_array, median_array, mask_array, is_observation_date, dates_array, starting_date):
    """
    # ndvi_array          # this is the observed/gapfilled/processed NDVI value
    # median_array        # this is the modelled median NDVI for the corresponding DOY
    # mask_array          # this is the integer processing status
    # is_observation_date # this is the True-False boolean if a date contains satellite images (is_observation_date?)
    # kwargs={"dates_array":   dates_array,  # this contains all daily dates
    #         "starting_date": start_date}   # this contains the starting date when to start ??
    """

    is_historic_case = True # this can be set to False for continuous case

    if not len(ndvi_array.shape)          == 1: raise Exception( "Expected 1D array as ndvi_array" )
    if not len(median_array.shape)        == 1: raise Exception( "Expected 1D array as median_array" )
    if not len(mask_array.shape)          == 1: raise Exception( "Expected 1D array as mask_array" )
    if not len(is_observation_date.shape) == 1: raise Exception( "Expected 1D array as is_observation_date" )
    

    # Ensure mask_array is writable
    mask_array = np.array(mask_array, copy=True) # TODO: check if this is needed

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


if __name__ == "__main__":

    N_WORKERS = 60

    with Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit='50GB',
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':1239') as client:
    
        print(client.dashboard_link)

        # already having medians computed

        #INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small.zarr" 
        INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
        INPUT_ZARR_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
        OUT_PATH = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr"

        # FROM 4_merge_zarr.py
        # =====================================================
        #  Load -------- and new observation data sets
        # =====================================================
        DATE_CHUNKS = 365
        PIXEL_CHUNKS = 40000
        DATE_CHUNKS_OUT = 365

        # --- load historic dataset ------------------------------------
        # --- ONLY IN CONTINUOUS ---

        # --- load new data dataset ------------------------------------
        new_observations_ds = xr.open_dataset(INPUT_ZARR, chunks={}, mask_and_scale= False).drop_vars("ndsi")
        # NOTE: delay the datetime chunking , "datetime": -1
        # NOTE: and directly drop unused ndsi
        
        # --- load median values for each doy --------------------------
        lookuptable  = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE).chunk({"doy": -1, "pixel": PIXEL_CHUNKS})

        #TODO: remove this when development
        # subset pixels for development: FOR DEVELOPMENT:
        new_observations_ds = new_observations_ds.isel(pixel=slice(0,10**3)) # , datetime = slice(0,30)
        # with 10 pixels:       runtime=55s,  storage=304KB
        # with 100 pixels:      runtime=78s,  storage=644KB
        # with 1000 pixels:     runtime=327s, storage=4.1MB
        # with 10000 pixels:    runtime=1200s, storage=39MB
        # with 100000 pixels:   runtime=XXs, storage=XXKB
        # with 1000000 pixels:  runtime=282min, storage=3.8GB
        # wit all pixels:       runtime=XXXmin, storage=XXXGB
        # END TODO

        # =====================================================
        #  Aggregate multiple daily observation
        #  and resample to daily intervals (between observations)
        # =====================================================

        # Decide how to collapse sub-daily duplicates to one observed value per day
        agg = 'first' # # TODO: choose 'mean' or 'first'
        if agg == 'first':
            ndvi_daily_between_obs = (new_observations_ds
                # NOTE: by filtering out NO_COVERAGE an INVALID they both become NaN
                #       and they are later both replace by only one of them NO_COVERAGE
                #       effectively this removes INVALID pixels TODO: is this desired behavior?
                .where((new_observations_ds['ndvi']  != NO_COVERAGE) &
                        (new_observations_ds['ndvi'] != INVALID))
                .groupby(datetime=xr.groupers.TimeResampler('1D'))
                .first()
                .fillna(NO_COVERAGE).astype(np.int16)
                .rename({'datetime': 'date'})
            )
        elif agg == 'mean':
            ndvi_daily_between_obs = (new_observations_ds
                # NOTE: by filtering out NO_COVERAGE an INVALID they both become NaN
                #       and they are later both replace by only one of them NO_COVERAGE
                #       effectively this removes INVALID pixels TODO: is this desired behavior?
                .where((new_observations_ds['ndvi'] != NO_COVERAGE) &
                        (new_observations_ds['ndvi'] != INVALID))
                .astype(np.float32)
                .groupby(datetime=xr.groupers.TimeResampler('1D'))
                .mean(skipna=True)
                .fillna(NO_COVERAGE).astype(np.int16)
                .rename({'datetime': 'date'})
            )
        else:
            raise ValueError(f"Unsupported agg={agg}")

        # keep track which dates were actually observation dates
        observation_datetimes = pd.DatetimeIndex(new_observations_ds["datetime"].values)
        observation_dates     = pd.DatetimeIndex(observation_datetimes).floor("D")

        # =====================================================
        #  Initialize empty daily dataset
        # =====================================================
        # note: we call this dataset since_last_historic in the continuous update.
        #       Here this is means simply since the first observation:
        start_date         = observation_dates.min()
        end_date           = observation_dates.max()

        # build full daily index from start_date to end_date (make sure start_date/end_date are pd-compatible)
        daily_dates_since_last_historic = pd.date_range(
            start=pd.to_datetime(start_date).floor("D"),
            end=pd.to_datetime(end_date).floor("D"),
            freq="D")

        # reindex coords to guarantee daily coverage starts at start_date
        # i.e. extending back to last historic date:
        ndvi_daily_since_last_historic = (ndvi_daily_between_obs
            .reindex(date=daily_dates_since_last_historic, 
                    method=None) # None (default): don’t fill gaps;
                                # fills missing days with NaN; fill later if desired
            .fillna(NO_COVERAGE).astype(np.int16)
        )
            
        # ndvi_daily_between_obs.date.values
        # ndvi_daily_since_last_historic.date.values

        # FOR DEVELOPMENT: observation_dates[1] # 2025-12-09
        # FOR DEVELOPMENT: observation_dates[2] # 2025-12-09
        # FOR DEVELOPMENT: plot_da_map(ndvi_daily_since_last_historic["ndvi"].sel(date= observation_dates[1]),
        # FOR DEVELOPMENT:             reduction_factor = 5, png_fname = f"NDVI_2025-12-09_combined_{agg}.png")
        
        # Print status
        print(
            f"Initialized n={len(daily_dates_since_last_historic)} daily dates:",
            #f"\nfrom {daily_dates_since_last_historic.min().date()}"+
            #f" to {daily_dates_since_last_historic.max().date()}"+
            # f"\nwith observations on days at:"+
            # f"\n"+"\n".join([f"  {d.strftime('%Y-%m-%d')}: {dt.strftime('%Y-%m-%d_%Hh%M')}" 
            #    for (d, dt) in zip(observation_dates, observation_datetimes)]),
            flush=True,
        )
        # group observation times (as strings) by date
        times = pd.Series(observation_datetimes.strftime("%H:%M:%S"), 
                        index=observation_datetimes.floor("D"))
        grouped = times.groupby(level=0).agg(lambda s: ",".join(s))

        # build DataFrame: 'daily', 'obs_date' (date or NaT), 'obs_times' (comma-joined times or NaN)
        status_df = pd.DataFrame({"daily": daily_dates_since_last_historic})
        status_df["obs_date"] = status_df["daily"].where(status_df["daily"].isin(grouped.index))
        status_df["obs_times"] = status_df["daily"].map(grouped).fillna("")
        print(status_df, flush=True)
        print(f"In total: {len(status_df["daily"])} days, {sum(status_df["obs_date"].notnull())} obs_dates, {len(times)} obs_times. ")

        # Append day-of-year (for merging of median expected NDVI from model)
        doy_array = daily_dates_since_last_historic.dayofyear.values
        ndvi_daily_since_last_historic = ndvi_daily_since_last_historic.assign_coords(
            doy   = ('date', doy_array.astype(np.int32))
        )
        
        # Keep track which dates were actually observation dates:
        # add a DataArray to Dataset, which specifies the dates that were observations
        ndvi_daily_since_last_historic["obs_date"] = ndvi_daily_since_last_historic.date.isin(observation_dates)

        # =====================================================
        #  Write daily dataset (containing NaN)
        #  for later i.   gapfilling, 
        #            ii.  outlier detection, and 
        #            iii. appending to historic
        # =====================================================
        new_ds = ndvi_daily_since_last_historic # NOTE: delay rechunking just before apply_ufunc(): (.chunk({"pixel": PIXEL_CHUNKS, "date": -1}))
        # new_ds has: 
        #   coords: x,y,x_idx,y_idx, pixel, date; 
        #   vars:   ndvi,obs_date
        #   attrs:  pixel_definition,transform_note,transform_coeffs,transform_instr,description_ndvi,description_ndsi,nodata,cloud_shadow
        
        # drop any coord/data var chunk encodings that conflict
        # for name in list(new_ds.coords) + list(new_ds.data_vars):
        #     new_ds[name].encoding.pop("chunks", None)
        #     new_ds[name].encoding.pop("compressor", None)
        #     new_ds[name].encoding.pop("compressors", None)

        # write out    
        # new_ds.to_zarr(OUT_ZARR_TMP, mode="w", zarr_format=3)
        # NOTE: here is end of 4_merge_zarr.py in the continuous case

        # NOTE: here is the start of 5_analyse_demo_efficient.py in the continuous case
        # def show_ds_structure(ds):
        #     for c in list(ds.coords) + list(ds.data_vars):
        #         print(str(c).ljust(15) + ":   " + str(ds[c].encoding))
        #show_ds_structure(new_ds)
        #show_ds_structure(lookuptable)

        print("First dates in newly downloaded:\n  "+"\n  ".join(np.datetime_as_string(new_ds.date.isel(date = slice(0,10)), unit='D')), flush=True)
        print("Last dates in newly downloaded:\n  "+"\n  ".join(np.datetime_as_string(new_ds.date.isel(date = slice(-10,None)), unit='D')), flush=True)
        # NOTE: here is end of 5_analyse_demo_efficient.py in the continuous case

        print("Newly downloaded dataset:", flush = True)
        print(new_ds, flush = True)
        
        # --- add median NDVI from model ----------------------------------
        doy_noLeap = xr.where(new_ds.doy == 366, 365, new_ds.doy) # remove leap year if encountered
        new_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
                doy=doy_noLeap,
                pixel=new_ds.pixel) # this is to join by pixels and doy


        # Add mask_array to new_ds (filled with 0 or 2 at this point):
            # mask_array == 0: the data is not an observation and is yet to be smoothed
            # mask_array == 1: the data is not an observation and is smoothed
            # mask_array == 2: the data is an observation and is yet to be smoothed
            # mask_array == 3: the data is an observation and is smoothed
            # mask_array == 4: the data is an observation and is an outlier
        mask_2or0 = (
            (new_ds["obs_date"]) & 
            (new_ds["ndvi"] < NO_COVERAGE) & 
            (new_ds["ndvi"] > INVALID))
        new_ds['mask_array'] = xr.where(mask_2or0, np.int8(2), np.int8(0))
        
        new_ds = new_ds.rename(
            {'ndvi':'ndvi_processed'})

        # Save for intermediate computation
        OUT_ZARR_TMP = OUT_PATH+"temporary.zarr"
        new_ds.chunk({"pixel": PIXEL_CHUNKS, "date": -1}).to_zarr(OUT_ZARR_TMP, mode="w", zarr_format=3)

        # Reload freshly:
        new_ds = xr.open_dataset(OUT_ZARR_TMP, chunks={}, mask_and_scale= False)
        

        # --- visual check of resulting new_ds ----------------------------------
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(7.2, 4), dpi = 200)

        # new_ds_subset = new_ds.isel(pixel=[0,1,2, 2100, 3500, 4900])
        # new_ds_subset["median_ndvi"].plot.line(x='date',hue='pixel')

        # indexer = (new_ds_subset["mask_array"] == 2).compute()
        # new_ds_subset2 = new_ds_subset.where(indexer, drop=True)
        # new_ds_subset2["ndvi_processed"].plot.scatter(x='date',hue='pixel',marker="x")
        
        # plt.savefig('test.png')
        
        # --- apply gapfilling and outlier detection function: historical_ndvi() ----------------------------------

        # prepare arguments spanning historic and new data: all lazy
        ndvi_array_arg   = new_ds["ndvi_processed"].chunk(dict(date=-1)).persist() # NOTE: in continuous integration this is the merged_ds
        median_array_arg = new_ds["median_ndvi"].chunk(dict(date=-1)).persist()    # NOTE: in continuous integration this is the merged_ds
        mask_array_arg   = new_ds["mask_array"].chunk(dict(date=-1)).persist()     # NOTE: in continuous integration this is the merged_ds
        is_obs_date_array_arg = new_ds["obs_date"].chunk(dict(date=-1)).persist()  # NOTE: in continuous integration this is the merged_ds
        dates_array_arg  = new_ds["date"].chunk(dict(date=-1)).persist()           # NOTE: in continuous integration this is the merged_ds
        start_date_arg   = dates_array_arg.values[0] # NOTE: in the historic case
        # using persist() reduces graph size

        # reduce graph size by using futures
        # dates_future  = client.scatter(dates_array)
        # ndvi_future   = client.scatter(ndvi_array)
        # median_future = client.scatter(median_array)
        # dates_future  = client.scatter(dates_array)
        # mask_future   = client.scatter(mask_array)
        # then reference *_future inside tasks/closures instead of passing *_array
        # visualize(dates_future)

        # reduce graph size by handing NumPy arrays to dask:
        # dates_daskarray = da.from_array(dates_array)   # Hand NumPy array to Dask

        # call gufunc where core dim is "time" (1D arrays per pixel)
        ndvi_processed, mask_processed = xr.apply_ufunc(
            historical_ndvi,
            ndvi_array_arg,        # this is the observed/gapfilled/processed NDVI value
            median_array_arg,      # this is the modelled median NDVI for the corresponding DOY
            mask_array_arg,        # this is the integer processing status
            is_obs_date_array_arg, # this is the True-False boolean if a date contains satellite images (is_observation_date?)
            input_core_dims=[["date"], ["date"],["date"], ["date"]],    # each call gets 1D time arrays
            output_core_dims=[["date"],["date"]],
            kwargs={
                "dates_array": dates_array_arg,           # this contains all daily dates
                "starting_date": start_date_arg},   # this contains the starting date when to start ??
            vectorize=True, 
            dask="parallelized",
            output_dtypes=[np.dtype('int16'), np.dtype('int8')],
            # dask_gufunc_kwargs={"allow_rechunk": True},
        )
        # ndvi_processed.isel(pixel=1, date=slice(3160,3170)).compute() # TODO: check why this is [ 4845,  4835,  4826,  4819, 32767, 32767, 32767, 32767, 32767, 32767]

        # FROM THE PREVIOUS HISTORIC VERSION: # call ufunc where core dim is "time" (1D arrays per pixel)
        # FROM THE PREVIOUS HISTORIC VERSION: ndvi_processed, mask_array = xr.apply_ufunc(
        # FROM THE PREVIOUS HISTORIC VERSION:     historical_ndvi,
        # FROM THE PREVIOUS HISTORIC VERSION:     ndvi_avg,
        # FROM THE PREVIOUS HISTORIC VERSION:     medians,
        # FROM THE PREVIOUS HISTORIC VERSION:     input_core_dims=[["datetime"],["date"]],    # each call gets 1D time arrays
        # FROM THE PREVIOUS HISTORIC VERSION:     output_core_dims=[["date"],["date"]],
        # FROM THE PREVIOUS HISTORIC VERSION:     vectorize=True, 
        # FROM THE PREVIOUS HISTORIC VERSION:     dask="parallelized",
        # FROM THE PREVIOUS HISTORIC VERSION:     kwargs={"obs_date" : obs_date},
        # FROM THE PREVIOUS HISTORIC VERSION:     output_dtypes=[np.int16, np.int8],
        # FROM THE PREVIOUS HISTORIC VERSION:     dask_gufunc_kwargs={"allow_rechunk": True},
        # FROM THE PREVIOUS HISTORIC VERSION: )
        
        # Ensure both outputs are computed in ONE scheduler pass (avoids re-running apply_ufunc twice)
        ndvi_processed, mask_processed = dask.persist(ndvi_processed, mask_processed)
        dask.distributed.wait([ndvi_processed, mask_processed])

        # g = mask_processed.__dask_graph__()
        g = ndvi_processed.__dask_graph__()
        print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
        #                    586_503 pixels:                 | 16_041_205 pixels:              | 105_715_396 pixels:
        # without persist(): 49    layers, and 196760 tasks  | 49    layers, and 1289428 tasks | xxx layers, and xxx tasks
        # with persist():    16-17 layers, and  31953 tasks  | 16-17 layers, and  872665 tasks | xxx layers, and xxx tasks
        # without persist(): .............. and 10.58 MiB
        # with persist():    size 23.08 MiB and 10.58 MiB
        
        # visualize(ndvi_processed)

        # create the dataset to write 
        out_ds = xr.Dataset(
            {
                "ndvi_processed": ndvi_processed,
                "mask_array": mask_processed
            }
        )
        
        #out_ds.attrs["pixel_definition"] = new_ds.attrs["pixel_definition"]
        out_ds.attrs = new_ds.attrs
        out_ds.attrs.pop("description_ndsi", None) # since we dropped ndsi, we also drop this attr


        if os.path.exists(OUT_PATH):
            shutil.rmtree(OUT_PATH)

        print(f"writing to new file: {OUT_PATH}", flush=True)
        out_ds = (
            out_ds
            .sortby("date")
            .chunk({"pixel": PIXEL_CHUNKS, 
                    "date": DATE_CHUNKS_OUT})
        )
        
        # Explicit encoding: simple compressor for each data var
        # encoding = {v: {"compressors": None      } for v in out_ds.data_vars} # TODO: why not? this should be following what was done to create v4 of historic
        encoding = {v: {"compressors": COMPRESSOR} for v in out_ds.data_vars}

        # drop any coord/data var chunk encodings that conflict   # TODO: is this needed?
        for name in list(out_ds.coords) + list(out_ds.data_vars): # TODO: remove this again if possilbe
            out_ds[name].encoding.pop("chunks", None)                           # TODO: remove this again if possilbe
            out_ds[name].encoding.pop("compressor", None)                       # TODO: remove this again if possilbe
            out_ds[name].encoding.pop("compressors", None)                      # TODO: remove this again if possilbe

        # overwrite (mode="w")
        out_ds.to_zarr(
            OUT_PATH, 
            mode="w", 
            compute=True,
            encoding=encoding, 
            zarr_format=3
        )
        
    print(OUT_PATH, flush = True)
    sys.exit(0)