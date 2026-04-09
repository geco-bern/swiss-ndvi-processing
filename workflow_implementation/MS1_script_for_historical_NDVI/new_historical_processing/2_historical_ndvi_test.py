from datetime import datetime, date, timedelta
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
from math import ceil

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

    N_WORKERS = 120
    # TODO: test later 120 workers, with each 30GB memory_limit, BATCH_SIZE = 200K, INNER_PIXEL_CHUNK=83
    # TODO: test later 150 workers, with each 24GB memory_limit, BATCH_SIZE = 150K, INNER_PIXEL_CHUNK=1K

    with Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit='30GB',
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':1235') as client:
    
        print(client, flush = True)
        print(client.dashboard_link, flush = True)
        print(dask.config.get("scheduler"), flush = True)

        INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
        INPUT_ZARR_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
        OUT_PATH = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr"

        # FROM 4_merge_zarr.py
        # =====================================================
        #  Load -------- and new observation data sets
        # =====================================================
        DATE_CHUNKS = 365
        PIXEL_CHUNKS = 40_000
        DATE_CHUNKS_OUT = 365

        # --- load historic dataset ------------------------------------
        # --- ONLY DONE IN CONTINUOUS INTEGRATION ---

        # --- load new data dataset ------------------------------------
        new_observations_ds = xr.open_dataset(INPUT_ZARR, chunks={}, mask_and_scale= False,
                                              consolidated=True  # use consolidated on open to avoid "OSError: [Errno 24] Too many open files: '/proc/1742817/stat'"
                                              ).drop_vars("ndsi")
        # NOTE: and directly drop unused ndsi
        
        # --- load median values for each doy --------------------------
        lookuptable  = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE, chunks={}, consolidated=True)

        print(new_observations_ds, flush=True)
        print(lookuptable, flush=True)

        ##TODO: remove this when development
        ## subset pixels for development: FOR DEVELOPMENT:
        ## new_observations_ds = new_observations_ds.isel(pixel=slice(0,int(600e3))) # , datetime = slice(0,30)
        ## with 10 pixels:         runtime=55s,  storage=304KB
        ## with 100 pixels:        runtime=54s,  storage=644KB
        ## with 1_000 pixels:      runtime=61s, storage=4.1MB
        ## with 10_000 pixels:     runtime=141s, storage=39MB
        ## with 100_000 pixels:    runtime=1080s, storage=XXKB
        ## with 120_000 pixels:    runtime=565s, storage=463MB
        ## with 130_000 pixels:    runtime=720s, storage=502MB
        ## with 150_000 pixels:    runtime=640s, storage=XXXMB
        ## with 160_000 pixels:    runtime=865s, storage=617MB
        ## with 600_000 pixels:    runtime=3090s, storage=3.3GB
        ## with 1_000_000 pixels:  runtime=6300s, storage=3.8GB
        ## with 5_000_000 pixels:  runtime=XXXmin, storage=XXXGB
        ## with all pixels:        runtime=XXXmin, storage=380GB
        ## END TODO

        # =====================================================
        #  Aggregate multiple daily observation
        #  and resample to daily intervals (between observations)
        # =====================================================

        # NOTE: lowest-hanging performance fruit: replace dask-based aggregation 
        #       below (that leads the very large dask graphs) with something 
        #       where we manually loop through observation dates:
        #           - generate a new empty data set: ndvi_daily_between_obs
        #           - loop manually through days with observations and fill in values
        #             - if multiple obs times per day: select either first value
        #             - if multiple obs times per day: or compute mean value
        #       For a 200k-pixel-batch, the current aggregation already uses 10/15 mins
        #       compute time.

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

        # FOR DEVELOPMENT: observation_dates[1] # 2025-12-09
        # FOR DEVELOPMENT: observation_dates[2] # 2025-12-09
        # FOR DEVELOPMENT: plot_da_map(ndvi_daily_since_last_historic["ndvi"].sel(date= observation_dates[1]),
        # FOR DEVELOPMENT:             reduction_factor = 5, png_fname = f"NDVI_2025-12-09_combined_{agg}.png")
        
        # Print multiple status messages for log
        print(f"Initialized n={len(daily_dates_since_last_historic)} daily dates:", flush=True)
        # group observation times (as strings) by date
        times = pd.Series(observation_datetimes.strftime("%H:%M:%S"),  index=observation_datetimes.floor("D"))
        grouped = times.groupby(level=0).agg(lambda s: ",".join(s))

        # build DataFrame: 'daily', 'obs_date' (date or NaT), 'obs_times' (comma-joined times or NaN)
        status_df = pd.DataFrame({"daily": daily_dates_since_last_historic})
        status_df["obs_date"] = status_df["daily"].where(status_df["daily"].isin(grouped.index))
        status_df["obs_times"] = status_df["daily"].map(grouped).fillna("")
        with pd.option_context("display.max_rows", 4000):
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
        #  Write daily dataset (containing NaN until filled)
        #  for later i.   gapfilling, 
        #            ii.  outlier detection, and 
        #            iii. appending to historic
        # =====================================================
        new_ds = ndvi_daily_since_last_historic # NOTE: delay rechunking just before apply_ufunc(): (.chunk({"pixel": PIXEL_CHUNKS, "date": -1}))
        # NOTE: here is end of 4_merge_zarr.py in the continuous case

        # NOTE: here is the start of 5_analyse_demo_efficient.py in the continuous case
        print("First dates in newly downloaded:\n  "+"\n  ".join(np.datetime_as_string(new_ds.date.isel(date = slice(0,10)), unit='D')), flush=True)
        print("Last dates in newly downloaded:\n  "+"\n  ".join(np.datetime_as_string(new_ds.date.isel(date = slice(-10,None)), unit='D')), flush=True)

        print("Newly downloaded dataset:", flush = True)
        print(new_ds, flush = True)
        
        # --- add median NDVI from model ----------------------------------
        doy_noLeap = xr.where(new_ds.doy == 366, 365, new_ds.doy) # remove leap year if encountered
        new_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
                doy=doy_noLeap,
                pixel=new_ds.pixel) # this is to join by pixels and doy

        # NOTE: in historic case 'mask_array' is not absolutely needed. We do it for similarity to continuous case.
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
        
        new_ds = new_ds.rename({'ndvi':'ndvi_processed'})
        
        print("Newly derived dataset:", flush = True)
        print(new_ds, flush = True)

        
        # --- apply gapfilling and outlier detection function: historical_ndvi() ----------------------------------

        # delete previous output if existing
        if os.path.exists(OUT_PATH):
            shutil.rmtree(OUT_PATH)

        # Attempt at doing this not for the whole data set but in batches of 1_000_000 pixels:
        # BATCH_SIZE = 1_000_000  # pixels per outer loop iteration
        # INNER_PIXEL_CHUNK = int(16_667)  # ~1M / 60 workers → 1 task per worker per round
        BATCH_SIZE = int(5*PIXEL_CHUNKS)  # pixels per outer loop iteration
        # INNER_PIXEL_CHUNK = ceil(BATCH_SIZE / N_WORKERS)  # ~60K / 60 workers → 1 task per worker per round
                                              # Set INNER_PIXEL_CHUNK ≈ BATCH_SIZE / N_WORKERS 
                                              # (e.g. 60_000 / 60 ≈ 1_000) so each worker 
                                              # gets close to 1 task per round → full utilization 
                                              # with minimal scheduling overhead.
        INNER_PIXEL_CHUNK = ceil(BATCH_SIZE / N_WORKERS / 20)  # ~60K / 60 workers / 4 → 4 task per worker per round
            # Splitting this into 40_000 batch size, with 8 workers / 4 results in 1250 pixels. 
            # Such a 1250 chunk takes about 3 mins to be computed for 3195 days. I.e. about 12 mins for one BATCH_SIZE of 40_000.
            # When storing, this amounts to 154MB and goes very fast. => We can probably increase batch size x5 to 200_000 without issue.

            # Splitting this into 200_000 batch size, with 120 workers / 2 results in 830 pixels. 
            # Such a 830 chunk takes about 300 seconds to be computed for 3195 days. I.e. about 10 mins (actually 16 mins) for one BATCH_SIZE of 200_000.
            # When storing, a 200_000 BATCH_SIZE amounts to 771MB and goes very fast.

            # Splitting this into 200_000 batch size, with 120 workers / 20 results in 83 pixels. 
            # Such a 83 chunk takes about 30 seconds to be computed for 3195 days. I.e. about 10 mins (actually 14 mins) for one BATCH_SIZE of 200_000.
            # When storing, a 200_000 BATCH_SIZE amounts to 771MB and goes very fast.


        n_pixels = len(new_ds.pixel)
        n_batches = (n_pixels + BATCH_SIZE - 1) // BATCH_SIZE

        # materialize coordinate arrays once (small, 1D)
        dates_array_arg  = new_ds["date"].values    # NumPy, not Dask
                                                    # Pass dates_array as NumPy, not a Dask array — 
                                                    # a persisted Dask coordinate passed as a kwarg 
                                                    # to apply_ufunc adds unnecessary graph edges. After 
                                                    # new_ds["date"].values it is small (3195 elements, <25 KB).
        start_date_arg   = dates_array_arg[0]
        t0 = datetime.now()
        for batch_idx in range(n_batches):
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  " + 
                f"Starting Batch {batch_idx+1}/{n_batches}",
                flush=True
            )
            pix_start = batch_idx * BATCH_SIZE
            pix_end   = min(pix_start + BATCH_SIZE, n_pixels)
            # slice one batch and rechunk for apply_ufunc: date must be one core chunk
            batch_ds = (
                new_ds
                .isel(pixel=slice(pix_start, pix_end))
                .chunk({"pixel": INNER_PIXEL_CHUNK, "date": -1})  # no allow_rechunk needed
            )
            # FOR DEVELOPMENT:
            #g = batch_ds.__dask_graph__()
            #print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)

            # Manifest batch_ds before applying apply_ufunc()) (either by store+reload, or by persist+wait)
            # print(type(batch_ds['ndvi_processed'].data), getattr(batch_ds['ndvi_processed'].data, "chunks", None))

            print(f"Start writing temporary: [{datetime.now():%Y-%m-%d %H:%M:%S}]",flush=True)
            g = batch_ds.__dask_graph__()
            print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
            ##  Save for intermediate computation (disk-backed rechunking)
            ##  NOTE: this requires 400GB of free, additional disk space (for all images from 2017-04-01 to 2025-12-31)
            ## variant 1) save and reload()
                # OUT_ZARR_TMP = OUT_PATH+"temporary.zarr"
                # batch_ds = batch_ds.chunk({"pixel": PIXEL_CHUNKS, "date": -1})
                # batch_ds.to_zarr(OUT_ZARR_TMP, mode="w", zarr_format=3)
                # Reload freshly:
                # batch_ds = xr.open_dataset(OUT_ZARR_TMP, chunks={}, mask_and_scale= False,
                #                         consolidated=True)  # use consolidated on open to avoid "OSError: [Errno 24] Too many open files: '/proc/1742817/stat'"
            ## variant 2) persist() (and wait)
            batch_ds = batch_ds.persist()     # Do persist instead of write and reload. [persist() is like compute() but does not collect]
            dask.distributed.wait(batch_ds) # While persist() goes on in the background, wait() stops the script until persist() is done.
            
            print(f"End writing temporary: [{datetime.now():%Y-%m-%d %H:%M:%S}]",flush=True)
            g = batch_ds.__dask_graph__()
            print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
            # This now should show a simplified dask graph, compared to before the persist() operation.

            # --- visual check of resulting batch_ds ----------------------------------
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(7.2, 4), dpi = 200)

            # batch_ds_subset = batch_ds.isel(pixel=[0,1,2, 2100, 3500, 4900])
            # batch_ds_subset["median_ndvi"].plot.line(x='date',hue='pixel')

            # indexer = (batch_ds_subset["mask_array"] == 2).compute()
            # batch_ds_subset2 = batch_ds_subset.where(indexer, drop=True)
            # batch_ds_subset2["ndvi_processed"].plot.scatter(x='date',hue='pixel',marker="x")
            
            # plt.savefig('test.png')

            # call gufunc where core dim is "time" (1D arrays per pixel)
            # NOTE: 2nd lowest-hanging performance fruit: compile historical_ndvi with numba,
            #       see: https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html
            ndvi_out, mask_out = xr.apply_ufunc(
                historical_ndvi,
                batch_ds["ndvi_processed"],
                batch_ds["median_ndvi"],
                batch_ds["mask_array"],
                batch_ds["obs_date"],
                input_core_dims=[["date"], ["date"], ["date"], ["date"]],   # These are dimensions that should not be broadcast
                output_core_dims=[["date"], ["date"]],
                kwargs={"dates_array": dates_array_arg,   # pass as NumPy, not Dask
                        "starting_date": start_date_arg},
                vectorize=True,        # This vectorizes `historical_ndvi` automatically with `numpy.vectorize`. NOTE: convenient but slow, see: 
                dask="parallelized",
                output_dtypes=[np.dtype('int16'), np.dtype('int8')],
                # allow_rechunk no longer needed: batch is already chunked correctly above
            )
            # g = mask_processed.__dask_graph__()
            g = ndvi_out.__dask_graph__()
            print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)

            # compute this batch with all workers (and collect on driver node for single save by driver node)
            ndvi_out, mask_out = dask.compute(ndvi_out, mask_out)
            # compute() collects them to the scheduler node (driver) and does driver-side write (batch needs to fit into memory)
            # alternatively persist()+wait() would force computation without collection, leaving results on the worker nodes,
            #     and doing worker-side write

            # create the dataset to write for this batch
            out_ds = xr.Dataset(
                {
                    "ndvi_processed": ndvi_out, #(["pixel", "date"], ndvi_out.data),
                    "mask_array":     mask_out  #(["pixel", "date"], mask_out.data)
                },
                # coords={c: batch_ds[c] for c in batch_ds.coords}
            )
            out_ds = out_ds.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS_OUT}) # Because of this BATCH_SIZE must be an integer multiple of PIXEL_CHUNKS

            #out_ds.attrs["pixel_definition"] = new_ds.attrs["pixel_definition"]
            out_ds.attrs = new_ds.attrs
            out_ds.attrs.pop("description_ndsi", None) # since we dropped ndsi, we also drop this attr

            print(f"Partial writing to new file: {OUT_PATH}", flush=True)
            # Explicit encoding: simple compressor for each data var
            encoding = {v: {"compressors": COMPRESSOR} for v in out_ds.data_vars}

            # drop any coord/data var chunk encodings that conflict
            for name in list(out_ds.coords) + list(out_ds.data_vars): # TODO: remove this again if possible
                out_ds[name].encoding.pop("chunks", None)                           # TODO: remove this again if possible
                out_ds[name].encoding.pop("compressor", None)                       # TODO: remove this again if possible
                out_ds[name].encoding.pop("compressors", None)                      # TODO: remove this again if possible
                
            # write: create on first batch, append on subsequent
            # write is done only by driver node (due to the compute() above), not by any worker node
            if batch_idx == 0:
                out_ds.to_zarr(OUT_PATH, mode="w", encoding=encoding, zarr_format=3)
            else:
                out_ds.to_zarr(OUT_PATH, append_dim="pixel")
                # append_dim="pixel" requires that all batches have the same 
                # date dimension. Since all batches come from the same new_ds, 
                # this is guaranteed.

                # The final write `.to_zarr()` is incremental — you don't need to 
                # hold all 100M pixel results in memory at once, only 200K from the BATCH_SIZE.
                # This was the main memory bottleneck of the previous approach.

            # progress log
            elapsed  = (datetime.now() - t0).total_seconds()
            done_pix = pix_end
            eta_s    = elapsed / done_pix * (n_pixels - done_pix) if done_pix < n_pixels else 0
            eta_datetime = datetime.now() + timedelta(seconds=eta_s)
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  "
                f"Batch {batch_idx+1}/{n_batches}  "
                f"pixels {pix_start:,}–{pix_end:,}  "
                f"elapsed {elapsed/60:.1f}min  ETA {eta_s/60:.0f}min ({eta_datetime})",
                flush=True
            )

        # small pause to allow dashboard websocket handshakes / metadata flush
        time.sleep(10)


    print(OUT_PATH, flush = True)

    # cleanup the temporary file:
    # if os.path.exists(OUT_ZARR_TMP):
    #     shutil.rmtree(OUT_ZARR_TMP)
        
    sys.exit(0)



# from GECO-Workstation-02:
# rsync -ahz --info=progress2 -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr /data_3/scratch
# rsync -ahz --info=progress2 -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr /data_3/scratch; mv /data_3/scratch/historical_2026-04-04_18h16_historical_v7.zarr /data_3/scratch/historical_2026-04-04_18h16_historical_v7_2026-04-09.zarr