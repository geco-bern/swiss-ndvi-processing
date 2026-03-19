import datetime as dt
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
from dask import visualize
import dask.array as da
import xarray as xr
import argparse
import os, shutil, sys
import time
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec

INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr"

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

# HOW TO RUN FROM BASH:
# source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
# SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient.py"
# LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"
# NEW_NDVI="/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_10kmX10km_4th.zarr"
# HISTO_INPUT="/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_10kmX10km.zarr"
# HISTO_OUTPUT="/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_10kmX10km_extended2.zarr"
# python -u $SCRIPT_FILE $NEW_NDVI $HISTO_INPUT --histo-output=$HISTO_OUTPUT > $LOG_FILE  2>&1 &

def historical_ndvi(ndvi_arr_original, medians, mask_array_original, is_observation_date, dates, starting_date):
        
        start_idx = np.searchsorted(dates, starting_date) 
        obs_prior = np.nonzero(is_observation_date[:start_idx])

        # Ensure mask_array is writable
        mask_array_original = np.array(mask_array_original, copy=True)


        if len(obs_prior) < 3:
            return ndvi_arr_original, mask_array_original

        crop_start = obs_prior[-3]  # Start at 3th prior obs, use to smooth
        ndvi_arr = ndvi_arr_original[crop_start:]
        medians = medians[crop_start:]
        is_observation_date = is_observation_date[crop_start:]
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

        obs_mask = (ndvi_arr > 0) & (ndvi_arr < 1) & is_observation_date
        
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
            valid_outlier_idx = outlier_idx[is_observation_date[outlier_idx] == 1]

            mask_array[ before & obs_mask ] = 3
            mask_array[ before & (~obs_mask) ] = 1

            mask_array[valid_outlier_idx] = 4


            mask_array_final =  np.concatenate([mask_array_not_analyzed, mask_array])
            final_ndvi_value =  np.concatenate([ndvi_not_analyzed, ndvi_smoothed])

            return final_ndvi_value, mask_array_final
        
        else:

            return ndvi_arr_original, mask_array_original

if __name__ == "__main__":

    # PARSE ARGUMENTS:
    parser = argparse.ArgumentParser()

    parser.add_argument("INPUT_ZARR",        help="Full path to Zarr folder with newly downloaded NDVI data")
    parser.add_argument("HISTO_ZARR_INPUT",  help="Full path to Zarr folder with historic NDVI data")
    parser.add_argument("--histo-output", dest = "HISTO_ZARR_OUTPUT", default=None,
                        help="Full path for updated historic Zarr (if omitted, defaults to HISTO_ZARR_INPUT)"+
                             "Path must either be a non-existing folder or then HISTO_ZARR_INPUT. In latter case data is appended.")
    args = parser.parse_args()

    INPUT_ZARR        = args.INPUT_ZARR
    HISTO_ZARR_INPUT  = args.HISTO_ZARR_INPUT
    HISTO_ZARR_OUTPUT = args.HISTO_ZARR_OUTPUT or HISTO_ZARR_INPUT # if None defaults to HISTO_ZARR_INPUT

    # if running interactively use e.g.:
    #   # HISTO_ZARR_INPUT  = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m_copy.zarr"
    #   # HISTO_ZARR_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m_copy.zarr"
    #   # INPUT_ZARR        = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12_processed.zarr"


    #   # HISTO_ZARR_INPUT     = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m.zarr"
    #   # HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m_extended.zarr" # TODO: remove this and instea do it circular
    #   # INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_1000mX1000m_4th.zarr"
    #   # HISTO_ZARR_INPUT     = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_10kmX10km.zarr"
    #   # HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_10kmX10km_extended.zarr" # TODO: remove this and instea do it circular
    #   # INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_10kmX10km_4th.zarr"
    #   HISTO_ZARR_INPUT     = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_100kmX100km.zarr"
    #   HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_100kmX100km_extended.zarr" # TODO: remove this and instea do it circular
    #   INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_100kmX100km_4th.zarr"
    #   # HISTO_ZARR_INPUT     = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr.zarr"
    #   # HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_extended.zarr" # TODO: remove this and instea do it circular
    #   # INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_4th.zarr"


    # START PROCESSING:
    t0 = time.perf_counter()

    # N_WORKERS = 10           # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # N_THREADS_PER_WORKER = 1 # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1         # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000     # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '240GB'

    # N_WORKERS = 20        # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 90s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1      # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 90s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000  # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 90s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '190GB'
    # N_THREADS_PER_WORKER = 1

    N_WORKERS = 30        # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 97s (incl Zarr); 586503 pixels => 53s; 16041205 pixels => 3300s; 105715396 pixels => XXs
    DATE_CHUNKS = -1      # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 97s (incl Zarr); 586503 pixels => 53s; 16041205 pixels => 3300s; 105715396 pixels => XXs
    PIXEL_CHUNKS = 10000  # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 97s (incl Zarr); 586503 pixels => 53s; 16041205 pixels => 3300s; 105715396 pixels => XXs
    MEMORY_PER_WORKER = '120GB'
    N_THREADS_PER_WORKER = 1
    # TODO: check: 16041205 pixels in 640s in pipeline_FB_2026-03-19_09h09m26.log
    #              16041205 pixels in 3300s in pipeline_FB_2026-03-19_11h38m18.log
    #              Why so much longer? 
    #                 Is it due to the compression when writing? 
    #                 If so, then this would be smaller in case of appending.
    #                 The dashboard showed some computation to be indeed over after 10mins. Then "PerformanceWarning: Increasing number of chunks by factor of 245". And then dashboard didn't show any activity anymore.

    # N_WORKERS = 60        # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1      # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000  # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '66GB'
    # N_THREADS_PER_WORKER = 1


    # Definition of output format of new
    # TODO: when going circular this is probably not needed anymore.
    DATE_CHUNKS_OUT = 30
    COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)
    
    
    
    t0=time.perf_counter()

    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=N_THREADS_PER_WORKER,
        memory_limit=MEMORY_PER_WORKER,
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':8343')  
    print(client, flush = True)
    print(client.dashboard_link, flush = True) # use this dashboard to follow progress

    # DATE_CHUNKS  = historic_ds.chunks['date'][0]  # should be 30 days # TODO: why not this?
    # PIXEL_CHUNKS = historic_ds.chunks['pixel'][0]                     # TODO: why not this?

    historic_ds  = xr.open_zarr(HISTO_ZARR_INPUT, chunks={"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    new_ds       = xr.open_zarr(INPUT_ZARR, chunks={"pixel": PIXEL_CHUNKS, "date": -1})
    lookuptable  = xr.open_zarr(INPUT_LOOKUPTABLE, consolidated=False).chunk({"pixel": PIXEL_CHUNKS})

    def show_ds_structure(ds):
        for c in list(ds.coords) + list(ds.data_vars):
            print(str(c).ljust(15) + ":   " + str(ds[c].encoding))
    
    #show_ds_structure(historic_ds)
    #show_ds_structure(new_ds)
    #show_ds_structure(lookuptable)

    print("Last dates in historic_ds:\n  "+"\n  ".join(np.datetime_as_string(historic_ds.date.isel(date = slice(-10,None)), unit='D')), flush=True)
    print("First dates in newly downloaded:\n  "+"\n  ".join(np.datetime_as_string(new_ds.date.isel(date = slice(0,10)), unit='D')), flush=True)
    # TODO: there is an overlap, do we need to remove this for application of historical_ndvi()

    print("Current historic dataset:", flush = True)
    print(historic_ds, flush = True)
    print("Newly downloaded dataset:", flush = True)
    print(new_ds, flush = True)
    
    # --- concatenate full datasets along time ----------------------------------
    # Add median NDVI from model    
    # to new_ds:
    doy_noLeap = xr.where(new_ds.doy == 366, 365, new_ds.doy) # remove leap year if encountered
    new_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_noLeap,
            pixel=new_ds.pixel) # this is to join by pixels and doy
    # to historic_ds: # TODO: note that each time we are adding the medians to the historic data again and again. Maybe just add it once and store it?
    doy_noLeap = xr.where(historic_ds.doy == 366, 365, historic_ds.doy) # remove leap year if encountered
    historic_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_noLeap,
            pixel=historic_ds.pixel) # this is to join by pixels and doy


    # Add mask_array to new_ds (filled with 0 or 2):
        # mask_array == 0: the data is not an observation and is yet to be smoothed
        # mask_array == 1: the data is not an observation and is smoothed
        # mask_array == 2: the data is an observation and is yet to be smoothed
        # mask_array == 3: the data is an observation and is smoothed
        # mask_array == 4: the data is an observation and is an outlier
    mask_0or2_1D = xr.where(new_ds["obs_date"], 2, 0).astype(np.int8)   # dims: date
    mask_0or2_2D = mask_0or2_1D.expand_dims({"pixel": new_ds.pixel})
    new_ds = new_ds.assign(mask_array=mask_0or2_2D)

    # --- concatenate full datasets along time ----------------------------------
    new_ds = new_ds.rename(
        {'ndvi_obs':'ndvi_processed',
            'ndsi_obs':'ndsi_processed'}
    ).drop_vars('ndsi_processed')
    # Bind together with historic:
    merged_ds = (
        xr.concat(
            [historic_ds, new_ds], 
            dim="date")
        .sortby("date")
    )
    merged_ds = merged_ds.chunk(
        {"pixel": PIXEL_CHUNKS, 
            "date": DATE_CHUNKS})

    # --- apply gapfilling and outlier detection function: historical_ndvi() ----------------------------------

    # prepare arguments spanning historic and new data: all lazy
    ndvi_array   = merged_ds["ndvi_processed"].persist()
    median_array = merged_ds["median_ndvi"].persist()
    dates_array  = merged_ds["date"].persist()
    mask_array   = merged_ds["mask_array"].persist()
    obs_dates    = merged_ds["obs_date"].persist()
    # using persist() reduces graph size

    # specifying where new data starts
    start_date = historic_ds['date'].max().values

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
    output_dtypes = [ndvi_array.dtype, mask_array.dtype] # prespecify types
    ndvi_processed, mask_processed = xr.apply_ufunc(
        historical_ndvi,
        ndvi_array,        # this is the observed/gapfilled/processed NDVI value
        median_array,      # this is the modelled median NDVI for the corresponding DOY
        mask_array,        # this is the integer processing status
        obs_dates,         # this is the True-False boolean if a date contains satellite images (is_observation_date?)
        input_core_dims=[["date"], ["date"],["date"], ["date"]],    # each call gets 1D time arrays
        output_core_dims=[["date"],["date"]],
        vectorize=True, 
        dask="parallelized",
        kwargs={
             "dates": dates_array,           # this contains all daily dates
             "starting_date": start_date},   # this contains the starting date when to start ??
        output_dtypes=output_dtypes, 
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    # g = mask_processed.__dask_graph__()
    g = ndvi_processed.__dask_graph__()
    print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
    #                    586_503 pixels:                 | 16_041_205 pixels:              | 105_715_396 pixels:
    # without persist(): 49    layers, and 196760 tasks  | 49    layers, and 1289428 tasks | xxx layers, and xxx tasks
    # with persist():    16-17 layers, and  31953 tasks  | 16-17 layers, and  872665 tasks | xxx layers, and xxx tasks
    # without persist(): .............. and 10.58 MiB
    # with persist():    size 23.08 MiB and 10.58 MiB
    
    # visualize(ndvi_processed)

    # --- append the new processed data to the historic_ds ----------------------------------

    historic_ds_to_extend = (
        historic_ds
        .drop_vars('median_ndvi')        # TODO: note that each time we are adding the medians to the historic data again and again. Maybe just add it once and store it?
        # .isel(date = slice(-10, None)) # NOTE just for development
    )

    ndvi_processed_to_append = ndvi_processed.sel(date = slice(start_date + 1, None)) # Note the shift +1
    mask_processed_to_append = mask_processed.sel(date = slice(start_date + 1, None)) # Note the shift +1
    ds_to_append = (
        xr.Dataset({"ndvi_processed": ndvi_processed_to_append, 
                     "mask_array":     mask_processed_to_append})
        .chunk({"pixel": PIXEL_CHUNKS, 
                 "date": DATE_CHUNKS_OUT})
    )
    #ndvi_processed_to_append.compute() 
    #mask_processed_to_append.compute()
    #ds_to_append.compute()               # starts on 2025-12-01 # Note the shift +1
    #historic_ds_to_extend.compute()      # ends   on 2025-11-30

    # For development
    # show_ds_structure(ds_to_append)
    # show_ds_structure(extended_historic_ds)
     


    def fallback_action_overwrite_zarr():
        # concatenate to complete dataset
        extended_historic_ds = (
            xr.concat([historic_ds_to_extend, ds_to_append], dim="date")
            .sortby("date")
            .chunk({"pixel": PIXEL_CHUNKS, 
                    "date": DATE_CHUNKS_OUT})
        )
        
        # Explicit encoding: simple compressor for each data var
        # encoding = {v: {"compressors": None      } for v in extended_historic_ds.data_vars} # TODO: why not? this should be following what was done to create v4 of historic
        encoding = {v: {"compressors": COMPRESSOR} for v in extended_historic_ds.data_vars}

        # drop any coord/data var chunk encodings that conflict   # TODO: we're already doing 
        for name in list(extended_historic_ds.coords) + list(extended_historic_ds.data_vars): # TODO: remove this again if possilbe
            extended_historic_ds[name].encoding.pop("chunks", None)                           # TODO: remove this again if possilbe
            extended_historic_ds[name].encoding.pop("compressor", None)                       # TODO: remove this again if possilbe
            extended_historic_ds[name].encoding.pop("compressors", None)                      # TODO: remove this again if possilbe

        # overwrite (mode="w")
        extended_historic_ds.to_zarr(
            HISTO_ZARR_OUTPUT, 
            mode="w", 
            # consolidated=True, # gave warning "consolidated metadata is currently not part in the Zarr format 3 specification."
            compute=True,
            encoding=encoding, 
            zarr_format=3
        )

    if HISTO_ZARR_OUTPUT == HISTO_ZARR_INPUT:
        print(f"appending to file\n  {HISTO_ZARR_OUTPUT}", flush=True)
        try:
            print("Appending new dates to existing zarr store...", flush=True)
            # test_ds = xr.open_dataset(HISTO_ZARR_OUTPUT) # final check what is in there
            # test_ds.date.max() # indeed 2025-11-30
            # extended_ds = xr.concat([test_ds, ds_to_append], dim="date")
            ds_to_append.to_zarr(
                HISTO_ZARR_OUTPUT,
                mode="a",
                append_dim="date",
                compute=True,
                encoding={},  # NOTE: since we append encoding must not be provided
                zarr_format=3,
            )

            # post-writing check of resulting file content, if fails do fallback of full rewrite. 
            # NOTE: (can we still access the old values to created extended_historic_ds??)
            n_appended = ds_to_append.dims['date']
            old_and_new_dates = (xr.open_dataset(HISTO_ZARR_OUTPUT)
                .isel(date = slice(-n_appended-1,-n_appended+1))
                .date.values) 
            if (old_and_new_dates[1] - old_and_new_dates[0]) != np.timedelta64(1, 'D'):
                raise ValueError(f"Dates of resulting data set are not exactly 1 day apart at interface: {old_and_new_dates}")
            else:
                print("Append completed.", flush=True)

        except Exception as e:
            print(f"Appending failed: {e}. Falling back to rewrite.", flush=True)
            # Backup original store (move directory) and write full dataset
            backup = HISTO_ZARR_INPUT + ".backup_" + dt.datetime.now().strftime("%Y%m%d%H%M%S")
            try:
                shutil.move(HISTO_ZARR_INPUT, backup)
                print(f"Backed up original store to {backup}", flush=True)
            except Exception as e2:
                print(f"Backup failed: {e2} -- continuing to overwrite.", flush=True)
            
            # duplicate of else
            fallback_action_overwrite_zarr()
    else:
        print(f"writing to new file\n  {HISTO_ZARR_INPUT}\n=> {HISTO_ZARR_OUTPUT}", flush=True)
        fallback_action_overwrite_zarr()

    client.close()

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds")

    print("Modified/Created file: ", flush = True)
    print(HISTO_ZARR_OUTPUT, flush = True)
    sys.exit(0)
