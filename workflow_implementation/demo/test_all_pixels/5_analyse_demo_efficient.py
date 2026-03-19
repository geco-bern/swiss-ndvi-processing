import datetime as dt
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
from dask import visualize
import dask.array as da
import xarray as xr
import os
import shutil
import time

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

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

if __name__ == "__main__":

    t0 = time.perf_counter()

    # N_WORKERS = 20        # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 30s; 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1      # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 30s; 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000  # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 30s; 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '190GB'

    N_WORKERS = 30        # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 30s; 586503 pixels => 53s; 16041205 pixels => 640s; 105715396 pixels => XXs
    DATE_CHUNKS = -1      # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 30s; 586503 pixels => 53s; 16041205 pixels => 640s; 105715396 pixels => XXs
    PIXEL_CHUNKS = 10000  # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 30s; 586503 pixels => 53s; 16041205 pixels => 640s; 105715396 pixels => XXs
    MEMORY_PER_WORKER = '120GB'

    # N_WORKERS = 60        # D) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1      # D) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000  # D) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '66GB'

    DATE_CHUNKS_OUT = 365

    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit=MEMORY_PER_WORKER,
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':8343')  
    print(client.dashboard_link)

    INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr"

    HISTO_ZARR_IN_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m.zarr"
    HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m_extended.zarr" # TODO: remove this and instea do it circular
    INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_1000mX1000m_4th.zarr"
    # HISTO_ZARR_IN_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_10kmX10km.zarr"
    # HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_10kmX10km_extended.zarr" # TODO: remove this and instea do it circular
    # INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_10kmX10km_4th.zarr"
    # HISTO_ZARR_IN_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_100kmX100km.zarr"
    # HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_100kmX100km_extended.zarr" # TODO: remove this and instea do it circular
    # INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_100kmX100km_4th.zarr"
    # HISTO_ZARR_IN_OUTPUT = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr.zarr"
    # HISTO_ZARR_OUTPUT    = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_extended.zarr" # TODO: remove this and instea do it circular
    # INPUT_ZARR           = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_4th.zarr"
    
    # DATE_CHUNKS  = historic_ds.chunks['date'][0]  # should be 30 days # TODO: why not this?
    # PIXEL_CHUNKS = historic_ds.chunks['pixel'][0]                     # TODO: why not this?


    historic_ds  = xr.open_zarr(HISTO_ZARR_IN_OUTPUT, chunks={"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    new_ds       = xr.open_zarr(INPUT_ZARR, chunks={"pixel": PIXEL_CHUNKS, "date": -1})
    lookuptable  = xr.open_zarr(INPUT_LOOKUPTABLE, consolidated=False).chunk({"pixel": PIXEL_CHUNKS})

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
        ndvi_array,
        median_array,
        mask_array,
        dates_array,
        input_core_dims=[["date"], ["date"],["date"], ["date"]],    # each call gets 1D time arrays
        output_core_dims=[["date"],["date"]],
        vectorize=True, 
        dask="parallelized",
        kwargs={
             "dates": dates_array,     # TODO: why do we specify dates_array twice??
             "starting_date": start_date},
        output_dtypes=output_dtypes, 
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    # g = mask_processed.__dask_graph__()
    g = ndvi_processed.__dask_graph__()
    print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
    #                    586503 pixels:                  | 16041205 pixels:                | 105715396 pixels:
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
    ds_to_append = xr.Dataset({"ndvi_processed": ndvi_processed_to_append, 
                               "mask_array":     mask_processed_to_append})
    #ndvi_processed_to_append.compute() 
    #mask_processed_to_append.compute()
    #ds_to_append.compute()               # starts on 2025-12-01 # Note the shift +1
    #historic_ds_to_extend.compute()      # ends   on 2025-11-30

    # concatenate
    extended_historic_ds = (
         xr.concat([historic_ds_to_extend, new_ds], dim="date")
         .sortby("date")
         .chunk({"pixel": PIXEL_CHUNKS, 
                 "date": DATE_CHUNKS_OUT})
    )

    # # Remove any incompatible 'compressor' metadata left over from the source dataset
    # for v in list(extended_historic_ds.data_vars):
    #     extended_historic_ds[v].encoding.pop("compressor", None)
    #     extended_historic_ds[v].encoding.pop("compressors", None)
    #     # ensure chunks entry exists to avoid surprises
    #     extended_historic_ds[v].encoding.setdefault("chunks", None)

    # for c in list(extended_historic_ds.coords):
    #     extended_historic_ds[c].encoding.pop("compressor", None)
    #     extended_historic_ds[c].encoding.pop("compressors", None)
    #     extended_historic_ds[c].encoding.setdefault("chunks", None)
    

    # # Explicit encoding: no compressor for each data var
    # encoding = {v: {"compressor": None} for v in extended_historic_ds.data_vars} # TODO: why not? this should be following what was done to create v4 of historic

    # # drop any coord/data var chunk encodings that conflict   # TODO: we're already doing 
    # for name in list(extended_historic_ds.coords) + list(extended_historic_ds.data_vars):
    #     extended_historic_ds[name].encoding.pop("chunks", None)
    #     extended_historic_ds[name].encoding.pop("compressor", None)
    #     extended_historic_ds[name].encoding.pop("compressors", None)

    if os.path.exists(HISTO_ZARR_OUTPUT): # TODO: remove this when going circular
        shutil.rmtree(HISTO_ZARR_OUTPUT)  # TODO: remove this when going circular
    extended_historic_ds.to_zarr(
          HISTO_ZARR_OUTPUT, 
          mode="w", 
          consolidated=True, 
          compute=True, 
          encoding=encoding, 
          zarr_version=3)
    # TODO: see if we want to go circular
    #       whether we use: HISTO_ZARR_IN_OUTPUT and the appending with mode="a"

    client.close()

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds")
