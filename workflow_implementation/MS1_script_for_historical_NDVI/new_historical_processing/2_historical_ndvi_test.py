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
def historical_ndvi_singleWindow(ndvi_arr, median_arr, is_observation_date, dates):
        
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
        delta_delta_left  = delta_ndvi[:-2] - delta_ndvi[1:-1]
        delta_delta_rigth = delta_ndvi[2:] - delta_ndvi[1:-1]
        outlier_mask = ((abs(delta_ndvi[1:-1])  > delta_threshold) & 
                        (abs(delta_delta_left)  > delta_delta_threshold) & 
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


if __name__ == "__main__":

    N_WORKERS = 80
    # TODO: test later 120 workers, with each 30GB memory_limit, BATCH_SIZE = 200K, INNER_PIXEL_CHUNK=83
    # TODO: test later 150 workers, with each 24GB memory_limit, BATCH_SIZE = 150K, INNER_PIXEL_CHUNK=1K

    with Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit='20GB',
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':1235') as client:
    
        print(client, flush = True)
        print(client.dashboard_link, flush = True)
        print(dask.config.get("scheduler"), flush = True)

        INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"
        INPUT_ZARR_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
        OUT_PATH = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7b.zarr"

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
        # new_observations_ds = new_observations_ds.isel(pixel=slice(0,int(1e6))) # , datetime = slice(0,30)
        ## with 10 pixels:         runtime=55s,  storage=304KB
        ## with 100 pixels:        runtime=54s,  storage=644KB
        ## with 1_000 pixels:      runtime=61s, storage=4.1MB
        ## with 10_000 pixels:     runtime=141s, storage=39MB
        ## with 100_000 pixels:    runtime=1080s, storage=XXKB          historical_ndvi_singleWindow: 165s
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

        # Decide how to collapse sub-daily duplicates to one observed value per day.
        # (Manual "first" avoids the previously used expensive groupby(TimeResampler("1D")) graph build.)
        observation_datetimes = pd.DatetimeIndex(new_observations_ds["datetime"].values)
        if not observation_datetimes.is_monotonic_increasing:
            new_observations_ds = new_observations_ds.sortby("datetime")
            observation_datetimes = pd.DatetimeIndex(new_observations_ds["datetime"].values)

        observation_dates = observation_datetimes.floor("D")
        first_obs_idx = np.flatnonzero(~observation_dates.duplicated(keep="first"))
        first_obs_dates = observation_dates[first_obs_idx]

        ndvi_daily_between_obs = (
            new_observations_ds
            .isel(datetime=first_obs_idx)
            .drop_vars("date", errors="ignore")
            .assign_coords(obs_day=("datetime", first_obs_dates.values))
            .swap_dims({"datetime": "obs_day"})
            .drop_vars("datetime")
            .rename({"obs_day": "date"})
        )

        ndvi_daily_between_obs["ndvi"] = xr.where(
            (ndvi_daily_between_obs["ndvi"] != NO_COVERAGE) &
            (ndvi_daily_between_obs["ndvi"] != INVALID),
            ndvi_daily_between_obs["ndvi"],
            np.int16(NO_COVERAGE),
        ).astype(np.int16)

        # =====================================================
        #  Initialize empty daily dataset
        # =====================================================
        # note: we call this dataset since_last_historic in the continuous update.
        #       Here this is means simply since the first observation:
        start_date         = first_obs_dates.min()
        end_date           = first_obs_dates.max()

        # build full daily index from start_date to end_date (make sure start_date/end_date are pd-compatible)
        daily_dates_since_last_historic = pd.date_range(
            start=pd.to_datetime(start_date).floor("D"),
            end=pd.to_datetime(end_date).floor("D"),
            freq="D")

        # reindex coords to guarantee daily coverage starts at start_date
        # i.e. extending back to last historic date:
        ndvi_daily_since_last_historic = ndvi_daily_between_obs.reindex(
            date=daily_dates_since_last_historic, 
            fill_value=np.int16(NO_COVERAGE),
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
                historical_ndvi_singleWindow,
                batch_ds["ndvi_processed"],
                batch_ds["median_ndvi"],
                batch_ds["obs_date"],
                input_core_dims=[["date"], ["date"], ["date"]],   # These are dimensions that should not be broadcast
                output_core_dims=[["date"], ["date"]],
                kwargs={"dates": dates_array_arg},   # pass as NumPy, not Dask
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