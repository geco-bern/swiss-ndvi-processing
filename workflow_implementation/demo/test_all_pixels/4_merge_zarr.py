"""
Merge all-time (historic) record onto newly downloaded NDVI data to prepare for script 5
"""
import numpy as np
import zarr
import pandas as pd
import os, sys
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import argparse
import datetime

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)
warnings.filterwarnings(
    "ignore", 
    message="Consolidated metadata is currently not part in the Zarr format 3 specification",
    module=r"zarr\.api\.asynchronous"
)

# HOW TO RUN FROM BASH:
# source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
# SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.py"
# LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"
# DOWNLOAD_FILE="/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12.zarr"
# HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr"
# python -u $SCRIPT_FILE $DOWNLOAD_FILE $HISTO_INPUT > $LOG_FILE  2>&1 &

# output will be a temporary zarr file that can be used for script 05
# script returns filename

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("DOWNLOAD_ZARR", help="Full path of Zarr folder, downloaded with script 01")
    parser.add_argument("HISTO_INPUT",   help="Full path to Zarr folder with historic NDVI data")
    args = parser.parse_args()

    DOWNLOAD_ZARR        = args.DOWNLOAD_ZARR
    HISTO_ZARR           = args.HISTO_INPUT
    # if running interactively use e.g.:
        # DOWNLOAD_ZARR     = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-23_12h50_ndvi_01_downloaded_2025-11-30_2026-03-22.zarr"
        # HISTO_ZARR         = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_10kmX10km.zarr"

        # DOWNLOAD_ZARR     = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-17_23h33_ndvi_01_downloaded_2025-11-30_2025-12-06.zarr"
        # DOWNLOAD_ZARR     = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_15h29_ndvi_01_downloaded_2025-11-30_2025-12-06.zarr"
        # DOWNLOAD_ZARR     = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12.zarr"
        # HISTO_ZARR         = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_1000mX1000m.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
        # OUT_ZARR_TMP      = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_1000mX1000m_4th.zarr" # TODO: do not create this but simply merge in script 5
        # HISTO_ZARR        = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
        # OUT_ZARR_TMP      = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_10kmX10km_4th.zarr" # TODO: do not create this but simply merge in script 5
        # HISTO_ZARR        = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_100kmX100km.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
        # OUT_ZARR_TMP      = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_100kmX100km_4th.zarr" # TODO: do not create this but simply merge in script 5
        # HISTO_ZARR        = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
        # OUT_ZARR_TMP      = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_4th.zarr" # TODO: do not create this but simply merge in script 5

    OUT_ZARR_TMP = DOWNLOAD_ZARR.replace(".zarr","_processed.zarr")
    DASK_TEMP_DIR = "/mnt/data2/UniBe-swiss-ndvi/tmp_data4/"
    os.makedirs(DASK_TEMP_DIR, exist_ok=True)

    N_WORKERS = 50
    MEMORY_PER_WORKER = "24GB"
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_PER_WORKER,
        dashboard_address=":8343",
        local_directory= DASK_TEMP_DIR,
    )
    client = Client(cluster)
    print(client, flush = True)
    print(client.dashboard_link, flush = True) # use this dashboard to follow progress


    # =====================================================
    #  Load historic and new observation data sets
    # =====================================================
    DATE_CHUNKS = 365
    PIXEL_CHUNKS = 40000
    # DATE_CHUNKS  = historic_ds.chunks['date'][0]  # should be 30 days # TODO: why not this?
    # PIXEL_CHUNKS = historic_ds.chunks['pixel'][0]                     # TODO: why not this?


    # --- load historic dataset ------------------------------------
    historic_ds = xr.open_zarr(HISTO_ZARR, chunks={})

    # --- load new data dataset ------------------------------------
    new_observations_ds = xr.open_dataset(DOWNLOAD_ZARR, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "datetime": -1})

    # Subset new_observations_ds to correspond to same pixels as in historic_ds
    if (len(historic_ds.pixel.values) < len(new_observations_ds.pixel.values)):
        print(f"Subsetting downloaded data to spatial extent of historic file:\n{HISTO_ZARR}", flush = True)
        print(f"Subsetting {len(historic_ds.pixel.values)} (historic) of {len(new_observations_ds.pixel.values)} (downloaded) pixels.", flush = True)
    
    new_observations_ds = new_observations_ds.sel(pixel=historic_ds.pixel)
    
    # attempt to plot
        # new_observations_ds
        # xmin, xmax = 2600000, 2601500
        # ymin, ymax = 1196000, 1197500
        # pixels_subset_mask = (
        #     (new_observations_ds.x.values >= xmin) &
        #     (new_observations_ds.x.values <= xmax) &
        #     (new_observations_ds.y.values >= ymin) &
        #     (new_observations_ds.y.values <= ymax)
        # )
        # new_observations_ds["ndvi"].x.values
        # new_observations_subset_ds = new_observations_ds["ndvi"].isel(pixel=pixels_subset_mask.nonzero()[0])
        # plot_da_map(new_observations_subset_ds.isel(datetime = 0))
    
    # =====================================================
    #  Aggregate multiple daily observation
    #  and resample to daily intervals (between observations)
    # =====================================================

    # FOR DEVELOPMENT: new_observations_ds["ndvi"].isel(datetime = 1) # 2025-12-09T10:33:29
    # FOR DEVELOPMENT: new_observations_ds["ndvi"].isel(datetime = 2) # 2025-12-09T10:44:51
    # FOR DEVELOPMENT: plot_da_map(new_observations_ds["ndvi"].isel(datetime = 1),
    # FOR DEVELOPMENT:             reduction_factor = 5, png_fname = 'NDVI_2025-12-09_10h33.png')
    # FOR DEVELOPMENT: plot_da_map(new_observations_ds["ndvi"].isel(datetime = 2),
    # FOR DEVELOPMENT:             reduction_factor = 5, png_fname = 'NDVI_2025-12-09_10h44.png')
    
    INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
    NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step
    
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
    #  Initialize empty daily dataset to append to historic
    #  i.e. extend it back to last historic date
    # =====================================================
    last_historic_date = historic_ds['date'].max().values
    start_date         = np.datetime_as_string(last_historic_date + 1, unit='D')  # Note the shift +1, since we want to avoid a duplicate
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
    # Initialized n=13 daily dates:
    #         daily   obs_date          obs_times
    # 0  2025-11-30        NaT                   
    # 1  2025-12-01        NaT                   
    # 2  2025-12-02        NaT                   
    # 3  2025-12-03        NaT                   
    # 4  2025-12-04        NaT                   
    # 5  2025-12-05        NaT                   
    # 6  2025-12-06 2025-12-06           10:23:19
    # 7  2025-12-07        NaT                   
    # 8  2025-12-08        NaT                   
    # 9  2025-12-09 2025-12-09  10:33:29,10:44:51
    # 10 2025-12-10        NaT                   
    # 11 2025-12-11        NaT                   
    # 12 2025-12-12 2025-12-12           10:43:39


    # new_observations_ds["ndvi"].values.shape    # (4,4216)
    # ndvi_daily_between_obs["ndvi"].values.shape # (7,4216)
    # new_observations_ds.datetime.values         # 4 values from (2025-12-06_10h23, 2025-12-09_10h33, 2025-12-09_10h44, 2025-12-12_10h43)
    # ndvi_daily_between_obs.datetime.values      # 7 values from (2025-12-06, ..., 2025-12-12)

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
    new_ds = (ndvi_daily_since_last_historic
                .rename({'ndvi':'ndvi_obs',  # TODO: this does not make sense. Given that we already interpolated ndvi would be a better name than ndvi_obs
                         'ndsi':'ndsi_obs'}) # TODO: this does not make sense. Given that we already interpolated ndvi would be a better name than ndvi_obs
                .chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS}))
    # new_ds has: 
    #   coords: x,y,x_idx,y_idx, pixel, date, datetime; 
    #   vars:   ndvi_obs,ndsi_obs,obs_date
    #   attrs:  pixel_definition,transform_note,transform_coeffs,transform_instr,description_ndvi,description_ndsi,nodata,cloud_shadow
    
    # drop any coord/data var chunk encodings that conflict
    for name in list(new_ds.coords) + list(new_ds.data_vars):
        new_ds[name].encoding.pop("chunks", None)
        new_ds[name].encoding.pop("compressor", None)
        new_ds[name].encoding.pop("compressors", None)

    # write out    
    new_ds.to_zarr(OUT_ZARR_TMP, mode="w", zarr_format=3)


    # overview of data structures: ---------------------------------------------
    # historic_ds
    # <xarray.Dataset> Size: 40MB
    # Dimensions:         (pixel: 4216, date: 3164)
    # Coordinates:
    #   * pixel           (pixel) int32 17kB 44311103 44311104 ... 45049987 45049988
    #   * date            (date) datetime64[ns] 25kB 2017-04-03 ... 2025-11-30
    #     doy             (date) int32 13kB dask.array<chunksize=(30,), meta=np.ndarray>
    #     x_idx           (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     y_idx           (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     x               (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     y               (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    # Data variables:
    #     ndvi_processed  (pixel, date) int16 27MB dask.array<chunksize=(4216, 30), meta=np.ndarray>
    #     mask_array      (pixel, date) int8 13MB dask.array<chunksize=(4216, 30), meta=np.ndarray>
    # Attributes:
    #     [...]

    # new_ds
    # <xarray.Dataset> Size: 304kB
    # Dimensions:   (date: 13, pixel: 4216)
    # Coordinates:
    #   * pixel     (pixel) int32 17kB 44311103 44311104 ... 45049987 45049988
    #   * date      (date) datetime64[ns] 104B 2025-11-30 2025-12-01 ... 2025-12-12
    #     x_idx     (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     y_idx     (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     y         (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     x         (pixel) int32 17kB dask.array<chunksize=(4216,), meta=np.ndarray>
    #     doy       (date) int32 52B dask.array<chunksize=(13,), meta=np.ndarray>
    # Data variables:
    #     ndsi_obs  (date, pixel) int16 110kB dask.array<chunksize=(1, 4216), meta=np.ndarray>
    #     ndvi_obs  (date, pixel) int16 110kB dask.array<chunksize=(1, 4216), meta=np.ndarray>
    #     obs_date  (date) int8 13B dask.array<chunksize=(13,), meta=np.ndarray>
    # Attributes:
    #     [...]


    # and materialized with compute() ------------------------------------------
    # new_ds.compute()
    # <xarray.Dataset> Size: 304kB
    # Dimensions:   (date: 13, pixel: 4216)
    # Coordinates:
    #   * pixel     (pixel) int32 17kB 44311103 44311104 ... 45049987 45049988
    #   * date      (date) datetime64[ns] 104B 2025-11-30 2025-12-01 ... 2025-12-12
    #     x_idx     (pixel) int32 17kB 11353 11353 11353 11353 ... 11452 11452 11452
    #     y_idx     (pixel) int32 17kB 12591 12592 12594 12595 ... 12635 12636 12690
    #     y         (pixel) int32 17kB 1196995 1196995 1196995 ... 1196005 1196005
    #     x         (pixel) int32 17kB 2600005 2600015 2600035 ... 2600455 2600995
    #     doy       (date) int32 52B 334 335 336 337 338 339 ... 342 343 344 345 346
    # Data variables:
    #     ndsi_obs  (date, pixel) int16 110kB 32767 32767 32767 ... 32767 32767 32767
    #     ndvi_obs  (date, pixel) int16 110kB 32767 32767 32767 ... 32767 32767 32767
    #     obs_date  (date) int8 13B TODO

    # Attributes:
    # <xarray.Dataset> Size: 40MB
    # Dimensions:         (pixel: 4216, date: 3164)
    # Coordinates:
    #   * pixel           (pixel) int32 17kB 44311103 44311104 ... 45049987 45049988
    #   * date            (date) datetime64[ns] 25kB 2017-04-03 ... 2025-11-30
    #     doy             (date) int32 13kB 93 94 95 96 97 98 ... 330 331 332 333 334
    #     x_idx           (pixel) int32 17kB 11353 11353 11353 ... 11452 11452 11452
    #     y_idx           (pixel) int32 17kB 12591 12592 12594 ... 12635 12636 12690
    #     y               (pixel) int32 17kB 1196995 1196995 ... 1196005 1196005
    #     x               (pixel) int32 17kB 2600005 2600015 ... 2600455 2600995
    # Data variables:
    #     ndvi_processed  (pixel, date) int16 27MB 7105 7108 7112 ... 5427 5398 5372
    #     mask_array      (pixel, date) int8 13MB TODO

    client.close()
    print("All done")

    print(f"4_merge_zarr.py created file: {OUT_ZARR_TMP}", flush = True)
    print(OUT_ZARR_TMP, flush = True)
    sys.exit(0)

