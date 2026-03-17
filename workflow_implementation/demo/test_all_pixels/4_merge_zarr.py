"""
Merge all-time (historic) record onto newly downloaded NDVI data to prepare for script 5
"""
import numpy as np
import zarr
import pandas as pd
import os
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import argparse
import datetime

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
    parser.add_argument("end_date", help="End date in YYYY-MM-DD")
    parser.add_argument("SOURCE_ZARR", help="Full path of Zarr folder, modified with script 2 and 3")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    SOURCE_ZARR = args.SOURCE_ZARR
    # if running interactively use e.g.:
    #   start_date = "2025-11-30" # for dates requested...
    #   end_date = "2026-03-10"   # ...in script 1 when downloading
    #   end_date = "2025-12-12"   # ...in script 1 when downloading
    #   SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-17_16h27_ndvi_02-03_downloadedB_2025-11-30_2025-12-12.zarr"
                    #SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr" # the zarr from script 3
                    #SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_02-03_downloadedB.zarr" # the zarr from script 3
                    #SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_02-03_downloadedB_2026-03-17.zarr" # the zarr from script 3
                    #SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-17_16h27_ndvi_02-03_downloadedB_2025-11-30_2025-12-12.zarr"
                    #SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-17_16h35_ndvi_02-03_downloadedB_2025-11-30_2025-12-06.zarr"
                    # SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr" 
    #   SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr"

    historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
    #historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v3_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
    
    OUT_ZARR_TMP   = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_1000mX1000m_3rd.zarr" # TODO: do not create this but simply merge in script 5
    #OUT_ZARR_TMPold= "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_2026-03-12/" # TODO: do not create this but simply merge in script 5
    #os.makedirs(OUT_ZARR_TMPold, exist_ok=True)
    
    DASK_TEMP_DIR = "/mnt/data1/UniBe-swiss-ndvi/tmp_data3/"
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
    #  Load Source Dataset lazily (no xarray metadata needed)
    # =====================================================

    # --- load historic dataset ------------------------------------
    historic_ds = xr.open_zarr(historical_ndvi_src, chunks={})
    
    # Rename ndvi_processed -> ndvi, mask_array -> obs_date to match new data schema
    historic_ds = historic_ds.rename(
        {
            "ndvi_processed": "ndvi",  # NOTE: we rename since the OUT_ZARR_TMP will contain part processed but also part unprocessed data
            "mask_array": "obs_date",  # NOTE: we rename since the OUT_ZARR_TMP will contain part processed but also part unprocessed data
        }
    )

    #DATE_CHUNKS = 365
    #PIXEL_CHUNKS = 10000
    DATE_CHUNKS  = historic_ds.chunks['date'][0]  # should be 30 days # TODO: why not this?
    PIXEL_CHUNKS = historic_ds.chunks['pixel'][0]                     # TODO: why not this?

        # TODO: with SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-17_16h27_ndvi_02-03_downloadedB_2025-11-30_2025-12-12.zarr"
        #            start_date = "2025-11-30" # for dates requested...
        #            end_date = "2025-12-12"   # ...in script 1 when downloading
        #       dates gives me duplicated dates: 
        #       ['2025-12-06' '2025-12-09' '2025-12-09' '2025-12-12'] 
        #           # these are 4 dates and ndvi_da is 4 long, but there are only 3 unique dates.
        #       dates_clean is still ['2025-12-06' '2025-12-09' '2025-12-09' '2025-12-12'] 
        #       unique_dates is ['2025-12-06' '2025-12-09' '2025-12-12']
        #       ndvi_xr is (4x105Mio)
        #       The code errors when ndvi_xr.assign_coords(date=unique_dates[:ndvi_da.shape[1]])  since unique_dates is too short.
        # TODO: with SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr"
        #            start_date = "2025-11-30"
        #            end_date   = "2026-03-10"
        #       dates does also give duplicated dates:
        #       ['2025-12-06' '2025-12-09' '2025-12-09' '2025-12-12' '2025-12-13'
        #        '2025-12-13' '2025-12-14' '2025-12-21' '2025-12-26' '2025-12-26'
        #        '2025-12-27' '2025-12-28' '2025-12-29' '2025-12-29' '2025-12-30'
        #        '2025-12-31' '2026-01-01' '2026-01-03' '2026-01-05' '2026-01-05'
        #        '2026-01-06' '2026-01-07' '2026-01-09' '2026-01-13' '2026-01-15'
        #        '2026-01-16' '2026-01-18' '2026-01-20' '2026-01-21' '2026-01-22'
        #        '2026-01-27' '2026-01-30' '2026-02-02' '2026-02-05' '2026-02-22'
        #        '2026-02-25' '2026-02-26' '2026-03-02' '2026-03-03' '2026-03-04'
        #        '2026-03-04'] 
        #             # these are 41 (repeated) dates, and ndvi_da is only 34 long and len(unique_dates) == 35
        #       dates_clean contains 34 values
        #       unique_dates contains 35 values
        #       ndvi_xr and ndvi_da is of dimension (34x105Mio)
        #       When doing ndvi_xr.assign_coords(date=unique_dates[:ndvi_da.shape[1]])
        #         this appears to be wrong: 1. we are just dropping the last unique_dates?
        #         2. MAIN QUESTION: how come len(dates) can be longer than ndvi_da
    # xr.open_dataset(SOURCE_ZARR) # TODO: this is not possible
    ds0 = zarr.open_group(SOURCE_ZARR, mode="r")
    ndvi_da = da.from_zarr(ds0["ndvi"])
    dates = da.from_zarr(ds0["date"]).astype("datetime64[D]").compute()
    dates.sort()

    start_date  = np.datetime64(start_date, "D")
    end_date    = np.datetime64(end_date, "D")
    
    dates_clean = dates[:ndvi_da.shape[1]] # TODO: what does it mean: dates_clean? This appears wrong, if dates is longer than ndvi_da, it only takes the first n dates.
    unique_dates, unique_idx = np.unique(dates, return_index=True) 
    # TODO: why is unique_idx never used. I believe this should be used to index only first occurrence of each date in ndvi_da, no? Otherwise we get ndvi values of the wrong dates...

    ndvi_xr = xr.DataArray(
        ndvi_da,                 # this is 34x105Mio
        dims=("pixel", "date"),
        coords={
            "pixel": np.arange(ndvi_da.shape[0], dtype=np.int32),
            "date": dates_clean
        },
        name="ndvi"
    ).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

    ndvi_xr    = ndvi_xr.assign_coords(date=unique_dates[:ndvi_da.shape[1]]) 
    # TODO: ValueError: conflicting sizes for dimension 'date': length 4 on <this-array> and length 3 on {'pixel': 'pixel', 'date': 'date'}
    # NOTE: because unique_dates is only 3 long
    # NOTE: but ndvi_da has a total of 4 date slices
    # TODO: with SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr"
    #            start_date = "2025-11-30"
    #            end_date   = "2026-03-10"
    # this 'appears' to work: ndvi_da is 34 long, len(unique_dates) == 35, so above line just drops '2026-03-04' to make it work
    # 

    daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    print(f"Generated {len(daily_dates)} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}", flush = True)
    obs_dates_xr = xr.DataArray(
        daily_dates.isin(dates), 
        dims=("date",), 
        coords={"date": daily_dates}, 
        name="obs_dates"
    ).chunk({"date": DATE_CHUNKS})
    ndvi_daily = ndvi_xr.reindex(date=daily_dates, method=None, fill_value=np.int16(32767)).astype(np.int16)

    # =====================================================
    #  Assemble Daily Dataset and write
    # =====================================================
    out_ds = xr.Dataset(
        {
            "ndvi": ndvi_daily,
            "obs" : obs_dates_xr
        }
    ).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    print("Date added", flush = True)

    # out_ds has: coords pixel, date; vars ndvi, obs
    # Rename obs -> obs_date so names match historic
    new_ds = out_ds.rename({"obs": "obs_date"})

    # Subset new_ds to correspond to same pixels as in historic_ds
    if (len(historic_ds.pixel.values) < len(new_ds.pixel.values)):
        print(f"Subsetting downloaded data to spatial extent of historic file:\n{historical_ndvi_src}", flush = True)
        print(f"Subsetting {len(historic_ds.pixel.values)} (historic) of {len(new_ds.pixel.values)} (downloaded) pixels.", flush = True)
    
    new_ds = new_ds.isel(pixel=historic_ds.pixel.values)
    
    # Ensure coords are exactly the same set of names (pixel, date, x, y)
    new_ds = new_ds.assign_coords(
        x=historic_ds["x"],
        y=historic_ds["y"],
    )
    #new_ds # TODO: FB this is an attempted workaround to move merging of the historic and new data into script 5. TODO_FB.zarr only contains new data.
    # new_ds.to_zarr("/mnt/data1/UniBe-swiss-ndvi/TODO_FB.zarr", mode="w", consolidated=True)
    # TODO: stop here and do rest in script 5.

    # --- concatenate full datasets along time ----------------------------------
    # TODO: add doy already here
    historic_ds_to_merge = historic_ds.drop_vars("doy") # NOTE: Drop DOY, we compute it later from date_stack. # TODO: why? Why not compute here for the small part where we need it?
    historic_ds_to_merge = historic_ds_to_merge.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    merged_ds = xr.concat([historic_ds_to_merge, new_ds], dim="date").sortby("date")
    
    merged_ds = merged_ds.chunk(
        {"pixel": min(PIXEL_CHUNKS, len(merged_ds.pixel)), 
         "date": min(DATE_CHUNKS, len(merged_ds.date))})
    
    # Now merged_ds has variables: ndvi, obs_date (for all time)
    # and coords: pixel, date, x, y
                        # TODO: can we simply write this out in a single (chunked data set instead of manually splitting by year?)
                        #       I would guess so. But currently first fix script 5 with the yearly files before attempting to make a shortcut here.
    # Write to Zarr (as single file)
    _now = datetime.datetime.now()
    print(f"{_now.strftime("%Y-%m-%d %H:%M:%S")} - Writing single-file data: {OUT_ZARR_TMP}", flush = True)
    # TODO: fix encoding if we want to change it from historic (30 days) to something else
    # merged_ds.encoding 
    merged_ds.to_zarr(OUT_ZARR_TMP, mode="w", consolidated=True, compute=True)
    _now = datetime.datetime.now()
    print(f"{_now.strftime("%Y-%m-%d %H:%M:%S")} - Finished writing single-file data: {OUT_ZARR_TMP}", flush = True)

    # TODO: IF THAT WORKS FINISH SCRIPT HERE.

    # # TODO: OTHERWISE CONTINUE WITH ALTERNATIVE:
    # # OR ALTERNATIVE: Write to Zarr (as yearly files)
    # date_stack = merged_ds["date"].astype("datetime64[D]")

    # years = pd.DatetimeIndex(date_stack).year   

    # # Years: 2017-2026
    # start_year = pd.to_datetime(start_date).year
    # end_year   = pd.to_datetime(end_date).year
    # years = [start_year] if start_year == end_year else [start_year, end_year]

    # for year in years:

    #     year_dates = merged_ds.date.dt.year == year
    #     year_ds = merged_ds.isel(date=year_dates)

    #     year_ds = year_ds.load() # TODO: this gives an error currently

    #     out_ds_year = xr.Dataset(
    #         {
    #             "ndvi": year_ds["ndvi"], 
    #             "obs_date": year_ds["obs_date"],
    #         },
    #         coords={
    #             "pixel": year_ds.pixel,
    #             "date": year_ds.date,
    #             "x": year_ds["x"],
    #             "y": year_ds["y"],
    #         },
    #     )
        
    #     out_ds_year["obs_date"].encoding = {"dtype": "bool"}
        
    #     for coord in ["x", "y", "pixel", "date"]:
    #         out_ds_year[coord].encoding = {}
        
    #     # Write to year-specific folder
    #     year_out_zarr = f"{OUT_ZARR_TMPold}/{year}.zarr"
    #     print(f"Writing {year}: {len(year_ds.date)} dates to {year_out_zarr}", flush = True)

    #     out_ds_year.to_zarr(year_out_zarr, mode="w", consolidated=True)

    print("All done")
