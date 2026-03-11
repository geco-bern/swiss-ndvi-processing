"""
Merge all-time (historic) record onto newly downloaded NDVI data to prepare for script 5
"""
# "Run Python File" in VSCode
import numpy as np
import zarr
import pandas as pd
import os
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import argparse

if __name__ == "__main__":
    
    SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr" # the zarr from script 3
    historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v3_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
                # TODO_SMALL_HIST_NDVI_TO_UPDATE = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v3_compr_subset.zarr" ## TODO: Small example for code development
                # xr.open_zarr(historical_ndvi_src, chunks={}
                #              ).isel(pixel = slice(0, 10**6) # TODO: generate here a small example on the fly.
                #                     ).to_zarr(TODO_SMALL_HIST_NDVI_TO_UPDATE, mode="w", consolidated=True)
    
    OUT_ZARR_TMP = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged.zarr" # TODO: do not create this but simply merge in script 5
    DASK_TEMP_DIR = "/mnt/data1/UniBe-swiss-ndvi/tmp_data/"
    os.makedirs(DASK_TEMP_DIR, exist_ok=True)

    N_WORKERS = 150
    MEMORY_PER_WORKER = "24GB"
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_PER_WORKER,
        dashboard_address=":8345",
        local_directory= DASK_TEMP_DIR,
    )
    client = Client(cluster)
    print(client.dashboard_link) # use this dashboard to follow progress


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

    # DATE_CHUNKS = 365
    # PIXEL_CHUNKS = 10000
    DATE_CHUNKS  = historic_ds.chunks['date'][1]  # should be 30 days
    PIXEL_CHUNKS = historic_ds.chunks['pixel'][1]

    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
    parser.add_argument("end_date", help="End date in YYYY-MM-DD")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    # if running interactively use e.g.:
    # start_date = "2025-11-30" # for dates requested...
    # end_date = "2026-03-10"   # ...in script 1 when downloading

    start_date = np.datetime64(start_date, "D")
    end_date = np.datetime64(end_date, "D")


    ds0 = zarr.open_group(SOURCE_ZARR, mode="r")
    ndvi_z = ds0["ndvi"]
    ndvi_da = da.from_zarr(ndvi_z)     # lazy
            # equivalently sel() but much slower

    dates = da.from_zarr(ds0["date"]).astype("datetime64[D]") # lazy

    dates = dates.compute()  # non-lazy


    daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    print(f"Generated {len(daily_dates)} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

    obs_dates = daily_dates.isin(dates)

    dates_clean = dates[:ndvi_da.shape[1]]  # 41→34 PERFECT MATCH

    ndvi_xr = xr.DataArray(
        ndvi_da,
        dims=("pixel", "date"),
        coords={
            "pixel": np.arange(ndvi_da.shape[0], dtype=np.int32),
            "date": dates_clean  # EXACTLY 34 ✅
        },
        name="ndvi"
    ).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

    unique_dates, unique_idx = np.unique(dates, return_index=True)
    ndvi_xr = ndvi_xr.assign_coords(date=unique_dates[:ndvi_da.shape[1]])


    ndvi_daily = ndvi_xr.reindex(date=daily_dates, method=None, fill_value=np.int16(32767)).astype(np.int16)

    obs_dates_xr = xr.DataArray(
        obs_dates, dims=("date",), coords={"date": daily_dates}, name="obs_dates"
    ).chunk({"date": DATE_CHUNKS})

    # =====================================================
    #  Assemble Daily Dataset and write
    # =====================================================
    out_ds = xr.Dataset(
        {
            "ndvi": ndvi_daily,
            "obs" : obs_dates_xr
        }
    ).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    print("Date added")

    # out_ds has: coords pixel, date; vars ndvi, obs
    # Rename obs -> obs_date so names match historic
    new_ds = out_ds.rename({"obs": "obs_date"})
    
    # Ensure coords are exactly the same set of names (pixel, date, x, y)
    new_ds = new_ds.assign_coords(
        x=historic_ds["x"],
        y=historic_ds["y"],
    )
    #new_ds # TODO: FB this is an attempted workaround to move merging of the historic and new data into script 5. TODO_FB.zarr only contains new data.
    # new_ds.to_zarr("/mnt/data1/UniBe-swiss-ndvi/TODO_FB.zarr", mode="w", consolidated=True)
    # TODO: stop here and do rest in script 5.

    # --- concatenate full datasets along time ----------------------------------
    merged_ds = xr.concat(
        [historic_ds.drop_vars("doy"), new_ds], # NOTE: Drop DOY, we compute it later from date_stack. # TODO: why? Why not compute here for the small part where we need it?
        dim="date",
    ).sortby("date").chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    
    # Now merged_ds has variables: ndvi, obs_date (for all time)
    # and coords: pixel, date, x, y
                        # TODO: can we simply write this out in a single (chunked data set instead of manually splitting by year?)
                        #       It appears that no. 
    # Write to Zarr
    merged_ds.to_zarr(OUT_ZARR_TMP, mode="w", consolidated=True, compute=True)

    # TODO: check: date_stack = merged_ds["date"].astype("datetime64[D]")
# TODO: check: 
    # TODO: check: years = pd.DatetimeIndex(date_stack).year   
# TODO: check: 
    # TODO: check: # Years: 2017-2026
    # TODO: check: start_year = pd.to_datetime(start_date).year
    # TODO: check: end_year   = pd.to_datetime(end_date).year
    # TODO: check: years = [start_year] if start_year == end_year else [start_year, end_year]
# TODO: check: 
    # TODO: check: for year in years:
# TODO: check: 
    # TODO: check:     year_dates = merged_ds.date.dt.year == year
    # TODO: check:     year_ds = merged_ds.isel(date=year_dates)
# TODO: check: 
    # TODO: check:     year_ds = year_ds.load()
# TODO: check: 
    # TODO: check:     out_ds_year = xr.Dataset(
    # TODO: check:         {
    # TODO: check:             "ndvi": year_ds["ndvi"], 
    # TODO: check:             "obs_date": year_ds["obs_date"],
    # TODO: check:         },
    # TODO: check:         coords={
    # TODO: check:             "pixel": year_ds.pixel,
    # TODO: check:             "date": year_ds.date,
    # TODO: check:             "x": year_ds["x"],
    # TODO: check:             "y": year_ds["y"],
    # TODO: check:         },
    # TODO: check:     )
    # TODO: check:     
    # TODO: check:     out_ds_year["obs_date"].encoding = {"dtype": "bool"}
    # TODO: check:     
    # TODO: check:     for coord in ["x", "y", "pixel", "date"]:
    # TODO: check:         out_ds_year[coord].encoding = {}
    # TODO: check:     
    # TODO: check:     # Write to year-specific folder
    # TODO: check:     year_out_zarr = f"{base_out_dir}/{year}.zarr"
    # TODO: check:     print(f"Writing {year}: {len(year_ds.date)} dates to {year_out_zarr}")
# TODO: check: 
    # TODO: check:     out_ds_year.to_zarr(year_out_zarr, mode="w", consolidated=True)

    print("All done")
