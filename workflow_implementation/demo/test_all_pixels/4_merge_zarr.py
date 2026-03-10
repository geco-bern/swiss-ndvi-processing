"""
Merge newly downloaded NDVI data to the all-time (historic) record
"""
# "Run Python File" in VSCode
import numpy as np
import math
import zarr
import pandas as pd
import os
import xarray as xr
import dask.array as da
import shutil
from dask.distributed import Client
import multiprocessing
import argparse

#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.log 2>&1 &


if __name__ == "__main__":
    
    # Two inputs from outside the workflow: # TODO: replace these two with data from the workflow.
    historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_processed_all_pixels_v3_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
    SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr" # the zarr from script 3
    OUT_ZARR_TMP = "/mnt/data1/UniBe-swiss-ndvi/data/temporary_demo.zarr"          # TODO: what does this file represent?
    base_out_dir = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/04_merged_ndvi"       # TODO: what does this file represent? Is it the updated historical_ndvi ? So in the real workflow this is the same as SOURCE_ZARR?

    # =====================================================
    #  Load Source Dataset lazily (no xarray metadata needed)
    # =====================================================

    DATE_CHUNKS = 365
    PIXEL_CHUNKS = 5000

    last_date_historical = "2025-11-30"
    

    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
    parser.add_argument("end_date", help="End date in YYYY-MM-DD")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

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
    )

    #out_ds = xr.Dataset({"ndvi": ndvi_daily, "obs": obs_dates_xr}).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

    # =====================================================
    #  Assemble Dataset and write
    # =====================================================
    out_ds = xr.Dataset(
        {
            "ndvi": ndvi_daily,
            "obs" : obs_dates_xr
        }
    ).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

    # Write to Zarr
    os.makedirs(OUT_ZARR_TMP, exist_ok=True)

    out_ds.to_zarr(OUT_ZARR_TMP, mode="w", consolidated=True)
    print("Date added")


    # --- load historic dataset and drop doy ------------------------------------
    historical_ndvi = xr.open_zarr(historical_ndvi_src, chunks={})

    # Keep only dates up to last_date_historical
    historic_ds = historical_ndvi.sel(date=slice(None, last_date_historical))

    # We don't need doy here; we compute it later from date_stack
    if "doy" in historic_ds.coords:
        historic_ds = historic_ds.drop_vars("doy")

    # Rename ndvi_processed -> ndvi, mask_array -> obs_date to match new data schema
    historic_ds = historic_ds.rename(
        {
            "ndvi_processed": "ndvi",
            "mask_array": "obs_date",
        }
    )

    # --- load new daily data ---------------------------------------------------
    ds_to_stack = xr.open_zarr(OUT_ZARR_TMP, chunks={})  # NO chunking


    # ds_to_stack has: coords pixel, date; vars ndvi, obs
    # Rename obs -> obs_date so names match historic
    new_ds = ds_to_stack.rename({"obs": "obs_date"})

    # Ensure coords are exactly the same set of names (pixel, date, x, y)
    new_ds = new_ds.assign_coords(
        x=historical_ndvi["x"],
        y=historical_ndvi["y"],
    )

    # --- concatenate full datasets along time ----------------------------------
    merged_ds = xr.concat(
        [historic_ds, new_ds],
        dim="date",
    ).sortby("date")
    # Now merged_ds has variables: ndvi, obs_date (for all time)
    # and coords: pixel, date, x, y

    date_stack = merged_ds["date"].astype("datetime64[D]")

    years = pd.DatetimeIndex(date_stack).year   

    # Years: 2017-2026
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    years = [start_year] if start_year == end_year else [start_year, end_year]

    for year in years:

        year_dates = merged_ds.date.dt.year == year
        year_ds = merged_ds.isel(date=year_dates)

        year_ds = year_ds.load()

        out_ds_year = xr.Dataset(
            {
                "ndvi": year_ds["ndvi"], 
                "obs_date": year_ds["obs_date"],
            },
            coords={
                "pixel": year_ds.pixel,
                "date": year_ds.date,
                "x": year_ds["x"],
                "y": year_ds["y"],
            },
        )
        
        out_ds_year["obs_date"].encoding = {"dtype": "bool"}
        
        for coord in ["x", "y", "pixel", "date"]:
            out_ds_year[coord].encoding = {}
        
        # Write to year-specific folder
        year_out_zarr = f"{base_out_dir}/{year}.zarr"
        print(f"Writing {year}: {len(year_ds.date)} dates to {year_out_zarr}")

        out_ds_year.to_zarr(year_out_zarr, mode="w", consolidated=True)

    print("All done")
