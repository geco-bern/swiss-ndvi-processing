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

#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr_all.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr_all.log 2>&1 &


if __name__ == "__main__":
    
    # Two inputs from outside the workflow: # TODO: replace these two with data from the workflow.
    historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_processed_all_pixels_v3_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
    base_out_dir = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/04_merged_ndvi"       # TODO: what does this file represent? Is it the updated historical_ndvi ? So in the real workflow this is the same as SOURCE_ZARR?

    # =====================================================
    #  Load Source Dataset lazily (no xarray metadata needed)
    # =====================================================

    DATE_CHUNKS = 365
    PIXEL_CHUNKS = 5000
    

    # --- load historic dataset and drop doy ------------------------------------
    historical_ndvi = xr.open_zarr(historical_ndvi_src, chunks={})

    # We don't need doy here; we compute it later from date_stack
    if "doy" in historical_ndvi.coords:
        historical_ndvi = historical_ndvi.drop_vars("doy")

    # Rename ndvi_processed -> ndvi, mask_array -> obs_date to match new data schema
    historical_ndvi = historical_ndvi.rename(
        {
            "ndvi_processed": "ndvi",
            "mask_array": "obs_date",
        }
    )


    # Years: 2017-2026
    start_year = 2017
    end_year = 2026
    years = list(range(start_year, end_year + 1))

    for year in years:

        year_dates = historical_ndvi.date.dt.year == year
        year_ds = historical_ndvi.isel(date=year_dates)

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
