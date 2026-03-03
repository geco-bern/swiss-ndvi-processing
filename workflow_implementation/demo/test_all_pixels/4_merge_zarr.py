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
import dask
from zarr.codecs import ZstdCodec

#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/4_merge_zarr.log 2>&1 &


if __name__ == "__main__":
    
    n_workers = 20
    client = Client(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit="8GB",  
        dashboard_address=":8787"
    )
    print(client.dashboard_link)

    # Two inputs from outside the workflow: # TODO: replace these two with data from the workflow.
    historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_processed_all_pixels_v3_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
    lookuptable_src = "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/data_for_demo/lookup_table.zarr" # TODO: can we replace this with:  "../../data/output/00_lookup_table_median_ndvi.zarr" ?
    # historical_ndvi_src = "../../data/output/04_merged_ndvi.zarr" # TODO: I guess this would be what we want. Right Francesco ??
    # lookuptable_src = "../../data/output/00_lookup_table_median_ndvi.zarr" # TODO: can we replace this with:  "../../data/output/00_lookup_table_median_ndvi.zarr" ?
    # TODO: should we: check if 00_lookup_table_median_ndvi.zarr exists? If it doesn't then run 1x a function that replaces 0_create_lookup_table.py ?

    SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr" # the zarr from script 3
    OUT_ZARR_TMP = "../../data/temporary_demo.zarr"          # TODO: what does this file represent?
    OUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/04_merged_ndvi_compressed.zarr"       # TODO: what does this file represent? Is it the updated historical_ndvi ? So in the real workflow this is the same as SOURCE_ZARR?

    # =====================================================
    #  Load Source Dataset lazily (no xarray metadata needed)
    # =====================================================

    DATE_CHUNKS = 365
    PIXEL_CHUNKS = 5000

    if os.path.exists(OUT_ZARR):
        shutil.rmtree(OUT_ZARR, ignore_errors=True)

    last_date_historical = "2025-11-30"
    start_dates = np.datetime64("2025-12-01", "D")
    end_dates = np.datetime64("2026-02-12", "D")

    ds0 = zarr.open_group(SOURCE_ZARR, mode="r")
    ndvi_z = ds0["ndvi"]
    ndvi_da = da.from_zarr(ndvi_z)     # lazy
            # equivalently sel() but much slower

    dates = da.from_zarr(ds0["date"]).astype("datetime64[D]") # lazy

    dates = dates.compute()  # non-lazy


    daily_dates = pd.date_range(start=start_dates, end=end_dates, freq="D")
    print(f"Generated {len(daily_dates)} daily dates from {daily_dates.min().date()} to {daily_dates.max().date()}")

    obs_dates = daily_dates.isin(dates)

    # =====================================================
    #  Forest Pixel Selection
    # =====================================================
    ndvi_xr = xr.DataArray(
        ndvi_da,
        dims=("pixel", "date"),
        coords={
            "pixel": np.arange(ndvi_da.shape[0], dtype=np.int64),   # the actual pixel coordinate
            "date": dates
        },
        name="ndvi"
    ).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

    # =====================================================
    #  Reindex NDVI to daily (lazy), fill with 32767 (int16)
    # =====================================================

    def _dedup_date_coord(da: xr.DataArray, how: str = "first") -> xr.DataArray:
        """
        Ensure da's date coordinate is unique.
        how: 'first' | 'last' | 'mean' | 'median' | 'max'
        - 'first'/'last': keep first/last occurrence (fast, no compute)
        - reductions: aggregate duplicates lazily with Dask
        """
        idx = pd.Index(da["date"].values)
        if not idx.has_duplicates:
            return da
        if how in ("first", "last"):
            keep_mask = ~idx.duplicated(keep=how)
            return da.isel(date=np.nonzero(keep_mask)[0])
        if how == "mean":
            return da.groupby("date").mean("date")
        if how == "median":
            return da.groupby("date").median("date")
        if how == "max":
            return da.groupby("date").max("date")
        raise ValueError(f"Unsupported how={how}")

    # Deduplicate time first, then reindex to daily
    # Change how='first' to 'mean'/'median'/'max' as needed
    ndvi_sel_nodup = _dedup_date_coord(ndvi_xr, how="first")
    ndvi_daily = ndvi_sel_nodup.astype(np.int16).reindex(date=daily_dates, method=None, fill_value=np.int16(32767))


    obs_dates_xr = xr.DataArray(
        obs_dates,
        dims=("date",),
        coords={"date": daily_dates},
        name="obs_dates"
    )

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
    historical_ndvi = xr.open_zarr(historical_ndvi_src).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

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
    ds_to_stack = xr.open_zarr(OUT_ZARR_TMP).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})

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

    lookuptable = xr.open_zarr(lookuptable_src).chunk({
    "pixel": PIXEL_CHUNKS, 
    "doy": DATE_CHUNKS })

    doy = date_stack.dt.dayofyear
    doy_array_fixed = np.where(doy.values == 366, 365, doy.values)
    median_ndvi = lookuptable.sel(doy=xr.DataArray(doy_array_fixed, dims="date"))

    merged_ds = merged_ds.chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})


    # =====================================================
    # Build final dataset (unchanged)
    # =====================================================
    out_ds = xr.Dataset(
        {
            "ndvi": merged_ds["ndvi"],
            "median_ndvi": median_ndvi["median_ndvi"],
            "obs_date": merged_ds["obs_date"],
        },
        coords={
            "pixel": merged_ds.pixel,
            "date": date_stack,
            "x": merged_ds["x"],
            "y": merged_ds["y"],
        },
    )

    # Drop doy if present
    if "doy" in out_ds.coords:
        out_ds = out_ds.drop_vars("doy")
    if "doy" in out_ds.data_vars:
        out_ds = out_ds.drop_vars("doy")

    # =====================================================
    # V3-ONLY COMPRESSION 
    # =====================================================

    for var_name in ["ndvi", "median_ndvi"]:
        out_ds[var_name].encoding = {
            "dtype": "int16",
            "scale_factor": 0.001,
            "add_offset": 0,
            "_FillValue": -32768,
            "compressors": [ZstdCodec(level=15)]
        }

    out_ds["obs_date"].encoding = {
    "dtype": "bool",
    "compressors": [ZstdCodec(level=15)]
    }

    for coord in ["x", "y", "pixel", "date"]:
        if coord in out_ds.coords:
            out_ds[coord].encoding = {}

    # WRITE
    with dask.config.set({"array.chunk-size": "128MB"}):
        out_ds.to_zarr(
            OUT_ZARR,
            mode="w",
            consolidated=True,
            safe_chunks=False
        )
    print("Done")

    # clean
    client.close()
    shutil.rmtree(OUT_ZARR_TMP)
