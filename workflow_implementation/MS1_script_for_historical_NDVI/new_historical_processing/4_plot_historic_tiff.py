# This plots maps of selected (all) time steps
 
import re
import rasterio
import textwrap
import rioxarray
import numpy as np
import xarray as xr
import dask.array as da
import pandas as pd
import os
import argparse
from dask.distributed import Client

# run this python script to create a TIFF
# source "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate"
# python 7_create_historic_tiff.py "2025-08-22"
# python 7_create_historic_tiff.py "all_dates"

if __name__ == '__main__':

    # Paths
    OUTPUT_TIFF_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7"
    # DASK_TEMP_DIR    = "/mnt/data1/UniBe-swiss-ndvi/tmp_data2/"

    os.makedirs(OUTPUT_TIFF_BASE, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_HISTORIC",  help="Full path to Zarr folder with historic NDVI data")
    parser.add_argument("date",            help="Start date in YYYY-MM-DD or then 'all_dates'")
    args = parser.parse_args()

    args = parser.parse_args()

    INPUT_HISTORIC = args.INPUT_HISTORIC
    date_arg       = args.date
    # if running interactively use e.g.:
        # date_arg = "2024-07-22" # for dates requested...
        # date_arg = "2024-07-31" # for dates requested...
        # date_arg = "all_dates"
        # INPUT_HISTORIC   = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7-test6.zarr" # TODO: remove -test


    # Load (processed) NDVI data set to output specific dates
    N_WORKERS = 10               # D: This takes: 1min55s for 50 pixels and 2min22s for 100 pixels and 5min43s for 500 pixels (when loading dates_array from disk)
    MEMORY_PER_WORKER = "12GB"   # D: This takes: 1min55s for 50 pixels and 2min22s for 100 pixels and 5min43s for 500 pixels (when loading dates_array from disk)

    with Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_PER_WORKER,
        dashboard_address=":8346",
        # local_directory= DASK_TEMP_DIR
        ) as client:
    
        print(client, flush = True)
        print(client.dashboard_link, flush = True)


        NDVI_historic = xr.open_zarr(INPUT_HISTORIC)

        # Starting from v4 of our format we have access to trans in the NDVI_historic.note:
        # Instructions are stored: in: NDVI_historic.transform_instr, telling you to do:
        from affine import Affine; 
        trans = Affine(*NDVI_historic.attrs['transform_coeffs'][0:6])
        
        # Starting from v4 we have access to x_idx and y_idx.
        # Compute minimal bounding window that contains all pixels, then shift indices
        # so the grid uses a compact array sized to that window.
        cols = NDVI_historic.y_idx.values.astype(int) # TODO: this fixes to rotation, but ideally 
        rows = NDVI_historic.x_idx.values.astype(int) # TODO: this fixes to rotation, but ideally 
        #y_coords = NDVI_historic.y.values.astype(int) # NOTE: this is correct, not affected by the rotation
        #x_coords = NDVI_historic.x.values.astype(int) # NOTE: this is correct, not affected by the rotation

        # bounding box in full-raster coordinates
        min_row = int(rows.min())
        min_col = int(cols.min())
        max_row = int(rows.max())
        max_col = int(cols.max())

        # size of the output window
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        # local indices inside the compact window
        local_rows = (rows - min_row).astype(int)
        local_cols = (cols - min_col).astype(int)

        # Compute affine transform for the compact window from the actual x/y coordinates
        if trans is None:
            raise RuntimeError("affine transform 'trans' not found in NDVI_historic.note and is required")

        # derive ordered unique coordinates and pixel size
        ux = np.unique(NDVI_historic.x.values)
        uy = np.unique(NDVI_historic.y.values)
        if ux.size < 2 or uy.size < 2:
            raise RuntimeError("Not enough unique x/y coordinates to derive pixel size")
        dx = float(ux[1] - ux[0])
        dy = float(uy[1] - uy[0])

        # origin for Affine.from_origin is top-left: (min_x - dx/2, max_y + dy/2)
        origin_x = float(ux.min() - dx / 2.0)
        origin_y = float(uy.max() + dy / 2.0)

        # create transform with pixel-size in x (dx) and negative y (so row index increases downward)
        window_trans = rasterio.Affine(dx, 0.0, origin_x, 0.0, -dy, origin_y)

        # Run tiff-generation for requested date
        dates_done = [s[:8] for s in os.listdir(OUTPUT_TIFF_BASE)]

        if date_arg == "all_dates": 
            curr_date_to_loop = [np.datetime64(d, "D") for d in NDVI_historic['date'].values]
        else:
            # date_arg = "2017-06-19" # NOTE: this date shows gapfilled data (in region Schaffhausen)
            # date_arg = "2017-06-02" # NOTE: this date shows observed data (in region Schaffhausen)
            # date_arg = "21-06-19"   # NOTE: this would trigger the wrong format warning
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_arg), f"Expected format YYYY-MM-DD with digits only. Received date_arg: {date_arg}"
            # curr_date_to_loop = [np.datetime64("2024-01-01", "D")]
            curr_date_to_loop = [np.datetime64(date_arg, "D")]

        # curr_date = curr_date_to_loop[0]
        for curr_date in curr_date_to_loop:
            curr_date_str = pd.to_datetime(curr_date).strftime('%Y%m%d')
            if (curr_date_str in dates_done):
                print(f"Skipping file (already exported): {curr_date_str}_historic.tiff")
            else:
                # Initialize compact window grid filled with NaN for tiff to be filled with values
                grid_ndvi = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16
                grid_mask = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16

                # Fill compact window grid using local indices
                grid_ndvi[local_rows, local_cols] = NDVI_historic.sel(date=curr_date)['ndvi_processed'].values
                grid_mask[local_rows, local_cols] = NDVI_historic.sel(date=curr_date)['mask_array'].values
                    # mask_array == 0: the data is not an observation and is yet to be smoothed
                    # mask_array == 1: the data is not an observation and is smoothed
                    # mask_array == 2: the data is an observation and is yet to be smoothed
                    # mask_array == 3: the data is an observation and is smoothed
                    # mask_array == 4: the data is an observation and is an outlier

                # Transform back into a xarray/rioxarray DataArray that spans the compact x-y-grid
                NDVI_processed_curr_date_gridded = xr.DataArray(grid_ndvi, dims=("y", "x"))
                NDVI_processed_curr_date_gridded = NDVI_processed_curr_date_gridded.rio.write_transform(window_trans)
                NDVI_processed_curr_date_gridded = NDVI_processed_curr_date_gridded.rio.write_crs("EPSG:2056")

                NDVI_status_curr_date_gridded = xr.DataArray(grid_mask, dims=("y", "x"))
                NDVI_status_curr_date_gridded = NDVI_status_curr_date_gridded.rio.write_transform(window_trans)
                NDVI_status_curr_date_gridded = NDVI_status_curr_date_gridded.rio.write_crs("EPSG:2056")

                # Output as cloud optimized Geotiff:
                output_tiff_ndvi = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}_historic.tiff"
                output_tiff_mask = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}_historic_mask.tiff"

                # NOTE: this should correspond to: https://github.com/geostandards-ch/cog-best-practices#lossy-numerical-raster
                #       e.g. gdal_translate -a_srs EPSG:2056 -of COG -co COMPRESS=LERC_ZSTD -co LEVEL=22 -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co STATISTICS=YES -co MAX_Z_ERROR=<threshold> -tr <resolution in meter> <resolution in meter> -r Cubic -a_nodata <value> -ot <datatype> <input.tif> <output.tif>
                NDVI_processed_curr_date_gridded.rio.to_raster(
                    output_tiff_ndvi,
                    driver="COG",
                    compress="deflate",
                    dtype="int16"
                )
                NDVI_status_curr_date_gridded.rio.to_raster(
                    output_tiff_mask,
                    driver="COG",
                    compress="deflate",
                    dtype="int16",
                )
                # mask_array == 0: the data is not an observation and is yet to be smoothed
                # mask_array == 1: the data is not an observation and is smoothed
                # mask_array == 2: the data is an observation and is yet to be smoothed
                # mask_array == 3: the data is an observation and is smoothed
                # mask_array == 4: the data is an observation and is an outlier

                print(f"Created {output_tiff_ndvi}")
                print(f"Created {output_tiff_mask}")

                # # This is for testing: we additionally produce normal GeoTiff
                # output_tiff_ndvi2 = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}-nonCOG_historic.tiff"
                # output_tiff_mask2 = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}-nonCOG_historic_mask.tiff"
                # NDVI_processed_curr_date_gridded.rio.to_raster(output_tiff_ndvi2)
                # NDVI_status_curr_date_gridded.rio.to_raster(output_tiff_mask2)

    # rsync 
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v3_compr/20240722-nonCOG_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v3_compr/20240722_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_compr_1000mX1000m/20240722_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_compr_1000mX1000m/20240722-nonCOG_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_compr_1000mX1000m/20240731_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_compr_1000mX1000m/20240731-nonCOG_historic.tiff ~/Downloads/test/tiffs_historic/

    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7_compr/20240731_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7_compr/20240731_historic_mask.tiff ~/Downloads/test/tiffs_historic/
