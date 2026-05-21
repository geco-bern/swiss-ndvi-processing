# This plots maps of selected (all) time steps
 
import re
import rasterio
import textwrap
import rioxarray
import numpy as np
import xarray as xr
import dask.array as da
import pandas as pd
from datetime import datetime, date, timedelta
import os
import argparse
from dask.distributed import Client

# NOTE: below only works with working directory at /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing
# from workflow_implementation.MS1_script_for_historical_NDVI.new_historical_processing.NDVI_utils.NDVI_plot_utils import NDVI_xarray_to_grid
# NOTE: below only works with working directory at /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_historical_processing
from NDVI_utils.NDVI_plot_utils import NDVI_xarray_to_grid

# run this python script to create a TIFF
# source "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate"
# python 7_create_historic_tiff.py "2025-08-22"
# python 7_create_historic_tiff.py "all_dates"

if __name__ == '__main__':

    # Paths
    OUTPUT_TIFF_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7final"
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
        # INPUT_HISTORIC   = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7-test10.zarr" # TODO: remove -test
        # INPUT_HISTORIC   = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr"
        # INPUT_HISTORIC   = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c.zarr"


    # Load (processed) NDVI data set to output specific dates
    N_WORKERS = 60               # D: This takes: 1min55s for 50 pixels and 2min22s for 100 pixels and 5min43s for 500 pixels (when loading dates_array from disk)
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
                NDVI_processed_curr_date_gridded = NDVI_xarray_to_grid(NDVI_historic, curr_date, variable = 'ndvi_processed')
                NDVI_status_curr_date_gridded    = NDVI_xarray_to_grid(NDVI_historic, curr_date, variable = 'mask_array')

                # Output as cloud optimized Geotiff:
                output_tiff_ndvi = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}_historic.tiff"
                output_tiff_mask = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}_historic_mask.tiff"

                # NOTE: this should correspond to: https://github.com/geostandards-ch/cog-best-practices#lossy-numerical-raster
                #       e.g. gdal_translate -a_srs EPSG:2056 -of COG -co COMPRESS=LERC_ZSTD -co LEVEL=22 -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co STATISTICS=YES -co MAX_Z_ERROR=<threshold> -tr <resolution in meter> <resolution in meter> -r Cubic -a_nodata <value> -ot <datatype> <input.tif> <output.tif>

                # variant 1)
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
                
                # variant 2)
                # arr = NDVI_processed_curr_date_gridded.values.astype("int16")
                # profile = {
                #     "driver": "COG",
                #     "dtype": "int16",
                #     "count": 1,
                #     "height": arr.shape[0],
                #     "width": arr.shape[1],
                #     "crs": "EPSG:2056",
                #     "transform": window_trans,
                #     "compress": "LERC_ZSTD",
                #     "LEVEL": 22,
                #     "NUM_THREADS": "ALL_CPUS",
                #     "BIGTIFF": "YES",
                #     "MAX_Z_ERROR": 0.02,   # tune for acceptable lossy error
                # }
                # with rasterio.open(output_tiff_ndvi, "w", **profile) as dst:
                #     dst.write(arr, 1)

                # arr = NDVI_status_curr_date_gridded.values.astype("int16")
                # profile = {
                #     "driver": "COG",
                #     "dtype": "int16",
                #     "count": 1,
                #     "height": arr.shape[0],
                #     "width": arr.shape[1],
                #     "crs": "EPSG:2056",
                #     "transform": window_trans,
                #     "compress": "LERC_ZSTD",
                #     "LEVEL": 22,
                #     "NUM_THREADS": "ALL_CPUS",
                #     "BIGTIFF": "YES",
                #     "MAX_Z_ERROR": 0.02,   # tune for acceptable lossy error
                # }
                # with rasterio.open(output_tiff_mask, "w", **profile) as dst:
                #     dst.write(arr, 1)
                # mask_array == 0: the data is not an observation and is yet to be smoothed
                # mask_array == 1: the data is not an observation and is smoothed
                # mask_array == 2: the data is an observation and is yet to be smoothed
                # mask_array == 3: the data is an observation and is smoothed
                # mask_array == 4: the data is an observation and is an outlier

                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]",flush=True)
                print(f"Created {output_tiff_ndvi}",flush=True)
                print(f"Created {output_tiff_mask}",flush=True)

                # # This is for testing: we additionally produce normal (i.e. non-cloud-optimized) GeoTiff
                # output_tiff_ndvi2 = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}-nonCOG_historic.tiff"
                # output_tiff_mask2 = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}-nonCOG_historic_mask.tiff"
                # NDVI_processed_curr_date_gridded.rio.to_raster(output_tiff_ndvi2)
                # NDVI_status_curr_date_gridded.rio.to_raster(output_tiff_mask2)

    # rsync 
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v3/20240722-nonCOG_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v3/20240722_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_1000mX1000m/20240722_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_1000mX1000m/20240722-nonCOG_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_1000mX1000m/20240731_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_1000mX1000m/20240731-nonCOG_historic.tiff ~/Downloads/test/tiffs_historic/

    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7final/20240731_historic.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7final/20240731_historic_mask.tiff ~/Downloads/test/tiffs_historic/
    # rsync -avhz --progress -e 'ssh -p 22' fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v7final/ ~/Downloads/test/tiffs_historic/
