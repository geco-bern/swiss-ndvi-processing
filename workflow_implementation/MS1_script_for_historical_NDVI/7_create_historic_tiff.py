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

# run this pythons script to create a TIFF (non-COG for the moment)
# python 7_create_historic_tiff.py "2025-08-22"


# Paths
INPUT_HISTORIC = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v4_compr_1000mX1000m.zarr"
OUTPUT_TIFF_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v4_compr_1000mX1000m"
DASK_TEMP_DIR    = "/mnt/data1/UniBe-swiss-ndvi/tmp_data/"

os.makedirs(OUTPUT_TIFF_BASE, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("date", help="Start date in YYYY-MM-DD")
args = parser.parse_args()

curr_date = args.date
# if running interactively use e.g.:
    # curr_date = "2024-07-22" # for dates requested...

curr_date = np.datetime64(curr_date, "D")


# Load (processed) NDVI data set to output specific dates
N_WORKERS = 10               # D: This takes: 1min55s for 50 pixels and 2min22s for 100 pixels and 5min43s for 500 pixels (when loading dates_array from disk)
MEMORY_PER_WORKER = "12GB"   # D: This takes: 1min55s for 50 pixels and 2min22s for 100 pixels and 5min43s for 500 pixels (when loading dates_array from disk)
client = Client(
    n_workers=N_WORKERS,
    threads_per_worker=1,
    processes=True,
    memory_limit=MEMORY_PER_WORKER,
    dashboard_address=":8345",
    local_directory= DASK_TEMP_DIR
)
print(client, flush = True)
print(client.dashboard_link, flush = True)


NDVI_historic = xr.open_zarr(INPUT_HISTORIC)

# NOTE: was needed for v3, obsolete for v4: # Define grid underlying PixelID and needed transformations
# NOTE: was needed for v3, obsolete for v4: # TODO: actually append this similarly to x and y as (x_idx, and y_idx) to historic data set already.
# NOTE: was needed for v3, obsolete for v4: #       this would simplify handling of historic data set
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # Raster info (of downloaded product)
# NOTE: was needed for v3, obsolete for v4: height, width = 24542, 37728
# NOTE: was needed for v3, obsolete for v4: left, bottom = 2474090.0, 1065110.0
# NOTE: was needed for v3, obsolete for v4: px = 10.0
# NOTE: was needed for v3, obsolete for v4: top = bottom + height * px
# NOTE: was needed for v3, obsolete for v4: # NOTE: 24542*37728 # = 925 Mio > 106 Mio pixel in ndvi_hist. This seems correct.
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # Define transform between row,col to coord (upper-left origin, pixel sizes)
# NOTE: was needed for v3, obsolete for v4: trans = rasterio.transform.from_origin(left, top, px, px)
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # Test how to use this transformation definition:
# NOTE: was needed for v3, obsolete for v4: # rasterio.transform.xy(trans, [0], [0])    # returns center coordinates of first upper left pixel
# NOTE: was needed for v3, obsolete for v4: # rasterio.transform.xy(trans, [10], [10])  # returns center coordinates of tenth upper left pixel (i.e. is more east and more south)
# NOTE: was needed for v3, obsolete for v4: # rasterio.transform.xy(trans, [0, 10], [0, 10]) # this goes from pixel index to coordinate
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # rows, cols = np.nonzero(forest_mask) # forest_mask = np.load(FOREST_MASK) # TODO: FOREST_MASK is unused
# NOTE: was needed for v3, obsolete for v4: # rows, cols = np.nonzero([[1, 0, 1],[0, 0, 0],[0, 0, 0]]) # returns [0 0] and [0 2]
# NOTE: was needed for v3, obsolete for v4: # ids = np.arange(len(rows))
# NOTE: was needed for v3, obsolete for v4: # xs, ys = rasterio.transform.xy(trans, rows, cols) # returns:  (array([2474095., 2474115.]), array([1310525., 1310525.]))
# NOTE: was needed for v3, obsolete for v4: # rasterio.transform.rowcol(trans, xs, ys)          # recovers: (array([0, 0]), array([0, 2]))
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # coord_lookup = pd.DataFrame({
# NOTE: was needed for v3, obsolete for v4: #     'pixelID': ids,
# NOTE: was needed for v3, obsolete for v4: #     'x': xs,
# NOTE: was needed for v3, obsolete for v4: #     'y': ys
# NOTE: was needed for v3, obsolete for v4: # })
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # coords = list(zip(xs, ys))
# NOTE: was needed for v3, obsolete for v4: # plt.plot(xs, ys)
# NOTE: was needed for v3, obsolete for v4: # plt.plot(coords)
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: # Get pixelID locations of regular grid for Tiff generation
# NOTE: was needed for v3, obsolete for v4: rows, cols = rasterio.transform.rowcol(
# NOTE: was needed for v3, obsolete for v4:     trans, 
# NOTE: was needed for v3, obsolete for v4:     NDVI_historic.x.values, 
# NOTE: was needed for v3, obsolete for v4:     NDVI_historic.y.values)
# NOTE: was needed for v3, obsolete for v4: 
# NOTE: was needed for v3, obsolete for v4: height = rows.max() + 1
# NOTE: was needed for v3, obsolete for v4: width = cols.max() + 1

# With v4 we now have access to trans in the NDVI_historic.not:
# execute in isolated namespace
code_str = textwrap.dedent(NDVI_historic.note).lstrip("\n")
ns = { # namespace (including globals the executed code can see)
    "rasterio": rasterio,
}
compiled = compile(code_str, "<NDVI_historic.note>", "exec")
exec(compiled, ns)
trans = ns.get("trans")

# With v4 we now have access to x_idx and y_idx:
# Define tiff-size:
rows = NDVI_historic.y_idx.values
cols = NDVI_historic.x_idx.values
height = rows.max() + 1 # TODO: actually would need to define window...
width = cols.max() + 1  # TODO: actually would need to define window...

# Run tiff-generation for requested date
dates_done = [s[:8] for s in os.listdir(OUTPUT_TIFF_BASE)]

for curr_date in [curr_date]:
    curr_date_str = pd.to_datetime(curr_date).strftime('%Y%m%d')
    if (curr_date_str in dates_done):
        print(f"Skipping file (already exported): {curr_date_str}.tiff")
    else:
        # Initialize regular grid filled with NaN for tiff to be filled with values
        grid_ndvi = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16
        grid_mask = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16

        # Fill regular grid with values
        grid_ndvi[rows, cols] = NDVI_historic.sel(date=curr_date)['ndvi_processed'].values
        grid_mask[rows, cols] = NDVI_historic.sel(date=curr_date)['mask_array'].values

        # Transform back into a xarray/rioxarray DataArray that spans a regular x-y-grid
        NDVI_processed_curr_date_gridded = xr.DataArray(grid_ndvi,dims=("y", "x"))
        NDVI_processed_curr_date_gridded = NDVI_processed_curr_date_gridded.rio.write_transform(trans)
        NDVI_processed_curr_date_gridded = NDVI_processed_curr_date_gridded.rio.write_crs("EPSG:2056")

        NDVI_status_curr_date_gridded = xr.DataArray(grid_mask,dims=("y", "x"))
        NDVI_status_curr_date_gridded = NDVI_status_curr_date_gridded.rio.write_transform(trans)
        NDVI_status_curr_date_gridded = NDVI_status_curr_date_gridded.rio.write_crs("EPSG:2056")

        # Output as cloud optimized Geotiff:
        output_tiff_ndvi = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}.tiff"
        output_tiff_mask = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}_mask.tiff"

        # TODO: check if this corresponds to: https://github.com/geostandards-ch/cog-best-practices#lossy-numerical-raster
        #       e.g. gdal_translate -a_srs EPSG:2056 -of COG -co COMPRESS=LERC_ZSTD -co LEVEL=22 -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co STATISTICS=YES -co MAX_Z_ERROR=<threshold> -tr <resolution in meter> <resolution in meter> -r Cubic -a_nodata <value> -ot <datatype> <input.tif> <output.tif>
        NDVI_processed_curr_date_gridded.rio.to_raster(
            output_tiff_ndvi,
            driver="COG",
            compress="deflate",
            dtype="int16",
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

        # This is for testing: we additionally produce normal GeoTiff
        output_tiff_ndvi2 = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}-nonCOG.tiff"
        output_tiff_mask2 = f"{OUTPUT_TIFF_BASE}/{pd.to_datetime(curr_date).strftime('%Y%m%d')}-nonCOG_mask.tiff"
        NDVI_processed_curr_date_gridded.rio.to_raster(output_tiff_ndvi2)
        NDVI_status_curr_date_gridded.rio.to_raster(output_tiff_mask2)

# rsync 
# rsync -avhz --progress -e 'ssh -p 2222' fabian-bernhard@dac3.ddns.net:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v3_compr/20240722-nonCOG.tiff ~/Downloads/test/tiffs_historic/
# rsync -avhz --progress -e 'ssh -p 2222' fabian-bernhard@dac3.ddns.net:/mnt/data1/UniBe-swiss-ndvi/data/tiffs_historic_v3_compr/20240722.tiff ~/Downloads/test/tiffs_historic/