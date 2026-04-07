# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py> /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tif.log 2>&1 &

import rasterio
import rioxarray
import numpy as np
import xarray as xr
import dask.array as da
import requests
import pandas as pd
import pystac_client
import os
import argparse

# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_tiff.log 2>&1 &
def get_swisstopo_sentinel_dates(start, end): # start='2025-12-01', end='2026-02-16' # TODO: we already have a similar function => consolidate

    # Connect to Swisstopo STAC API
    service = pystac_client.Client.open('https://data.geo.admin.ch/api/stac/v0.9/')
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    bbox_swiss_4326 = [5.70, 45.8, 10.6, 47.95]

    item_search = service.search(
        bbox=bbox_swiss_4326,
        datetime=f'{start}/{end}',
        collections=['ch.swisstopo.swisseo_s2-sr_v100']
    )
    s2_files = list(item_search.items())

    dates = []
    for item in s2_files:
        assets = item.assets
        asset_key_metadata = next((key for key in assets.keys() if key.endswith('metadata.json')), None)
        metadata_asset = assets[asset_key_metadata]
        json_link_metadata = metadata_asset.href
        response = requests.get(json_link_metadata)
        metadata_json = response.json()
        dates.append(metadata_json['BANDS-10M']['SOURCE_COLLECTION_PROPERTIES']['date'])

    # Convert to datetime64[D] array
    pd_dates = pd.to_datetime(dates)
    dates_array = pd_dates.values.astype('datetime64[D]')

    return dates_array


# Paths
#INPUT_BASE = "/mnt/data2/UniBe-swiss-ndvi/data/demo_all_pixel/05_processed/"
INPUT_BASE = "/mnt/data2/UniBe-swiss-ndvi/tmp_ndvi_05_processed.zarr"
OUTPUT_TIFF_BASE = "/mnt/data2/UniBe-swiss-ndvi/data/tiffs/"

os.makedirs(OUTPUT_TIFF_BASE, exist_ok=True)

# ssh dash-WS02-GECO; rsync -avhz --progress -e 'ssh -p 2222' /data_2/scratch/sbiegel/processed/forest_mask.npy fabian-bernhard@tunder.dev.admin.ch:/mnt/data2/UniBe-swiss-ndvi/data/forest_mask.npy
# FOREST_MASK = "/mnt/data2/UniBe-swiss-ndvi/data/forest_mask.npy" # NOTE: this is different from 'mask_array'  # TODO: FOREST_MASK is unused


parser = argparse.ArgumentParser()
parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
parser.add_argument("end_date", help="End date in YYYY-MM-DD")
args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date
# if running interactively use e.g.:
    # start_date = "2025-08-22" # for dates requested...
    # end_date = "2025-09-15"   # ...in script 1 when downloading

    # start_date = "2025-11-30" # for dates requested...
    # end_date = "2026-03-10"   # ...in script 1 when downloading

start_date = np.datetime64(start_date, "D")
end_date = np.datetime64(end_date, "D")

dates_array = get_swisstopo_sentinel_dates(start=start_date, end=end_date)
dates_array = np.sort(np.unique(dates_array))

# Tiff generation lags behind:
    # Tiff is created when an observation has at least three value before and after, 
    # thus tiff generated between the fifth-to-last and fourth-to-last obs. date specified
start_tiff_date = dates_array[-5] # fifth-to-last
end_tiff_date = dates_array[-4]   # fourth-to-last
# NOTE Fabian: Francesco, does that mean we do not anymore impose a threshold of valid pixels?

dates_tiff_array = np.arange(
    start_tiff_date, 
    end_tiff_date + np.timedelta64(1, 'D'), 
    dtype='datetime64[D]')





# Load (processed) NDVI data set to output specific dates
NDVI_processed = xr.open_zarr(INPUT_BASE)

# Define grid underlying PixelID and needed transformations
# TODO: actually append this similarly to x and y as (x_idx, and y_idx) to historic data set already.
#       this would simplify handling of historic data set

# Raster info (of downloaded product)
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px
# NOTE: 24542*37728 # = 925 Mio > 106 Mio pixel in ndvi_hist. This seems correct.

# Define transform between row,col to coord (upper-left origin, pixel sizes)
trans = rasterio.transform.from_origin(left, top, px, px)

# Test how to use this transformation definition:
# rasterio.transform.xy(trans, [0], [0])    # returns center coordinates of first upper left pixel
# rasterio.transform.xy(trans, [10], [10])  # returns center coordinates of tenth upper left pixel (i.e. is more east and more south)
# rasterio.transform.xy(trans, [0, 10], [0, 10]) # this goes from pixel index to coordinate

# rows, cols = np.nonzero(forest_mask) # forest_mask = np.load(FOREST_MASK) # TODO: FOREST_MASK is unused
# rows, cols = np.nonzero([[1, 0, 1],[0, 0, 0],[0, 0, 0]]) # returns [0 0] and [0 2]
# ids = np.arange(len(rows))
# xs, ys = rasterio.transform.xy(trans, rows, cols) # returns:  (array([2474095., 2474115.]), array([1310525., 1310525.]))
# rasterio.transform.rowcol(trans, xs, ys)          # recovers: (array([0, 0]), array([0, 2]))

# coord_lookup = pd.DataFrame({
#     'pixelID': ids,
#     'x': xs,
#     'y': ys
# })

# coords = list(zip(xs, ys))
# plt.plot(xs, ys)
# plt.plot(coords)





# Loop over dates
dates_done = [s[:8] for s in os.listdir(OUTPUT_TIFF_BASE)]

for curr_date in dates_tiff_array:
    curr_date_str = pd.to_datetime(curr_date).strftime('%Y%m%d')
    if (curr_date_str in dates_done):
        print(f"Skipping file (already exported): {curr_date_str}.tiff")
    else:
        # Initialize regular grid filled with NaN for tiff to be filled with values
        rows, cols = rasterio.transform.rowcol(
            trans, 
            NDVI_processed.x.values, 
            NDVI_processed.y.values)

        height = rows.max() + 1
        width = cols.max() + 1
        grid_ndvi = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16
        grid_mask = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16

        # Fill regular grid with values
        grid_ndvi[rows, cols] = NDVI_processed.sel(date=curr_date)['ndvi_processed'].values
        grid_mask[rows, cols] = NDVI_processed.sel(date=curr_date)['mask_array'].values

        # Transform back into a xarray/rioxarray DataArray that spans a regular x-y-grid
        NDVI_processed_curr_date_gridded = xr.DataArray(grid_ndvi,dims=("y", "x"))
        NDVI_processed_curr_date_gridded = NDVI_processed_curr_date_gridded.rio.write_transform(trans)
        NDVI_processed_curr_date_gridded = NDVI_processed_curr_date_gridded.rio.write_crs("EPSG:2056")

        NDVI_status_curr_date_gridded = xr.DataArray(grid_mask,dims=("y", "x"))
        NDVI_status_curr_date_gridded = NDVI_status_curr_date_gridded.rio.write_transform(trans)
        NDVI_status_curr_date_gridded = NDVI_status_curr_date_gridded.rio.write_crs("EPSG:2056")

        # Output as cloud optimized Geotiff:
        output_tiff_ndvi = f"{OUTPUT_TIFF_BASE}{pd.to_datetime(curr_date).strftime('%Y%m%d')}.tiff"
        output_tiff_mask = f"{OUTPUT_TIFF_BASE}{pd.to_datetime(curr_date).strftime('%Y%m%d')}_mask.tiff"

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


# rsync 
# rsync -avh --progress -e 'ssh -p 2222' fabian-bernhard@dac3.ddns.net:/mnt/data1/UniBe-swiss-ndvi/data/tiffs/20250829.tiff ~/Downloads/test/
# rsync -avh --progress -e 'ssh -p 2222' fabian-bernhard@dac3.ddns.net:/mnt/data1/UniBe-swiss-ndvi/data/tiffs/20250829.tiff ~/Downloads/test/
# gdal_translate -of PNG -scale Downloads/test/20250829.tiff Downloads/test/20250829.tiff.png
# open Downloads/test/20250829.tiff.png
