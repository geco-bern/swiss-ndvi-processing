# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py> /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tif.log 2>&1 &

import rasterio
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
#INPUT_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/05_processed/"
INPUT_BASE = "/mnt/data1/UniBe-swiss-ndvi/tmp_ndvi_05_processed.zarr"
OUTPUT_TIFF_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/tiffs/"
# forest_mask = "../../data/input/forest_mask.npy"


os.makedirs(OUTPUT_TIFF_BASE, exist_ok=True)


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

store2 = xarray.open_zarr(INPUT_BASE)

dates_done = [s[:8] for s in os.listdir(OUTPUT_TIFF_BASE)]

#curr_date = dates_tiff_array[0]
#for curr_date in dates_tiff_array:
curr_date_str = pd.to_datetime(curr_date).strftime('%Y%m%d')
if (curr_date_str in dates_done):
    print(f"Skipping file (already exported): {curr_date_str}.tiff")
    break

# Subst single date slice
ndvi_array = store2.sel(date=curr_date)['ndvi_processed']
mask_array = store2.sel(date=curr_date)['mask_array']
    # mask_array == 0: the data is not an observation and is yet to be smoothed
    # mask_array == 1: the data is not an observation and is smoothed
    # mask_array == 2: the data is an observation and is yet to be smoothed
    # mask_array == 3: the data is an observation and is smoothed
    # mask_array == 4: the data is an observation and is an outlier

x_array = store2['x']
y_array = store2['y']

# NOTE: previously we had something like: 
#   if (np.sum((mask_array.data.compute() == 3)|(mask_array.data.compute() == 1))/ len(mask_array.data)) > 0.9:

ds_tiff = xr.Dataset({
    'ndvi': (['pixel'], ndvi_array.data),
    'mask': (['pixel'], mask_array.data),
    'x': (['pixel'], x_array.data),
    'y': (['pixel'], y_array.data)
})

# # Metadata
# ds_tiff.attrs = {
#     'crs': 'EPSG:2056',
#     #'date_value': date, # TODO: this should be added again (as string)
#     'total_pixels': len(ds_tiff.pixel)
# }

# # dtypes
# encoding = {
#     'ndvi': {'zlib': True, 'complevel': 5, 'dtype': np.int16},
#     'mask': {'zlib': True, 'complevel': 5, 'dtype': np.int8},
#     'x': {'zlib': True, 'dtype': np.int32},
#     'y': {'zlib': True, 'dtype': np.int32}
# }

output_tiff = f"{OUTPUT_TIFF_BASE}{pd.to_datetime(curr_date).strftime('%Y%m%d')}.tiff"

# ds_tiff.to_netcdf(output_tiff, encoding=encoding) # Is this Tiff format? This looks more like netcdf. Where did they give specifications?

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# ----- center cooridnates  -----
center_x, center_y = 2694491.82, 1126023.20
# Rectangle corners (UL and BR)
UL_x, UL_y = center_x - 300, center_y - 300 
BR_x, BR_y = center_x + 300, center_y + 300


# ----- compute pixel window (row 0 = top) -----
x_min, x_max = min(UL_x, BR_x), max(UL_x, BR_x)
y_min, y_max = min(UL_y, BR_y), max(UL_y, BR_y)

col_min = int(np.floor((x_min - left) / px))
col_max = int(np.floor((x_max - left) / px))

row_min = int(np.floor((top - y_max) / px))
row_max = int(np.floor((top - y_min) / px))

# clip to bounds
col_min = max(0, min(width - 1, col_min))
col_max = max(0, min(width - 1, col_max))
row_min = max(0, min(height - 1, row_min))
row_max = max(0, min(height - 1, row_max))

win_cols = col_max - col_min + 1
win_rows = row_max - row_min + 1

# ----- load mask -----
forest_mask = "../../data/input/forest_mask.npy" # TODO: note this is not mask_array
mask = np.load(forest_mask)
assert mask.shape == (height, width), f"Mask shape {mask.shape} != raster {(height, width)}"
mask_array.shape
#mask.size
mask_flat = mask.ravel(order="C")
masked_positions = np.flatnonzero(mask_flat)
n_masked = masked_positions.size

# build index map from full array -> masked array
idx_map = np.full(mask_flat.shape[0], -1, dtype=np.int64)
idx_map[masked_positions] = np.arange(n_masked, dtype=np.int64)

# ----- compute flat indices in window -----
rows = np.arange(row_min, row_max + 1, dtype=np.int64)
cols = np.arange(col_min, col_max + 1, dtype=np.int64)
rr, cc = np.meshgrid(rows, cols, indexing="ij")
full_flat_idx = (rr * width + cc).ravel()

masked_idx_in_window = idx_map[full_flat_idx]
is_masked = masked_idx_in_window >= 0
n_masked_in_window = is_masked.sum()

sel = masked_idx_in_window[is_masked].tolist()

values = np.empty(n_masked_in_window, dtype=float)
window = np.full(win_rows * win_cols, np.nan, dtype=float)

window[is_masked] = 1 # TODO switch back to: ndvi_array.values # TODO: make this work (with subset)
window = window.reshape((win_rows, win_cols))

arr = np.nan_to_num(window, nan=-9999).astype('int16')

x_min = 2694491 - 300
y_max = 1126023 - 300
pixel_width = 10
pixel_height = 10

# TODO: activate tiff output: transform = from_origin(x_min, y_max, pixel_width, pixel_height) # TODO make this work
# TODO: activate tiff output: 
# TODO: activate tiff output: # TODO: define output_tiff = ... from 
# TODO: activate tiff output: with rasterio.open(
# TODO: activate tiff output:     output_tiff,
# TODO: activate tiff output:     'w',
# TODO: activate tiff output:     driver='COG',
# TODO: activate tiff output:     height=arr.shape[0],
# TODO: activate tiff output:     width=arr.shape[1],
# TODO: activate tiff output:     count=1,
# TODO: activate tiff output:     dtype=arr.dtype,
# TODO: activate tiff output:     crs='EPSG:2056', 
# TODO: activate tiff output:     # transform=transform, # TODO: make this work
# TODO: activate tiff output:     nodata=np.nan,
# TODO: activate tiff output:     compress='deflate',
# TODO: activate tiff output:     tiled=True
# TODO: activate tiff output: ) as dst:
# TODO: activate tiff output:     dst.write(arr, 1) 

#TODO: potentially also store mask_array as separate TIFF

print(f"Created {output_tiff}")




