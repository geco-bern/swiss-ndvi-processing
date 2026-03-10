# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py> /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tif.log 2>&1 &

import numpy as np
import xarray as xr
import zarr
import dask.array as da
import requests
import pandas as pd
import pystac_client
import os
import argparse


# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_tiff.log 2>&1 &
def get_swisstopo_sentinel_dates(start='2025-12-01', end='2026-02-16'):

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
INPUT_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/05_processed/"
OUTPUT_TIFF_BASE = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/tiffs/"

os.makedirs(OUTPUT_TIFF_BASE, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
parser.add_argument("end_date", help="End date in YYYY-MM-DD")
args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date

start_date = np.datetime64(start_date, "D")
end_date = np.datetime64(end_date, "D")

dates_array = get_swisstopo_sentinel_dates(start=start_date, end=end_date)

dates_array = np.sort(np.unique(dates_array))

start_tiff_date = dates_array[-5] # TODO: Fabian: why did you limit to only four (two?) dates? # NOTE: Francesco according to the workflow, we create the tiff when an observation has at least three value before and after, for this reason each time we'll create the tiff between the lsat foruth and last third obs. date obtained with the previous function
end_tiff_date = dates_array[-4] # TODO: Fabian: why did you limit to only four (two?) dates?

dates_tiff_array = np.arange(start_tiff_date, end_tiff_date + np.timedelta64(1, 'D'), dtype='datetime64[D]')

for date in dates_tiff_array:
    # Get year for input zarr
    year = pd.to_datetime(date).year
    input_zarr = f"{INPUT_BASE}{year}.zarr"

    store = zarr.open(input_zarr, mode='r')
    
    # Find date index in this year's data
    dates_store = store['date']
    date_idx = np.where((dates_store[:] == date).astype(bool))[0]

    date_idx = date_idx[0]
    
    # Load single date slice
    ndvi_array = store['ndvi_processed'][:, date_idx]
    mask_array = store['mask_array'][:, date_idx]
    x_array = store['x'][:]
    y_array = store['y'][:]

    ds_single = xr.Dataset({
        'ndvi': (['pixel'], ndvi_array),
        'mask': (['pixel'], mask_array),
        'x': (['pixel'], x_array),
        'y': (['pixel'], y_array)
    })

    # Metadata
    ds_single.attrs = {
        'crs': 'EPSG:2056',
        'date_value': date,
        'total_pixels': len(ds_single.pixel)
    }

    # dtypes
    encoding = {
        'ndvi': {'zlib': True, 'complevel': 5, 'dtype': np.int16},
        'mask': {'zlib': True, 'complevel': 5, 'dtype': np.int8},
        'x': {'zlib': True, 'dtype': np.int32},
        'y': {'zlib': True, 'dtype': np.int32}
    }
    output_tiff = f"{OUTPUT_TIFF_BASE}{pd.to_datetime(date).strftime('%Y%m%d')}.tiff"

    ds_single.to_netcdf(output_tiff, encoding=encoding)
    print(f"Created {output_tiff}")
    store.close()

