# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py> /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tif.log 2>&1 &

import numpy as np
import xarray as xr
import zarr
import dask.array as da
import requests
import pandas as pd
import pystac_client

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
INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/05_processed_ndvi.zarr"
OUTPUT_NC = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/2018-06-01.nc"


store = zarr.open(INPUT_ZARR, mode='r')


start_date = "2018-01-01"
end_date = "2026-02-16"

dates_array = get_swisstopo_sentinel_dates(start=start_date, end=end_date)

start_tiff_date = dates_array[-4]
end_tiff_date = dates_array[-3]

target_date_idx = 180
print(f"Loading ONLY date index {target_date_idx}...")

# CORRECT SLICING for (pixels, dates) shape
ndvi_array = store['ndvi_processed']

# all pixels, date 365
ndvi_single_date = da.from_array(ndvi_array[:, target_date_idx], chunks=100000)
x_da = da.from_array(store['x'][:], chunks=100000)
y_da = da.from_array(store['y'][:], chunks=100000)
date_da = da.from_array(store['date'][:], chunks='auto')

mask_array = store['mask_array']

mask_array_single_date = da.from_array(mask_array[:, target_date_idx], chunks=100000)

# Build single-date Dataset
ds_single = xr.Dataset({
    'ndvi': (['pixel'], ndvi_single_date),
    "mask": (['pixel'], ndvi_single_date),
    'x': (['pixel'], x_da),
    'y': (['pixel'], y_da)
})

selected_date = int(date_da[target_date_idx].compute().item())
print(f"Date {target_date_idx} = day {selected_date}")
print(f"Pixels: {len(ds_single.pixel):,}")

# Metadata
ds_single.attrs = {
    'crs': 'EPSG:2056',
    'date_idx': target_date_idx,
    'date_value': selected_date,
    'total_pixels': len(ds_single.pixel)
}

# Your exact dtypes
encoding = {
    'ndvi': {'zlib': True, 'complevel': 5, 'dtype': np.int16},
    'mask': {'zlib': True, 'complevel': 5, 'dtype': np.int8},
    'x': {'zlib': True, 'dtype': np.int32},
    'y': {'zlib': True, 'dtype': np.int32}
}

ds_single.to_netcdf(OUTPUT_NC, encoding=encoding)
print("saved")
