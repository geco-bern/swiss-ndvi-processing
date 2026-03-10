"""
Fetch acquisition dates from Swisstopo STAC API and add them to the Zarr dataset.
"""
# "Run Python File" in VSCode

import requests
import pandas as pd
import pystac_client
import zarr
import argparse

# Append dates to this (acts as in- and output zarr file)
IN_OUTPUT_ZARR_TEMP = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_02-03_downloadedB.zarr" # or store into /var/tmp/

#def get_swisstopo_sentinel_dates(start='2025-12-01', end='2026-02-16'):
def get_swisstopo_sentinel_dates(start, end): # start='2025-12-01', end='2026-02-16'
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
    pd_dates = pd.to_datetime(dates)
    pd_dates_str = pd_dates.strftime('%Y-%m-%d')

    root = zarr.open_group(IN_OUTPUT_ZARR_TEMP, mode='a', zarr_format=3)
    root.create_array(
        name='date',
        dtype='S10',
        shape=(len(pd_dates_str),),
        chunks=(len(pd_dates_str),),
    )
    root['date'][:] = pd_dates_str.values.astype('S10')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
    parser.add_argument("end_date", help="End date in YYYY-MM-DD")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    # if running interactively use e.g.:
    # start_date = "2025-11-30" # for dates requested...
    # end_date = "2026-03-10"   # ...in script 1 when downloading

    get_swisstopo_sentinel_dates(start=start_date, end=end_date)
    print("Dates added to Zarr dataset.")
