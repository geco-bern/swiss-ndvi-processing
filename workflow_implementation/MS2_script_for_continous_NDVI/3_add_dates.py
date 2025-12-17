"""
Fetch acquisition dates from Swisstopo STAC API and add them to the Zarr dataset.
"""
# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/3_add_dates.py >  /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/add_dates_swisstopo.log &


import requests
import pandas as pd
import pystac_client
import zarr

start_date = '2018-06-01'
end_date = '2018-06-05'

def get_swisstopo_sentinel_dates(start=start_date, end=end_date):
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

    root = zarr.open_group("/data_3/scratch/francesco/processed/ndvi_dataset_temporal.zarr", mode='a', zarr_format=3)
    root.create_array(
        name='date',
        dtype='S10',
        shape=(len(pd_dates_str),),
        chunks=(len(pd_dates_str),),
    )
    root['date'][:] = pd_dates_str.values.astype('S10')
    
if __name__ == "__main__":
    get_swisstopo_sentinel_dates()
    print("Dates added to Zarr dataset.")