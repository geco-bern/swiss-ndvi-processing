"""
Add forest mix rate from Swiss National Forest Inventory to the dataset.
"""
import numpy as np
import rasterio
import zarr
import pystac_client

from config import TEMPORAL_DATASET_ZARR, CHUNK_SIZE, FOREST_MASK, REF_BBOX, REF_BBOX_4326, SERVICE_URL

service = pystac_client.Client.open(SERVICE_URL)
service.add_conforms_to("COLLECTIONS")
service.add_conforms_to("ITEM_SEARCH")

item_search = service.search(
    bbox=REF_BBOX_4326,
    collections=['ch.bafu.landesforstinventar-waldmischungsgrad']
)
item = list(item_search.items())[0]
forest_mask = np.load(FOREST_MASK)
forest_flat_indices = np.flatnonzero(forest_mask)
N = forest_flat_indices.size

asset = next(iter(item.assets.values()))
with rasterio.open(asset.href) as src:
    window = src.window(*REF_BBOX)
    wm = src.read(1, window=window, boundless=True, fill_value=src.nodata)
    src_nodata = src.nodata
wm_flat = wm.ravel()
wm_flat_forest = wm_flat[forest_flat_indices]
wm_flat_forest[wm_flat_forest == src_nodata] = -1

group = zarr.open_group(TEMPORAL_DATASET_ZARR, mode='a')
feat_grp = group.require_group('features')

feat_grp.create_array(
    name='forest_mix_rate',
    shape=(N,),
    dtype='int8',
    chunks=(CHUNK_SIZE,),
    fill_value=src_nodata,
    overwrite=True
)

feat_grp['forest_mix_rate'][:] = wm_flat_forest
feat_grp['forest_mix_rate'].attrs['nodata'] = src_nodata
