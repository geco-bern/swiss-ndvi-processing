import numpy as np
import rasterio
from config import DATASET_ZARR, CHUNK_SIZE, FOREST_MASK, REF_BBOX, REF_BBOX_4326, SERVICE_URL
import zarr
import pystac_client
from tqdm import tqdm

service = pystac_client.Client.open(SERVICE_URL)
service.add_conforms_to("COLLECTIONS")
service.add_conforms_to("ITEM_SEARCH")

# Search for all annual height-model items
item_search = service.search(
    bbox=REF_BBOX_4326,
    collections=['ch.bafu.landesforstinventar-vegetationshoehenmodell_sentinel']
)
items = list(item_search.items())
# Leave out 2024 for testing
items = [item for item in items if '2024' not in item.id]
n_years = len(items)
forest_mask = np.load(FOREST_MASK)
forest_flat_indices = np.flatnonzero(forest_mask)
N = forest_flat_indices.size

# Stack forest‐pixel heights for each year
forest_heights = np.empty((n_years, N), dtype=float)
for idx, item in tqdm(enumerate(items)):
    asset = next(iter(item.assets.values()))
    with rasterio.open(asset.href) as src:
        window = src.window(*REF_BBOX)
        vh = src.read(1, window=window, boundless=True, fill_value=src.nodata)
    vh_flat = vh.ravel()
    forest_heights[idx] = vh_flat[forest_flat_indices]

# Compute median height per forest pixel
median_per_pixel = np.nanmedian(forest_heights, axis=0)

group = zarr.open_group(DATASET_ZARR, mode='a')
feat_grp = group.require_group('features')

feat_grp.create_array(
    name='median_forest_height',
    shape=(N,),
    dtype='float32',
    chunks=(CHUNK_SIZE,),
    overwrite=True
)

feat_grp['median_forest_height'][:] = median_per_pixel
