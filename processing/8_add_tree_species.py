import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import config
import zarr

tree_species_path = '/data/archive_restricted/treespecies_koch_2024/data_share_FORWARDS/tree_species_map_aoa_raster.tif'

with rasterio.open(tree_species_path) as src:
    src_crs = src.crs
    src_transform = src.transform
    src_dtype = 'uint8'
    src_nodata = src.nodata

    if src_nodata is None or np.isnan(src_nodata):
        src_nodata = 255

    src_data = src.read(1)
    src_data = np.where(np.isnan(src_data), src_nodata, src_data).astype(src_dtype)

dst_height = int(config.REF_HEIGHT)
dst_width = int(config.REF_WIDTH)
dst_transform = config.REF_TRANSFORM
dst_crs = config.REF_CRS

dst = np.full((dst_height, dst_width), src_nodata, dtype=src_dtype)

reproject(
    source=src_data,
    destination=dst,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.nearest,
    src_nodata=src_nodata,
    dst_nodata=src_nodata,
)

forest_mask = np.load(config.FOREST_MASK)
forest_flat_indices = np.flatnonzero(forest_mask)
N = forest_flat_indices.size

dst_flat = dst.ravel()[forest_flat_indices]

group = zarr.open_group(config.DATASET_ZARR, mode='a')
feat_grp = group.require_group('features')

feat_grp.create_array(
    name='tree_species',
    shape=(N,),
    dtype='uint8',
    fill_value=src_nodata,
    chunks=(config.CHUNK_SIZE,),
    overwrite=True
)

feat_grp['tree_species'][:] = dst_flat