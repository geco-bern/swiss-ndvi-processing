#  nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/00_donwload_swisstopo.py >  /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/donwload_swisstopo.log &

"""
Extract Swisstopo Sentinel-2 dataset for Switzerland and compute NDVI and NDSI time series for forested areas.
"""
import pystac_client
import rasterio
from rasterio.coords import BoundingBox
import numpy as np
import zarr
import numcodecs
from tqdm import tqdm
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import shutil


# Connect to Swisstopo STAC API
service = pystac_client.Client.open('https://data.geo.admin.ch/api/stac/v0.9/')
service.add_conforms_to("COLLECTIONS")
service.add_conforms_to("ITEM_SEARCH")

# EPSG: 4326
# WGS 84
# Swiss bounds: left, bottom, right, top
bbox_swiss_4326 = [5.70, 45.8, 10.6, 47.95]

src = "/data_3/scratch/francesco/download_zarr_prova"
dst = "/data_3/scratch/francesco/download_zarr_prova_v2"

shutil.rmtree(dst, ignore_errors=True)
shutil.rmtree(src, ignore_errors=True)

# Retrieve the spatial coverage (bounds) of all 4 possible orbits covering Switzerland
def collect_bounds_all_orbits():
    """
    Collects the bounds of all orbits in the Swiss dataset.
    Returns a list of BoundingBox objects.
    """
    item_search = service.search(
        bbox=bbox_swiss_4326,
        datetime='2025-04-30/2025-05-02',
        collections=['ch.swisstopo.swisseo_s2-sr_v100']
    )
    s2_files_sample_orbits = list(item_search.items())

    all_bounds = []

    for item in tqdm(s2_files_sample_orbits):
        assets = item.assets
        key_bands = [k for k in assets.keys() if k.endswith('bands-10m.tif')][0]
        bands_asset = assets[key_bands]
        with rasterio.open(bands_asset.href) as src:
            bounds = src.bounds
            all_bounds.append(bounds)

    return all_bounds

# Combine all bounding boxes into one global bounding box and compute its pixel dimensions
def union_bounds(bounds_list):
    """
    Takes a list of BoundingBox objects and returns a single BoundingBox
    that encompasses all the bounds, along with the width and height
    of the bounding box in pixels, assuming a resolution of 10 meters.
    """
    left = min(b.left for b in bounds_list)
    bottom = min(b.bottom for b in bounds_list)
    right = max(b.right for b in bounds_list)
    top = max(b.top for b in bounds_list)
    resolution = 10
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)
    return BoundingBox(left, bottom, right, top), width, height

all_bounds = collect_bounds_all_orbits()

# EPSG: 2056
# Swiss coordinate system (CH1903+ / LV95)
# This is the full reference bounding box for the Swisstopo dataset covering the 4 orbits
bbox_swisstopo_2056, width_swisstopo, height_swisstopo = union_bounds(all_bounds)

# Take the forest mask from the Swisstopo VHI dataset 
# The VHI dataset contains the forest mask that Swisstopo derived from the habitat map
# Also collect the metadata using the forest mask as a reference raster
def get_forest_mask():
    """
    Downloads the forest mask from the Swisstopo VHI dataset.
    Returns a numpy array representing the forest mask.
    Also returns the metadata for the reference raster.
    """
    item_search = service.search(
        bbox=bbox_swiss_4326,
        datetime='2025-05-01/2025-05-01',
        collections=['ch.swisstopo.swisseo_vhi_v100']
    )
    items = list(item_search.items())
    item = items[0]
    assets = item.assets
    key_bands = [k for k in assets.keys() if k.endswith('forest-10m.tif')][0]
    bands_asset = assets[key_bands]
    
    with rasterio.open(bands_asset.href) as src:
        window = src.window(*bbox_swisstopo_2056)
        vhi = src.read(1, window=window)
        forest_mask = (vhi != 255).astype('uint8')
        ref_meta = {
            "transform": src.window_transform(window),
            "crs": src.crs,
            "width": window.width,
            "height": window.height
        }
    
    return forest_mask, ref_meta

forest_mask, ref_meta = get_forest_mask()
print("Reference raster metadata:")
print(ref_meta)

# Build index mapping from forest pixels in the full reference raster to 1D flat indices
forest_flat_indices = np.flatnonzero(forest_mask == 1)
max_index = forest_flat_indices.max() + 1
index_map = np.full(max_index, -1, dtype=np.int32)
index_map[forest_flat_indices] = np.arange(len(forest_flat_indices))

# Search all images for the full CH bounding box for the whole time period
item_search = service.search(
    bbox=bbox_swiss_4326,
    datetime='2018-01-01/2018-05-31',
    collections=['ch.swisstopo.swisseo_s2-sr_v100']
)
s2_files = list(item_search.items())

# Prepare constants
N = len(forest_flat_indices)
T = len(s2_files)
INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step

# Define the datasets for NDVI and NDSI values
# Shape is (T, N) where T is the number of time steps and N is the number of forest pixels
# Use int16 to save space, with a fill value for no coverage
# Use compression to save space
compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
ndvi_ds = zarr.create_array(
    name="ndvi",
    store='/data_3/scratch/francesco/download_zarr_prova',
    shape=(T, N),
    chunks=(1, N),
    dtype="int16",
    fill_value=NO_COVERAGE,
    compressors=compressors,
    zarr_format=3,
)

ndsi_ds = zarr.create_array(
    name="ndsi",
    store='/data_3/scratch/francesco/download_zarr_prova',
    shape=(T, N),
    chunks=(1, N),
    dtype="int16",
    fill_value=NO_COVERAGE,
    compressors=compressors,
    zarr_format=3,
)

failed_timesteps = []

def add_timestep_to_zarr(t, item):
    assets = item.assets
    bands10_asset = assets[[k for k in assets if k.endswith('bands-10m.tif')][0]]
    bands20_asset = assets[[k for k in assets if k.endswith('bands-20m.tif')][0]]
    masks_asset = assets[[k for k in assets if k.endswith('masks-10m.tif')][0]]

    with rasterio.open(bands10_asset.href) as b10_src, \
         rasterio.open(bands20_asset.href) as b20_src, \
         rasterio.open(masks_asset.href) as masks_src:

        # Handle alignment mismatches between bands and masks
        if not (
            (b10_src.transform == masks_src.transform) and
            (b10_src.width, b10_src.height) == (masks_src.width, masks_src.height)
        ):
            b10_window = from_bounds(*bbox_swisstopo_2056, transform=b10_src.transform)
            mask_window = from_bounds(*bbox_swisstopo_2056, transform=masks_src.transform)
            b20_window = from_bounds(*bbox_swisstopo_2056, transform=b20_src.transform)

            red, green, nir = b10_src.read([1, 2, 4], window=b10_window, boundless=True, fill_value=9999)
            swir = b20_src.read(3, window=b20_window, boundless=True, fill_value=9999)
            masks = masks_src.read([1, 2], window=mask_window, boundless=True, fill_value=255).astype("uint8")

        else:
            b10_window = b10_src.window(*bbox_swisstopo_2056)
            b20_window = b20_src.window(*bbox_swisstopo_2056)
            red, green, nir = b10_src.read([1, 2, 4], window=b10_window)
            swir = b20_src.read(3, window=b20_window)
            masks = masks_src.read([1, 2], window=b10_window).astype("uint8")

        terrain_mask, cloud_mask = masks
        cloud_shadows_mask = (terrain_mask == 100) | (cloud_mask == 1)
        nodata_mask_ndvi = (red == 9999) | (nir == 9999) | (terrain_mask == 255) | (cloud_mask == 255)

        # Compute NDVI
        red = red.astype("float32") / 10000.0
        nir = nir.astype("float32") / 10000.0
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        ndvi_scaled = (np.nan_to_num(ndvi, nan=NO_COVERAGE / 10000.0) * 10000.0).astype("int16")

        # Reproject SWIR to align with green band
        h, w = green.shape
        src_transform = b20_src.window_transform(b20_window)
        target_transform = b10_src.window_transform(b10_window)

        swir_10m = np.full((h, w), 9999, dtype=np.float32)
        reproject(
            source=swir,
            destination=swir_10m,
            src_transform=src_transform,
            src_crs=b20_src.crs,
            dst_transform=target_transform,
            dst_crs=b10_src.crs,
            resampling=Resampling.bilinear,
            src_nodata=9999,
            dst_nodata=9999
        )

        nodata_mask_ndsi = (green == 9999) | (swir_10m == 9999) | (terrain_mask == 255) | (cloud_mask == 255)

        # Compute NDSI
        green = green.astype("float32") / 10000.0
        swir_10m = swir_10m.astype("float32") / 10000.0
        ndsi = (green - swir_10m) / (green + swir_10m)
        ndsi = np.clip(ndsi, -1.0, 1.0)
        ndsi_scaled = (np.nan_to_num(ndsi, nan=NO_COVERAGE / 10000.0) * 10000.0).astype("int16")

    # Window for slicing forest mask and index map
    window = from_bounds(*b10_src.bounds, transform=ref_meta["transform"]).round_offsets().round_lengths()
    row_start, row_stop = window.row_off, window.row_off + window.height
    col_start, col_stop = window.col_off, window.col_off + window.width

    local_forest_mask = forest_mask[row_start:row_stop, col_start:col_stop]
    local_rows, local_cols = np.where(local_forest_mask)

    global_rows = local_rows + row_start
    global_cols = local_cols + col_start
    global_flat = global_rows * width_swisstopo + global_cols
    current_flat_indices = index_map[global_flat]

    # Flat masks
    cloud_shadows_mask_flat = cloud_shadows_mask[local_rows, local_cols]
    nodata_mask_flat_ndvi = nodata_mask_ndvi[local_rows, local_cols]
    valid_ndvi = ~(cloud_shadows_mask_flat | nodata_mask_flat_ndvi)
    cloud_only_ndvi = cloud_shadows_mask_flat & ~nodata_mask_flat_ndvi
    nodata_mask_flat_ndsi = nodata_mask_ndsi[local_rows, local_cols]
    valid_ndsi = ~(cloud_shadows_mask_flat | nodata_mask_flat_ndsi)
    cloud_only_ndsi = cloud_shadows_mask_flat & ~nodata_mask_flat_ndsi

    # Write NDVI
    ndvi_flat = ndvi_scaled[local_rows, local_cols]
    ndvi_row = np.full(N, NO_COVERAGE, dtype="int16")
    ndvi_row[current_flat_indices[valid_ndvi]] = ndvi_flat[valid_ndvi]
    ndvi_row[current_flat_indices[cloud_only_ndvi]] = INVALID
    ndvi_ds[t] = ndvi_row

    # Write NDSI
    ndsi_flat = ndsi_scaled[local_rows, local_cols]
    ndsi_row = np.full(N, NO_COVERAGE, dtype="int16")
    ndsi_row[current_flat_indices[valid_ndsi]] = ndsi_flat[valid_ndsi]
    ndsi_row[current_flat_indices[cloud_only_ndsi]] = INVALID
    ndsi_ds[t] = ndsi_row

ndvi_ds.attrs['description'] = 'NDVI (scaled int16: -10000 to 10000)'
ndsi_ds.attrs['description'] = 'NDSI (scaled int16: -10000 to 10000)'
ndvi_ds.attrs['nodata'] = NO_COVERAGE
ndvi_ds.attrs['cloud_shadow'] = INVALID

for t, path in tqdm(enumerate(s2_files), total=len(s2_files)):
    try:
        add_timestep_to_zarr(t, path)
        print(f"Time step {t} processed successfully.")
    except Exception as e:
        print(f"Time step {t} failed: {e}")
        failed_timesteps.append((t, path))
        continue  # skip to the next time step

# Retry the failed time steps
if failed_timesteps:
    print(f"Retrying {len(failed_timesteps)} failed time steps...")
    for t, path in tqdm(failed_timesteps):
        try:
            add_timestep_to_zarr(t, path)
            print(f"Time step {t} retried successfully.")
        except Exception as e:
            print(f"Time step {t} retry failed: {e}")
            continue  # skip to the next time step


"""
Fetch acquisition dates from Swisstopo STAC API and add them to the Zarr dataset.
"""
import requests
import pandas as pd
import pystac_client
import zarr

def get_swisstopo_sentinel_dates(start='2018-04-01', end='2028-05-31'):
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

    root = zarr.open_group("/data_3/scratch/francesco/download_zarr_prova", mode='a', zarr_format=3)
    root.create_array(
        name='dates',
        dtype='S10',
        shape=(len(pd_dates_str),),
        chunks=(len(pd_dates_str),),
    )
    root['dates'][:] = pd_dates_str.values.astype('S10')
    
get_swisstopo_sentinel_dates(start='2018-04-01', end='2028-05-31')
print("Dates added to Zarr dataset.")

# convert from zarr v3 to zarr v2 (necessary because xarray do not supprort zarr v3)

import shutil
import zarr
import numpy as np
import xarray as xr
from datetime import datetime, date

src = "/data_3/scratch/francesco/download_zarr_prova"
dst = "/data_3/scratch/francesco/download_zarr_prova_v2"


# Remove dst if it exists
shutil.rmtree(dst, ignore_errors=True)

import os
import math
import itertools
import zarr
from zarr.storage import LocalStore   # <= THIS IS CORRECT FOR ZARR 3
import numcodecs
import numpy as np

src_root = "/data_3/scratch/francesco/download_zarr_prova"
dst_root = "/data_3/scratch/francesco/download_zarr_v2"

vars_to_copy = ["ndvi", "ndsi", "dates"]

os.makedirs(dst_root, exist_ok=True)

# Open v3 arrays
src_arrays = {}
for v in vars_to_copy:
    path = os.path.join(src_root, v)
    src_arrays[v] = zarr.open(path, mode="r")

# Create v2 store
store = LocalStore(dst_root)             # Zarr v3 local filesystem store
root = zarr.group(store=store, overwrite=True, zarr_version=2)

compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

def create_dst_array(name, src_a):
    shape = tuple(src_a.shape)
    chunks = src_a.chunks or None
    dtype = src_a.dtype
    fill_value = getattr(src_a, "fill_value", None)

    if chunks is None:
        if len(shape) == 1:
            chunks = (min(1024, shape[0]),)
        else:
            chunks = (1, min(65536, shape[-1]))

    dst = root.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        overwrite=True,
    )
    return dst

def chunked_copy(src_a, dst_a):
    shape = dst_a.shape
    chunks = dst_a.chunks
    ranges = [range(0, shape[i], chunks[i]) for i in range(len(shape))]
    total = math.prod([math.ceil(shape[i] / chunks[i]) for i in range(len(shape))])
    done = 0

    for start_indices in itertools.product(*ranges):
        slices = tuple(slice(s, min(s + chunks[i], shape[i])) for i, s in enumerate(start_indices))
        dst_a[slices] = np.asarray(src_a[slices])
        done += 1
        if done % 50 == 0 or done == total:
            print(f"{dst_a.name}: {done}/{total} chunks copied")

for name, src_a in src_arrays.items():
    print(f"Creating {name}, shape={src_a.shape}, chunks={src_a.chunks}")
    dst = create_dst_array(name, src_a)
    chunked_copy(src_a, dst)

print("DONE. Zarr v2 store is at:", dst_root)




# extended the dataset to be evenly spaced

ds = xr.open_zarr(dst)

first_date = date(2018,1,1)
last_date = date(2018,5,31)
