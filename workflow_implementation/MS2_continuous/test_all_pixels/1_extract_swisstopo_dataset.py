"""
Extract Swisstopo Sentinel-2 dataset for Switzerland and compute NDVI and NDSI time series for forested areas.
"""
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import pystac_client
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import numpy as np
import zarr
from tqdm import tqdm
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import argparse
import os, shutil
import sys
from datetime import datetime
from affine import Affine
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

# HOW TO RUN FROM BASH:
# source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
# SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/1_extract_swisstopo_dataset.py"
# LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/1_extract_swisstopo_dataset_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"
# START_DATE="2026-01-01"
# END_DATE="2026-01-15"
# python -u $SCRIPT_FILE $START_DATE $END_DATE > $LOG_FILE  2>&1 &


# PARSE ARGUMENTS:
parser = argparse.ArgumentParser()
parser.add_argument("start_date", help="Start date in YYYY-MM-DD")
parser.add_argument("end_date", help="End date in YYYY-MM-DD")
args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date
# if running interactively use e.g.:
# start_date = "2025-11-30" # for the pipeline this should correspond to 
#                           # the last date in the historic NDVI data set
# end_date = "2026-03-22"
# end_date = "2025-12-04"
# end_date = "2025-12-06"
# end_date = "2025-12-12"
# start_date="2025-06-30"
# end_date="2025-07-01"

# CONFIGURE:
today = datetime.today().strftime("%Y-%m-%d_%Hh%M")
OUTPUT_ZARR_TEMP = f"/mnt/data2/UniBe-swiss-ndvi/data/tmp_{today}_ndvi_01_downloadedA_{start_date}_{end_date}.zarr"
OUTPUT_ZARR      = f"/mnt/data2/UniBe-swiss-ndvi/data/tmp_{today}_ndvi_01_downloaded_{start_date}_{end_date}.zarr"
# ==============================================================================

# Start script:
date_range = f"{start_date}/{end_date}"

# Connect to Swisstopo STAC API
service = pystac_client.Client.open('https://data.geo.admin.ch/api/stac/v0.9/')
service.add_conforms_to("COLLECTIONS")
service.add_conforms_to("ITEM_SEARCH")

# EPSG: 4326
# WGS 84
# Swiss bounds: left, bottom, right, top
bbox_swiss_4326 = [5.70, 45.8, 10.6, 47.95]

# Search all images for the full CH bounding box for the requested date range
item_search = service.search(
    bbox=bbox_swiss_4326,
    datetime = date_range,               
    collections=['ch.swisstopo.swisseo_s2-sr_v100']
)
s2_files = np.array(list(item_search.items()))

# And filter the images (if date_range includes the date of newest satellite image, 
#                        s2_files contains an additional first element 'current'):
    # s2_files[0] # <Item id=swisseo_s2-sr_v100> TODO: we need to drop this first element.
    #             #                              TODO(Joan): is it correct that this first element is different from the rest?
    # s2_files[1] # <Item id=2025-12-06t102319>
    # s2_files[2] # <Item id=2025-12-09t103329>
    # s2_files[-1] # <Item id=2026-03-18t100741>

    # FOR DEVELOPMENT: INVESTIGATION:
    # FOR DEVELOPMENT: s2_files[0].datetime # 2026-03-18 10:07:41+00:00 # is this just always the datetime of the newest element?
    # FOR DEVELOPMENT: s2_files[0].id "swisseo_s2-sr_v100"
    # FOR DEVELOPMENT: s2_files[0].assets # {'ch.swisstopo.swisseo_s2-sr_v100_mosaic_current_cloudprobability-10m.tif':
    # FOR DEVELOPMENT: s2_files[1].assets # {'ch.swisstopo.swisseo_s2-sr_v100_mosaic_2025-12-06t102319_bands-10m.tif':
    # FOR DEVELOPMENT: Aha, based on above it looks like the first is always the current

# mark which indices do not represent specific dates, but just the current state:
remove_idx = np.array(["current" in list(itm.assets.keys())[0] for itm in s2_files])
# s2_files[np.array(remove_idx)]           # This one is removed
s2_files = s2_files[~remove_idx] # These are kept

        # TO FIX A PREVIOUSLY DOWNLOADED DATA SET CONTAINING THE FIRST WRONG DATE, DO THE FOLLOWING:
        # ds_out2_initial = xr.open_dataset("/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-23_06h15_ndvi_01_downloaded_2025-11-30_2026-03-22.zarr")
        # drop the current time step
        # ds_out2_initial.datetime.values # Yes, the first line shows the newest date.
        # ds_out2_initial_fixed = ds_out2_initial.isel(datetime = slice(1,None))
        # ds_out2_initial_fixed.to_zarr(OUTPUT_ZARR, mode="w", consolidated=True, compute=True)


# If some images (s2_files) are available within the requested date_range
if (len(s2_files) > 0):
    print(f"Starting download for:\n{"\n".join([item.datetime.strftime('%Y-%m-%d_%Hh%M') for item in s2_files])}",
          file=sys.stdout,
          flush=True)

    # ==========================================================================

    ### # BELOW WAS DONE ONCE AND IS NOW HARDCODED OR STORED IN ../data/forest_mask_bits.zarr
    ### # Retrieve the spatial coverage (bounds) of all 4 possible orbits covering Switzerland
    ### def collect_bounds_all_orbits():
    ###     """
    ###     Collects the bounds of all orbits in the Swiss dataset.
    ###     Returns a list of BoundingBox objects.
    ###     """
    ###     item_search = service.search(
    ###         bbox=bbox_swiss_4326,
    ###         datetime="2025-04-30/2025-05-02",               # NOTE: this must be kept fixed since it defines the bounding box,
    ###                                                         #       and thus also the grid size and pixel ID.
    ###                                                         #       Thus it must remain the same as for the historic data set,
    ###                                                         #       and also for the median data set.
    ###         collections=['ch.swisstopo.swisseo_s2-sr_v100']
    ###     )
    ###     s2_files_sample_orbits = list(item_search.items())
    ###
    ###     all_bounds = []
    ###
    ###     for item in tqdm(s2_files_sample_orbits):
    ###         assets = item.assets
    ###         key_bands = [k for k in assets.keys() if k.endswith('bands-10m.tif')][0]
    ###         bands_asset = assets[key_bands]
    ###         with rasterio.open(bands_asset.href) as src:
    ###             bounds = src.bounds
    ###             all_bounds.append(bounds)
    ###
    ###     return all_bounds
    ###
    ### # Combine all bounding boxes into one global bounding box and compute its pixel dimensions
    ### def union_bounds(bounds_list):
    ###     """
    ###     Takes a list of BoundingBox objects and returns a single BoundingBox
    ###     that encompasses all the bounds, along with the width and height
    ###     of the bounding box in pixels, assuming a resolution of 10 meters.
    ###     """
    ###     left = min(b.left for b in bounds_list)
    ###     bottom = min(b.bottom for b in bounds_list)
    ###     right = max(b.right for b in bounds_list)
    ###     top = max(b.top for b in bounds_list)
    ###     resolution = 10
    ###     width = int((right - left) / resolution)
    ###     height = int((top - bottom) / resolution)
    ###     return BoundingBox(left, bottom, right, top), width, height
    ### 
    ### # EPSG: 2056
    ### # Swiss coordinate system (CH1903+ / LV95)
    ### # This is the full reference bounding box for the Swisstopo dataset covering the 4 orbits
    ### bbox_swisstopo_2056, width_swisstopo, height_swisstopo = union_bounds(collect_bounds_all_orbits())
    ### NOW HARDCODE THIS
    bbox_swisstopo_2056 = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
    width_swisstopo     = int((bbox_swisstopo_2056.right - bbox_swisstopo_2056.left) / 10) # 37728
    height_swisstopo    = int((bbox_swisstopo_2056.top - bbox_swisstopo_2056.bottom) / 10) # 24542
    ### END HARDCODING
    print("bbox_swisstopo_2056, width_swisstopo, height_swisstopo", flush = True)
    print(bbox_swisstopo_2056, flush = True)
    print(width_swisstopo, flush = True)
    print(height_swisstopo, flush = True)
    

    ### # BELOW WAS DONE ONCE AND IS NOW HARDCODED OR STORED IN ../data/forest_mask_bits.zarr
    ### # Take the forest mask from the Swisstopo VHI dataset 
    ### # The VHI dataset contains the forest mask that Swisstopo derived from the habitat map
    ### # Also collect the metadata using the forest mask as a reference raster
    ### def get_forest_mask():
    ###     """
    ###     Downloads the forest mask from the Swisstopo VHI dataset.
    ###     Returns a numpy array representing the forest mask.
    ###     Also returns the metadata for the reference raster.
    ###     """
    ###     item_search = service.search(
    ###         bbox=bbox_swiss_4326,
    ###         datetime='2025-05-01/2025-05-01', # use the forest mask of a hardcoded date
    ###                                           # NOTE: this must be kept fixed since it defines the forest mask
    ###                                           #       and thus also the pixel ID.
    ###                                           #       Thus it must remain the same as for the historic data set,
    ###                                           #       and also for the median data set.
    ###         collections=['ch.swisstopo.swisseo_vhi_v100']
    ###     )
    ###     items = list(item_search.items())
    ###     item = items[0]
    ###     assets = item.assets
    ###     key_bands = [k for k in assets.keys() if k.endswith('forest-10m.tif')][0]
    ###     bands_asset = assets[key_bands]
    ###     
    ###     with rasterio.open(bands_asset.href) as src:
    ###         window = src.window(*bbox_swisstopo_2056)
    ###         vhi = src.read(1, window=window)
    ###         forest_mask = (vhi != 255).astype('uint8')
    ###         ref_meta = {
    ###             "transform": src.window_transform(window),
    ###             "crs": src.crs,
    ###             "width": window.width,
    ###             "height": window.height
    ###         }
    ###     
    ###     return forest_mask, ref_meta
    ### 
    ### forest_mask, ref_meta = get_forest_mask()
    ### # save forest mask to disk
    ### bits = np.packbits(forest_mask.ravel())           # uint8 array, 8x smaller before compression
    ### store = zarr.open("/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/data/forest_mask_bits.zarr", mode="w")
    ### store.create_dataset("bits", data=bits, compressor=zarr.codecs.BloscCodec(cname="zstd", clevel=3), shape = bits.shape)
    ### NOW HARDCODE THIS
    ref_meta = {
        'transform': Affine(10.0, 0.0,  bbox_swisstopo_2056.left,   # Affine(10.0, 0.0, np.float64(2474090.0),
                            0.0, -10.0, bbox_swisstopo_2056.top),   #        0.0, -10.0, np.float64(1310530.0)), 
        'crs': CRS.from_wkt('PROJCS["CH1903+ / LV95",GEOGCS["CH1903+",DATUM["CH1903+",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],AUTHORITY["EPSG","6150"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4150"]],PROJECTION["Hotine_Oblique_Mercator_Azimuth_Center"],PARAMETER["latitude_of_center",46.9524055555556],PARAMETER["longitude_of_center",7.43958333333333],PARAMETER["azimuth",90],PARAMETER["rectified_grid_angle",90],PARAMETER["scale_factor",1],PARAMETER["false_easting",2600000],PARAMETER["false_northing",1200000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","2056"]]'), 
        'width': np.float64(width_swisstopo), 
        'height': np.float64(height_swisstopo)}
    # load forest mask from disk
    forest_mask_zarr = zarr.open("/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/data/forest_mask_bits.zarr", mode="r")
    forest_mask_shape = (height_swisstopo, width_swisstopo)
    forest_mask = np.unpackbits(forest_mask_zarr["bits"][:])[:np.prod(forest_mask_shape)].reshape(forest_mask_shape)
    # np.array_equiv(forest_mask2, forest_mask) # True, this confirmed that recovery was good.
    ### END HARDCODING
    reference_summary_msg = (
        f"Total of global grid used for pixel ID (based on forest mask): " + 
        f"\nBox: {bbox_swisstopo_2056} pixels" + 
        f"\nGrid: {forest_mask.shape} = {forest_mask.size:_} pixels" + 
        f", of which {np.flatnonzero(forest_mask).size:_} are identified as forest pixels."
    ) # Box: BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0) pixels
      # Grid: (24542, 37728) = 925_920_576 pixels, of which 105_715_396 are identified as forest pixels.
    print(reference_summary_msg, flush = True)
    print("Reference raster metadata:")
    print(ref_meta, flush = True)

    # ==========================================================================

    # Build index mapping from forest pixels in the full reference raster to 1D flat indices
    global_forest_pixelIDs = np.flatnonzero(forest_mask == 1)
    # index_map = np.full(global_forest_pixelIDs.max() + 1, -1, dtype=np.int32) # contains as many elements as forest_mask (width_swisstopo * height_swisstopo)
    index_map = np.full(forest_mask.size, -1, dtype=np.int32)                # contains as many elements as forest_mask (width_swisstopo * height_swisstopo)
                                                                             # NOTE that using max() it stops at largest index needed, whereas with .size the last rows are filled with -1
    index_map[global_forest_pixelIDs] = np.arange(len(global_forest_pixelIDs))
    # at locations of the global grid 
    # index_map contains
    # the corresponding index (pixelID) in the forest-only pixel vector
    # i.e. np.reshape(index_map, (height_swisstopo, width_swisstopo)) re-builds the global grid
    #
    # at locations of forest-only pixels (pixelID)
    # global_forest_pixelIDs contains
    # index (pixel_index) of the flattened global grid (24542 x 37728)
    # i.e. index_map[global_forest_pixelIDs] gives the continuous indices from 0 to 105715395

    # Prepare constants
    N = len(global_forest_pixelIDs)
    T = len(s2_files)
    INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
    NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step

    ## Define the datasets stores for NDVI, NDSI and timestep values to be filled in the loop
    # Shape is (T, N) where T is the number of time steps and N is the number of forest pixels
    # Use int16 to save space, with a fill value for no coverage
    # Use compression to save space
    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    # delete temporary Zarr store if it is existing already
    if os.path.exists(OUTPUT_ZARR_TEMP):
        shutil.rmtree(OUTPUT_ZARR_TEMP)
    
    ndvi_ds = zarr.create_array(
        name="ndvi",
        store= OUTPUT_ZARR_TEMP,
        shape=(T, N),
        chunks=(1, N),
        dtype="int16",
        fill_value=NO_COVERAGE,
        compressors=compressors,
        zarr_format=3, overwrite=True
    )

    ndsi_ds = zarr.create_array(
        name="ndsi",
        store= OUTPUT_ZARR_TEMP,
        shape=(T, N),
        chunks=(1, N),
        dtype="int16",
        fill_value=NO_COVERAGE,
        compressors=compressors,
        zarr_format=3, overwrite=True
    )

    timesteps_ds = zarr.create_array(
        name="timestep",
        store= OUTPUT_ZARR_TEMP,
        shape=(T,),
        chunks=(1,),
        dtype="int64", # use int64 as nanoseconds since 1970. Good until year ~2262.
        fill_value=np.iinfo(np.int64).min,
        compressors=compressors,
        zarr_format=3, overwrite=True
    )   # np.iinfo(np.int32).max / 3600 / 24 / 365  # = 68 when representing seconds, int32 are only valid until 1970+68=2038
        # np.iinfo(np.int64).max / 3600 / 24 / 365  # = 292e9 (if seconds => good for 300e9 years, if nanoseconds => good for 300 years)

    timesteps_ds.attrs['description'] = 'Datetime in nanoseconds since 1970-01-01 (int64)'
    ndvi_ds.attrs['description'] = 'NDVI (scaled int16: -10000 to 10000)'
    ndsi_ds.attrs['description'] = 'NDSI (scaled int16: -10000 to 10000)'
    ndvi_ds.attrs['nodata'] = NO_COVERAGE
    ndvi_ds.attrs['cloud_shadow'] = INVALID
    # FOR INTERACTIVE DEVELOPMENT:
    #     from time import sleep
    #     for t, item in tqdm(enumerate(s2_files), total=len(s2_files)):
    #         print(item)
    #         print(item.datetime)
    #         sleep(0.1)
    def add_timestep_to_zarr(t, item):
        timestep_dttm = item.datetime
        assets = item.assets
        bands10_asset = assets[[k for k in assets if k.endswith('bands-10m.tif')][0]]
        bands20_asset = assets[[k for k in assets if k.endswith('bands-20m.tif')][0]]
        masks_asset = assets[[k for k in assets if k.endswith('masks-10m.tif')][0]]

        # FOR INTERACTIVE DEVELOPMENT
        #     from contextlib import ExitStack
        #     import rasterio
        #     stack = ExitStack()
        #     b10_src = stack.enter_context(rasterio.open(bands10_asset.href))
        #     b20_src = stack.enter_context(rasterio.open(bands20_asset.href))
        #     masks_src = stack.enter_context(rasterio.open(masks_asset.href))
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
        current_pixelIDs = index_map[global_flat] # this contains pixelIDs (i.e. indices in the forest-only pixel vector)

        # Flat masks
        cloud_shadows_mask_flat = cloud_shadows_mask[local_rows, local_cols]
        nodata_mask_flat_ndvi = nodata_mask_ndvi[local_rows, local_cols]
        valid_ndvi = ~(cloud_shadows_mask_flat | nodata_mask_flat_ndvi)
        cloud_only_ndvi = cloud_shadows_mask_flat & ~nodata_mask_flat_ndvi
        nodata_mask_flat_ndsi = nodata_mask_ndsi[local_rows, local_cols]
        valid_ndsi = ~(cloud_shadows_mask_flat | nodata_mask_flat_ndsi)
        cloud_only_ndsi = cloud_shadows_mask_flat & ~nodata_mask_flat_ndsi

        # Append to Zarr storage:
        # Write timestep
        # NOTE: since zarr does not supprt NumPy datetime64[ns] dytpes, we
        #       store the times as int64 epoch values (nanoseconds since 1970-01-01)
        timesteps_ds[t] = np.datetime64(timestep_dttm).astype("datetime64[ns]").astype("int64")

        # Write NDVI
        ndvi_flat = ndvi_scaled[local_rows, local_cols]
        ndvi_row = np.full(N, NO_COVERAGE, dtype="int16")
        ndvi_row[current_pixelIDs[valid_ndvi]] = ndvi_flat[valid_ndvi]
        ndvi_row[current_pixelIDs[cloud_only_ndvi]] = INVALID
        ndvi_ds[t] = ndvi_row # write to zarr

        # Write NDSI
        ndsi_flat = ndsi_scaled[local_rows, local_cols]
        ndsi_row = np.full(N, NO_COVERAGE, dtype="int16")
        ndsi_row[current_pixelIDs[valid_ndsi]] = ndsi_flat[valid_ndsi]
        ndsi_row[current_pixelIDs[cloud_only_ndsi]] = INVALID
        ndsi_ds[t] = ndsi_row # write to zarr

    failed_timesteps = []
    for t, path in tqdm(enumerate(s2_files), total=len(s2_files)):
        try:
            add_timestep_to_zarr(t, path)
            print(f"Time step {t} processed successfully.", flush = True)
        except Exception as e:
            print(f"Time step {t} failed: {e}", flush = True)
            failed_timesteps.append((t, path))
            continue  # skip to the next time step

    # Retry the failed time steps a second time
    if failed_timesteps:    
        print(f"Retrying {len(failed_timesteps)} failed time steps...", flush = True)
        for t, path in tqdm(failed_timesteps):
            try:
                add_timestep_to_zarr(t, path)
                print(f"Time step {t} retried successfully.", flush = True)
            except Exception as e:
                print(f"Time step {t} retry failed: {e}", flush = True)
                continue  # skip to the next time step

    # Transform unstructured zarr to structured xarray dataset stored in zarr:
    # TODO: check if needed for speedup: DASK_TEMP_DIR = "/mnt/data1/UniBe-swiss-ndvi/tmp_data"
    # TODO: check if needed for speedup: os.makedirs(DASK_TEMP_DIR, exist_ok=True)

    # TODO: check if needed for speedup: N_WORKERS = 40
    # TODO: check if needed for speedup: MEMORY_LIMIT = "300GB"
    # TODO: check if needed for speedup: cluster = LocalCluster(
    # TODO: check if needed for speedup:     n_workers=N_WORKERS,
    # TODO: check if needed for speedup:     threads_per_worker=1,
    # TODO: check if needed for speedup:     processes=True,
    # TODO: check if needed for speedup:     memory_limit=MEMORY_LIMIT,
    # TODO: check if needed for speedup:     dashboard_address=":8340",
    # TODO: check if needed for speedup:     local_directory=DASK_TEMP_DIR,
    # TODO: check if needed for speedup: )
    # TODO: check if needed for speedup: client = Client(cluster)
    # TODO: check if needed for speedup: print(client, flush = True)
    # TODO: check if needed for speedup: print(client.dashboard_link, flush = True) # use this dashboard to follow progress

    ds0 = zarr.open_group(OUTPUT_ZARR_TEMP, mode="r")
    ndvi_da = da.from_zarr(ds0["ndvi"])
    ndsi_da = da.from_zarr(ds0["ndsi"])
    times_da = da.from_zarr(ds0["timestep"]).astype("datetime64[ns]").compute()
    

    PIXEL_CHUNKS = 10000

    ndvi_xr = xr.DataArray(
        ndvi_da, # this is of shape (timesteps, 105Mio pixel)
        dims=("datetime", "pixel"),
        coords={
            "pixel": np.arange(ndvi_da.shape[1], dtype=np.int32),
            "datetime": times_da
        },
        name="ndvi"
    ).chunk({"pixel": PIXEL_CHUNKS, "datetime": -1})


    ndsi_xr = xr.DataArray(
        ndsi_da, # this is of shape (timesteps, 105Mio pixel)
        dims=("datetime", "pixel"),
        coords={
            "pixel": np.arange(ndsi_da.shape[1], dtype=np.int32),
            "datetime": times_da
        },
        name="ndvi"
    ).chunk({"pixel": PIXEL_CHUNKS, "datetime": -1})

    ds_out = xr.Dataset(
        {
            "ndvi": ndvi_xr,
            "ndsi": ndsi_xr
        }
    )

    # Add a day-level coord used for grouping (keeps original 'datetime' untouched).
    # append date (rounding) => multiple datetimes can have same date
    ds_out = ds_out.assign_coords(
        date=ds_out.datetime.dt.floor("D")
    )

    # Extend pixel dimensions by appending x and y:

    # Define grid underlying PixelID and needed transformations
    # Define transform between row,col to coord (upper-left origin, pixel sizes)
    trans = ref_meta["transform"]
    rows, cols = np.nonzero(forest_mask)
    ids = np.arange(len(rows))
    xs, ys = rasterio.transform.xy(trans, rows, cols)
    coord_lookup = pd.DataFrame({
        'pixel': ids,
        'x': xs,
        'y': ys,
        'x_idx': rows,
        'y_idx': cols,
    }).set_index('pixel')
    
    # Align lookup by pixel values
    pixel_coords   = ds_out.pixel.values
    coord_lookup_aligned = coord_lookup.loc[pixel_coords]

    ds_out2 = ds_out.assign_coords(
        # change number types of dimensions (pixel is that way 420MB instead of 840MB)
        pixel = ('pixel', ds_out.pixel.values.astype(np.int32)),
        # doy   = ('date', ds_out.doy.values.astype(np.int32)),
        # and add coordinates and indices in regular grid
        x=('pixel', coord_lookup_aligned['x'].values.astype(np.int32)), # or uint32
        y=('pixel', coord_lookup_aligned['y'].values.astype(np.int32)), # or uint32
        x_idx=('pixel', coord_lookup_aligned['x_idx'].values.astype(np.int32)), # or uint32
        y_idx=('pixel', coord_lookup_aligned['y_idx'].values.astype(np.int32))  # or uint32
    )
    
    ds_out2.attrs["transform_note"] = str(trans)
    ds_out2.attrs["transform_coeffs"] = tuple(float(v) for v in trans)
    ds_out2.attrs["transform_instr"] = "from affine import Affine; t = Affine(*ds.attrs['transform_coeffs'][0:6])"
    ds_out2.attrs['description_ndvi'] = 'NDVI (scaled int16: -10000 to 10000)'
    ds_out2.attrs['description_ndsi'] = 'NDSI (scaled int16: -10000 to 10000)'
    ds_out2.attrs['nodata'] = NO_COVERAGE
    ds_out2.attrs['cloud_shadow'] = INVALID
    
    ds_out2.attrs['pixel_definition'] = reference_summary_msg
    # ==========================================================================
    # Write out
    ds_out2.to_zarr(OUTPUT_ZARR, mode="w", consolidated=True, compute=True)

    # test load this dataset:
    # ds_test = xr.open_dataset(OUTPUT_ZARR)
    # # ds_test = xr.open_dataset("/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-23_22h26_ndvi_01_downloaded_2026-01-01_2026-01-15.zarr")
    # # ds_test = xr.open_dataset("/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-23_22h26_ndvi_01_downloaded_2026-01-01_2026-01-15.zarr")
    # # ds_test = xr.open_dataset("/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-23_12h50_ndvi_01_downloaded_2025-11-30_2026-03-22.zarr")
    # # test plot this dataset    
    # xmin, xmax = 2650000, 2750000 # focus on Ticino
    # ymin, ymax = 1070000, 1160000 # focus on Ticino
    # pixels_subset_mask = (
    #     (ds_test.x.values >= xmin) &
    #     (ds_test.x.values <= xmax) &
    #     (ds_test.y.values >= ymin) &
    #     (ds_test.y.values <= ymax)
    # )
    # ds_test_subset = ds_test["ndvi"].isel(pixel=pixels_subset_mask.nonzero()[0])
    # plot_da_map(ds_test_subset.isel(datetime = 0), png_fname = 'foo5.png')
    # plot_da_map(ds_test_subset.isel(datetime = 18), png_fname = 'foo5ter.png')
    # pixels_subset_mask2 = (
    #     (ds_test.ndvi.values != 32767) &
    #     (ds_test.ndvi.values != -32768) &
    #     (ds_test.y.values >= ymin) &
    #     (ds_test.y.values <= ymax)
    # )
    # plot_da_map(ds_test_subset.isel(datetime = 0), reduction_factor = 1, png_fname = 'foo1.png')


    # delete temporary Zarr store to clean up
    if os.path.exists(OUTPUT_ZARR_TEMP): 
       shutil.rmtree(OUTPUT_ZARR_TEMP)

    print(f"'1_... .py created file: {OUTPUT_ZARR}", flush = True)
    print(OUTPUT_ZARR, flush = True)
    sys.exit(0)

else:

    print("No data downloaded.", flush=True)
    sys.exit(0)

