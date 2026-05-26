# This plots time series of selected pixels, illustrating our processing

from datetime import datetime, date
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
import xarray as xr
import os, sys
import shutil
import pandas as pd
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import datetime as dt
import argparse

# for the download
import dask.array as da
import pystac_client
import rasterio
import rioxarray
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import zarr
from tqdm import tqdm
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import sys
from affine import Affine
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)


# NOTE: below only works with working directory at /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing
# from workflow_implementation.MS1_script_for_historical_NDVI.new_historical_processing.NDVI_utils.NDVI_plot_utils import NDVI_xarray_to_grid
# NOTE: below only works with working directory at /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_historical_processing
from NDVI_utils.NDVI_plot_utils import NDVI_xarray_to_grid


# =====================================================
#  Define input data sets
# =====================================================

# PROC_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr"
# PROC_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7_2025-11.zarr" # this one is cropped to November 2025
# PROC_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7b.zarr"
PROC_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c.zarr"
# PROC_ZARR = "/mnt/data1/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_2025-11.zarr"
# PROC_ZARR = "/mnt/data1/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365.zarr" # this is the initially processed file (just appended with metadata x,y, x_idx, etc.)

INPUT_ZARR_LOOKUPTABLE = "/mnt/data2/UniBe-swiss-ndvi/input_data/lookup_table_median_ndvi_v7.zarr"
OBS_ZARR = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"

# =====================================================
#  Define some pixels of interest
# =====================================================

site_selection = "_toughSites"
site_selection = "_SHSites"
site_selection = "_evaluationSites"

match site_selection:
    case "_toughSites":
        XY_COORDS = {
            # Specific sites
            'Bitsch': (2644035, 1133765), # pix 84856712 # NOTE: Bitsch forest fire selection Fabian
            # some Western border:
            # NOTE: if PixelID is off-by-one Western and Eastern border should indicate
            'Auberson':          (2523115, 1185065), # pix 52709046
            'La Chaux-de-Fonds': (2546995, 1217685), # pix 30185728
            # some Eastern border:
            'Widerberg':  (2762585, 1230445),        # pix 23069246
            'Diepoldsau': (2768685, 1251225),        # pix 10771926
            'Tschlin':    (2831125, 1195615),        # pix 45329883
            # Randomly selected sites:
            # # some Ticino sites:
            # 'Ticino1': (2720645, 1118245),          # pix95774249
            'Tenero': (2710385, 1116375),          # pix97148954
            # # 'Ticino3': (2710005, 1109995),
            # Wabern:
            'Wabern': (2600875, 1197275)      # pix 44088229
        }
    case "_SHSites":
        XY_COORDS = {   # some Schaffhausen sites:
            'SH1_pix0': (2684595, 1295915), #pix0
            'SH2': (2684555, 1295715),      #pix210
            'SH3': (2684955, 1295675),      #pix350
            'SH4': (2684395, 1295635),      #pix490
            'SH5': (2684895, 1295545),      #pix999
        }
    case "_evaluationSites":
        XY_COORDS = { # Taken from report
            'Lowland broadleaf':                    (2694491, 1126023),
            'Highland broadleaf':                   (2692020, 1121443),
            'Lowland evergreen':                    (2761097, 1194613),
            'Highland evergreen':                   (2781537, 1182974),
            # 'Bitsch fire affected area':            (2644029, 1134128),
            'Bitsch fire affected area':            (2644035, 1133765), # pix 84856712 # NOTE: Bitsch forest fire selection Fabian
            'Bitsch fire nearby non-affected area': (2644328, 1134342),
            '2018 Drought-affected area':           (2690025, 1287413),
            'Storm affected area':             (2689564, 1154411),
        }


SHORTNAMES = {
    'Lowland broadleaf': '_low_blf',
    'Highland broadleaf': '_high_blf',
    'Lowland evergreen': '_low_enf',
    'Highland evergreen': '_high_enf',
    'Bitsch fire affected area': '_bitsch_fire',
    'Bitsch fire nearby non-affected area': '_bitsch_nonfire',
    '2018 Drought-affected area': '_drought2018',
    'Storm affected area': '_storm',
}

def round_to_5_ending(n):
    return round((n - 5) / 10) * 10 + 5

NAMES    = [nm for nm in XY_COORDS.keys()]
X_COORDS = [round_to_5_ending(xy[0]) for xy in XY_COORDS.values()]
Y_COORDS = [round_to_5_ending(xy[1]) for xy in XY_COORDS.values()]


# For figure 01 we also need an interval of interest:
FIGURE01_START_DATE="2025-04-01"
FIGURE01_END_DATE="2025-08-01"

# =====================================================
#  Define other configuration details
# =====================================================
NO_COVERAGE = 32767
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step
INVALID     = -32768
INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows

N_WORKERS = 10

client = Client(
n_workers=N_WORKERS,
threads_per_worker=1,
memory_limit='50GB',
processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
dashboard_address=':2234')  
print(client.dashboard_link)



def download_timeseries_NDVI_singlePixel(
    x, y,                   # Integer coordinates of pixel center (must end with 5)
    start_date, end_date):  # String dates ("YYYY-MM-DD")
    # NOTE: below code is duplicated (copy/paste) from 1_extract_swisstopo_dataset.py

    downloadpath = (
        "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig/prova2/"+
        "TESTSUITE_"+
        f"{start_date.replace("-","")}to{end_date.replace("-","")}_"+
        f"location_{x}x{y}"+
        ".csv")
    
    if os.path.exists(downloadpath):
        df = pd.read_csv(downloadpath, index_col=0, parse_dates=[1])
        return(df)
    else:
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
        # And filter the images:
        # FOR DEVELOPMENT: s2_files[0].assets # {'ch.swisstopo.swisseo_s2-sr_v100_mosaic_current_cloudprobability-10m.tif':
        # FOR DEVELOPMENT: s2_files[1].assets # {'ch.swisstopo.swisseo_s2-sr_v100_mosaic_2025-12-06t102319_bands-10m.tif':
        
        # mark which indices do not represent specific dates, but just the current state:
        remove_idx = np.array(["current" in list(itm.assets.keys())[0] for itm in s2_files])
        s2_files = s2_files[~remove_idx] # These are kept


        # If some images (s2_files) are available within the requested date_range
        if (len(s2_files) > 0):
            print(f"Starting download for:\n{"\n".join([item.datetime.strftime('%Y-%m-%d_%Hh%M') for item in s2_files])}",
                file=sys.stdout,
                flush=True)

            ### NOW HARDCODE THIS
            bbox_swisstopo_2056 = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
            width_swisstopo     = int((bbox_swisstopo_2056.right - bbox_swisstopo_2056.left) / 10) # 37728
            height_swisstopo    = int((bbox_swisstopo_2056.top - bbox_swisstopo_2056.bottom) / 10) # 24542        
            ref_meta = {
            'transform': Affine(10.0, 0.0,  bbox_swisstopo_2056.left,   # Affine(10.0, 0.0, np.float64(2474090.0),
                                0.0, -10.0, bbox_swisstopo_2056.top),   #        0.0, -10.0, np.float64(1310530.0)), 
            'crs': CRS.from_wkt('PROJCS["CH1903+ / LV95",GEOGCS["CH1903+",DATUM["CH1903+",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],AUTHORITY["EPSG","6150"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4150"]],PROJECTION["Hotine_Oblique_Mercator_Azimuth_Center"],PARAMETER["latitude_of_center",46.9524055555556],PARAMETER["longitude_of_center",7.43958333333333],PARAMETER["azimuth",90],PARAMETER["rectified_grid_angle",90],PARAMETER["scale_factor",1],PARAMETER["false_easting",2600000],PARAMETER["false_northing",1200000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","2056"]]'), 
            'width': np.float64(width_swisstopo), 
            'height': np.float64(height_swisstopo)}
            # load forest mask from disk
            # UNNEEDED: forest_mask_zarr = zarr.open("/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/data/forest_mask_bits.zarr", mode="r")
            # UNNEEDED: forest_mask_shape = (height_swisstopo, width_swisstopo)
            # UNNEEDED: forest_mask = np.unpackbits(forest_mask_zarr["bits"][:])[:np.prod(forest_mask_shape)].reshape(forest_mask_shape)
            ### END HARDCODING
            reference_summary_msg = (
                f"Total of global grid used for pixel ID (based on forest mask): " + 
                f"\nBox: {bbox_swisstopo_2056} pixels" #+ 
                # f"\nGrid: {forest_mask.shape} = {forest_mask.size:_} pixels" + 
                # f", of which {np.flatnonzero(forest_mask).size:_} are identified as forest pixels."
            ) # Box: BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0) pixels
            # Grid: (24542, 37728) = 925_920_576 pixels, of which 105_715_396 are identified as forest pixels.
            print(reference_summary_msg, flush = True)
            print("Reference raster metadata:")
            print(ref_meta, flush = True)

            # item = s2_files[0]; coords = (2710005, 1109995) # For development.
            def get_timestep_NDVI_singlePixel(item, coords):

                # Prepare constants
                INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
                NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step

                #bbox_single_pixel_swisstopo_2056 = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
                half_px_size = 5 # 10/2 = 5 m
                bbox_single_pixel_swisstopo_2056 = BoundingBox(
                    left =coords[0]-half_px_size, bottom=coords[1]-half_px_size, 
                    right=coords[0]+half_px_size, top   =coords[1]+half_px_size)
                

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
                        b10_window = from_bounds(*bbox_single_pixel_swisstopo_2056, transform=b10_src.transform)
                        mask_window = from_bounds(*bbox_single_pixel_swisstopo_2056, transform=masks_src.transform)
                        b20_window = from_bounds(*bbox_single_pixel_swisstopo_2056, transform=b20_src.transform)

                        red, green, nir = b10_src.read([1, 2, 4], window=b10_window, boundless=True, fill_value=9999)
                        swir = b20_src.read(3, window=b20_window, boundless=True, fill_value=9999)
                        masks = masks_src.read([1, 2], window=mask_window, boundless=True, fill_value=255).astype("uint8")

                    else:
                        b10_window = b10_src.window(*bbox_single_pixel_swisstopo_2056)
                        b20_window = b20_src.window(*bbox_single_pixel_swisstopo_2056)
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

                assert ndvi.size == 1 # below only works for 1 pixel

                # Return a 1-row dataframe for the given date (item)
                # out_df = pd.DataFrame(); out_df['a'] = [12]
                out_df = pd.DataFrame(); 
                out_df['datetime'] = [timestep_dttm]
                out_df['x'] = [coords[0]]
                out_df['y'] = [coords[1]]

                out_df['ndvi']        = [ndvi[0][0]]
                out_df['ndvi_scaled'] = [ndvi_scaled[0][0]]
                out_df['ndsi']        = [ndsi[0][0]]
                out_df['ndsi_scaled'] = [ndsi_scaled[0][0]]

                return(out_df)

            failed_timesteps = []
            dataframerows = []
            # FOR DEVELOPMENT: t = 0; path = s2_files[t]
            for t, path in tqdm(enumerate(s2_files), total=len(s2_files)):
                try:
                    row = get_timestep_NDVI_singlePixel(path, (x, y))
                    dataframerows.append(row)
                    print(f"Time step {t} processed successfully.", flush = True)
                except Exception as e:
                    print(f"Time step {t} failed: {e}", flush = True)
                    failed_timesteps.append((path, (x, y)))
                    continue  # skip to the next time step

            df = pd.concat(dataframerows, ignore_index=True)

            # store the downloaded data:
            df.to_csv(downloadpath)

        return(df)



# =====================================================
#  Load data sets
# =====================================================
obs_ds  = xr.open_dataset(OBS_ZARR,  chunks={}, mask_and_scale= False)
proc_ds = xr.open_dataset(PROC_ZARR, chunks={}, mask_and_scale= False)

# --- load median values for each doy --------------------------
lookuptable  = xr.open_zarr(INPUT_ZARR_LOOKUPTABLE)


# --- add median NDVI from model ----------------------------------
# Append day-of-year (for merging of median expected NDVI from model)
doy_array = pd.to_datetime(obs_ds['datetime']).dayofyear
obs_ds = obs_ds.assign_coords(doy = ('datetime', doy_array))
doy_noLeap = xr.where(obs_ds.doy == 366, 365, obs_ds.doy) # remove leap year if encountered
obs_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
        doy=doy_noLeap,
        pixel=obs_ds.pixel) # this is to join by pixels and doy

doy_array = pd.to_datetime(proc_ds['date']).dayofyear
proc_ds = proc_ds.assign_coords(doy = ('date', doy_array))
doy_noLeap = xr.where(proc_ds.doy == 366, 365, proc_ds.doy) # remove leap year if encountered
proc_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
        doy=doy_noLeap,
        pixel=proc_ds.pixel) # this is to join by pixels and doy




# =====================================================
#  create MultiIndex, for simpler indexing
# =====================================================

# create MultiIndex (keeping also the option of using pixelID)
proc_ds_idx = proc_ds.assign_coords(pixelID=proc_ds["pixel"])
obs_ds_idx = obs_ds.assign_coords(pixelID=obs_ds["pixel"])
# create MultiIndex with three levels: x, y, pixelID
proc_ds_midx = proc_ds_idx.set_index(pixel=["x", "y", "pixelID"])
obs_ds_midx = obs_ds_idx.set_index(pixel=["x", "y", "pixelID"])

def select_by_xy(ds_mi, xs, ys):
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    if not isinstance(ys, (list, tuple)):
        ys = [ys]
    assert len(xs) == len(ys)
    selections = [ds_mi.sel(pixel=(x, y, slice(None)), drop=True)
                   for x, y in zip(xs, ys)]
    return xr.concat(selections, dim='pixel')

def select_by_pixelID(ds_mi, pid):
    return ds_mi.sel(pixel=(slice(None), slice(None), pid), drop=True)

# select_by_xy(proc_ds_midx, [2684595, 2684555, 2684525], [1295915, 1295715, 1295715])
# select_by_pixelID(proc_ds_midx, [0, 210, 350])
# select_by_xy(obs_ds_midx, [2684595, 2684555, 2684525], [1295915, 1295715, 1295715])
# select_by_pixelID(obs_ds_midx, [0, 210, 350])

def select_area_box_by_xy(ds_mi, xs, ys, Lx=10, Ly=10):
    # if not isinstance(xs, (list, tuple)):
    #     xs = [xs]
    # if not isinstance(ys, (list, tuple)):
    #     ys = [ys]
    # assert len(xs) == len(ys)
    # selections = [select_area_box_by_single_xy(ds_mi, x, y, Lx=Lx, Ly=Ly)
    #                for x, y in zip(xs, ys)]
    # return xr.concat(selections, dim='pixel')
    return select_area_box_by_single_xy(ds_mi, xs, ys, Lx=Lx, Ly=Ly)

def select_area_box_by_single_xy(ds_mi, x_center, y_center, Lx=10, Ly=10):
    half_x = (Lx/2.0) # in m
    half_y = (Ly/2.0) # in m
    xmin, xmax = x_center - half_x, x_center + half_x
    ymin, ymax = y_center - half_y, y_center + half_y
    # # Variant 1:
    # mask = (ds_mi.x >= xmin) & (ds_mi.x <= xmax) & (ds_mi.y >= ymin) & (ds_mi.y <= ymax)
    # return ds_mi.where(mask, drop=True)
    # # Variant 2:
    # # NOTE: this requires lexsorting first: ds_mi = ds_mi.sortby('pixel')   # reorder so tuple-slicing works
    # ds_mi.sel(
    #     pixel = slice((xmin, ymin, -np.inf), (xmax, ymax, np.inf)),
    #     drop=True)
    # Variant 3:
    mi = ds_mi['pixel'].to_index() # multi-index
    mask = (
        (mi.get_level_values(0) >= xmin) & (mi.get_level_values(0) <= xmax) &
        (mi.get_level_values(1) >= ymin) & (mi.get_level_values(1) <= ymax)
    )
    return ds_mi.sel(pixel=mi[mask], drop=True)




# =====================================================
#  Plot figures
# =====================================================

# Figure 0
# --- visual check of resulting data sets ----------------------------------
            # proc_ds_subset = select_by_pixelID(proc_ds_midx, [0, 210, 350, 490, 999]).drop(["y_idx","x_idx"])
            # obs_ds_subset = select_by_pixelID(obs_ds_midx, [0, 210, 350, 490, 999]).drop(["y_idx","x_idx"])
            # proc_ds_subset = select_by_xy(proc_ds_midx, [X_COORD],[Y_COORD]).drop(["y_idx","x_idx"])
            # obs_ds_subset  = select_by_xy(obs_ds_midx,  [X_COORD],[Y_COORD]).drop(["y_idx","x_idx"])
proc_ds_subset = select_by_xy(proc_ds_midx, X_COORDS, Y_COORDS).drop(["y_idx","x_idx"])
obs_ds_subset  = select_by_xy(obs_ds_midx,  X_COORDS, Y_COORDS).drop(["y_idx","x_idx"])

smoothed_cmap = {
    # 0: ("no_obs_to_smooth", "black"),
    # 1: ("no_obs_smoothed",  "orange"),
    2: ("2: obs_to_smooth",    "orange"),
    3: ("3: obs_smoothed",     "black"),
    4: ("4: obs_smoothed_outlier", "red"),
}
obs_cmap     = {0: ("obs_raw",   "green")}
gapfill_cmap = {0: ("gapfilled", "black"),
                1: ("median",    "lightgrey")}

# proc_ds_subset["median_ndvi"].plot.line(x='date',row='pixelID')
# plot all processed
# proc_ds_subset["ndvi_processed"].plot.scatter(x='date',hue='pixel',marker=".", edgecolors="none")
gr = proc_ds_subset["ndvi_processed"].plot.line(
    x='date',row='pixel', color = gapfill_cmap[0][1],
    figsize=(7.2*2, 7.2*2),
    zorder = 3)

# plot processed observation
        # mask_array == 0: the data is not an observation and is yet to be smoothed
        # mask_array == 1: the data is not an observation and is smoothed
        # mask_array == 2: the data is an observation and is yet to be smoothed
        # mask_array == 3: the data is an observation and is smoothed
        # mask_array == 4: the data is an observation and is an outlier
indexer_proc = ((proc_ds_subset["mask_array"] == 2) |
                (proc_ds_subset["mask_array"] == 3) |
                (proc_ds_subset["mask_array"] == 4)).compute()
proc_ds_subset2 = proc_ds_subset.where(indexer_proc, drop=True)

# automatic facetting (doesn't work: https://github.com/pydata/xarray/issues/10176):
# proc_ds_subset2["ndvi_processed"].plot.scatter(x='date',col='pixel',marker="x")
# manual facetting (works):
colors = [smoothed_cmap[k][1] for k in sorted(smoothed_cmap)]
cmap = mcolors.ListedColormap(colors)
# boundaries = np.arange(-0.5, len(colors) + 0.5, 1.0)
# norm = mcolors.BoundaryNorm(boundaries, cmap.N)
for i in range(proc_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.set_prop_cycle(None)
    # add underlying median model: to use plot.line we need to subset "median_ndvi" as a xarrray.DataArray
    proc_ds_subset["median_ndvi"].isel(pixel=i).plot.line( # do not use subset2, but simply subset
        ax=ax, x='date', color = gapfill_cmap[1][1],
        zorder = 0)

    # add ndvi_processed:
    proc_ds_subset2.isel(pixel=i).plot.scatter(
        ax=ax, x='date', marker="x",
        y="ndvi_processed", hue="mask_array",
        cmap=cmap, add_colorbar=False,
        zorder = 2) # norm=norm, 

# plot raw observation
indexer_obs = ((obs_ds_subset["ndvi"] < NO_COVERAGE) &
               (obs_ds_subset["ndvi"] > INVALID)).compute()
obs_ds_subset2 = obs_ds_subset.where(indexer_obs, drop=True)
# automatic facetting (doesn't work: https://github.com/pydata/xarray/issues/10176):
# obs_ds_subset2["ndvi"].plot.scatter(row='pixel',x='datetime',marker="o", hue=None, color="red", alpha=0.2)
# manual facetting (works):
for i in range(obs_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.set_prop_cycle(None)
    obs_ds_subset2.isel(pixel=i).plot.scatter(
        ax=ax, x='datetime',marker="o", hue=None, 
        color=obs_cmap[0][1],
        alpha=0.2, y = "ndvi",
        zorder = 1)

# layouting/formatting
# Format legend
handles_for_legend = [ # legend entry for raw observations:
    Line2D([0], [0], marker='o', linestyle='', color=color, markerfacecolor=color, markersize=6, label=label)
    for _, (label, color) in obs_cmap.items()
] + [                  # legend entry for gapfilled line:
    Line2D([0], [0], marker='', linestyle='-', color=color, markerfacecolor=color, markersize=6, label=label)
    for _, (label, color) in gapfill_cmap.items()
] + [                  # legend entry for processed (smoothed) observations:
    Line2D([0], [0], marker='x', linestyle='', color=color, markerfacecolor=color, markersize=6, label=label)
    for _, (label, color) in smoothed_cmap.items()
]
# Add legend
for i in [obs_ds_subset2.pixel.size-1]: # range(obs_ds_subset2.pixel.size): # legend in last facet is enough
    ax = gr.axs.flat[i]
    ax.legend(handles=handles_for_legend, title="", fontsize="small", loc="lower left") # add discrete legend

# Format x- and y-axes
for i in range(obs_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    ax.set_xlabel("") # remove x labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: val / 10000)) # fix y-axis tick labels
    ax.set_ylim(0, 10000)

# Format default per-facet title 'pixel = (X, Y, pixelID)'
gr.set_titles(template='{coord} = {value}', maxchar=40) # increase maxchar from 30 to 40
# Add per-facet title (NAME, X, Y) to each facet based on NAMES, X_COORDS, Y_COORDS
assert obs_ds_subset2.pixel.size == len(NAMES)
for i in range(obs_ds_subset2.pixel.size):
    ax = gr.axs.flat[i]
    label = f"{NAMES[i]} — ({X_COORDS[i]}, {Y_COORDS[i]})"
    # ax.set_title(label, fontsize="small")
    ax.text(0.02, 1.15, label,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize="large",
        bbox=dict(facecolor="none", alpha=0.7, edgecolor="none")
    )
plt.tight_layout()


# save plot:    
plotpath = (PROC_ZARR+"-TESTSUITE_Fig0"+site_selection+".png")
plt.savefig(plotpath)

# save plot for report:
plotpath = os.path.join(
    "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig", 
    os.path.basename(PROC_ZARR)+"-TESTSUITE_Fig0"+site_selection+".png")
plt.savefig(plotpath)
plt.close()





# Figure 2,3 Maps (taken from tmp_area_visualization.py):

# Test subsetting:
# area  = select_area_box_by_xy(proc_ds_midx, 2684595, 1295915, Lx=100, Ly=20)
# area

# make_map(2761095, 1194615, "_lowland_ENF", "Highland evergreen")
# X_coord, Y_coord, fname_suffix, site_name = 2761095, 1194615, "_lowland_ENF", "Highland evergreen"
# X_coord, Y_coord, fname_suffix, site_name = 2644325, 1134345, "_bitsch_nonfire", "Bitsch fire nearby non-affected area"
# X_coord, Y_coord, fname_suffix, site_name = 2690025, 1287413, "_drought2018", "2018 Drought-affected area"
# X_coord, Y_coord, fname_suffix, site_name = 2689564, 1154411, "_storm", "Storm affected area"

def make_map(X_coord, Y_coord, fname_suffix, site_name, Lx=10000, Ly=10000, DATE_A = FIGURE02_MAP_DATE_A, DATE_B = FIGURE02_MAP_DATE_B):
    print(f"\n--- Processing: {fname_suffix} ---")
    
    # subset # TODO: start using X_COORDS and Y_COORDS
    proc_ds_area  = select_area_box_by_xy(proc_ds_midx, X_coord, Y_coord, Lx=Lx, Ly=Ly)#.drop(["y_idx","x_idx"])
    obs_ds_area   = select_area_box_by_xy(obs_ds_midx,  X_coord, Y_coord, Lx=Lx, Ly=Ly)#.drop(["y_idx","x_idx"])
    
    # 1. Subset Historical
    proc_sub = proc_ds_area # NOTE: uses date
    
    # 2. Subset Observations (raw data)
    obs_sub = (obs_ds_area.
        # subset first entry for each 'date', remove datetime make date main index
        isel(datetime = ~obs_ds_area.date.to_index().duplicated()).
        # rename datetime to date (actually subset first time of each day)
        swap_dims({"datetime": "date"}) # NOTE: used datetime, now uses date
    )
    
    # Plotting function:
    # ax, grid, title, cmap = axes[1, 1], obs_b, f"Obs Raw – {DATE_B}", cmap
    def _panel(ax, grid, title, cmap="RdYlGn", vmin=0, vmax=10000):
        extent = tuple(grid.rio.bounds()[i] for i in [0,2,3,1]) # reorder to have xmin, xmax, ymax, ymin
        origin = "upper"
        
        # variant 1:
        # cmap_obj = cmap

        # ax.set_facecolor('white')
        # im = ax.imshow(grid, origin=origin, extent=extent, 
        #                cmap=cmap_obj, vmin=vmin, vmax=vmax, 
        #                aspect="equal", interpolation="nearest")

        # variant 2:
        # prepare arrays (accept xarray or numpy input)
        grid_arr = np.asarray(grid).astype(float)

        # prepare masks
        nodata_mask = (grid_arr == NO_COVERAGE) | (grid_arr == INVALID)
        grid_arr[(grid_arr == NO_COVERAGE) | (grid_arr == INVALID)] = np.nan

        # scale to NDVI range (0..1 expects vmin/vmax in same units)
        # grid_scaled = grid_arr # / 10000.0
        # grid_scaled[(grid_scaled < -1) | (grid_scaled > 1)] = np.nan

        # layer 1) draw nodata mask as grey background where True (should highlight clouds)
        mask_img = np.zeros_like(grid_arr, dtype=float)
        mask_img[nodata_mask] = 1.0
        cmap_mask = mcolors.ListedColormap(['white', 'darkgrey']) # [background, clouds]
        ax.imshow(mask_img, origin=origin, extent=extent, 
                  cmap=cmap_mask, vmin=0, vmax=1,
                  aspect="equal", interpolation="nearest", zorder=1)

        # layer 2) draw main data on top; NaNs (including original nodata replaced above) will use `bad` color
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad('white', alpha=0) # NaN not shown (transparent: alpha=0)
        # norm = Normalize(vmin=vmin/10000.0, vmax=vmax/10000.0, clip=True)  # clip outside range to endpoints

        im = ax.imshow(grid_arr, origin=origin, extent=extent,
                    cmap=cmap_obj, vmin=vmin, vmax=vmax, 
                    aspect="equal", interpolation="nearest", zorder=2)

        # common code for both variants:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=7)
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.6f}m"))
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.6f}m"))
        ax.set_xlabel("x [m LV95]", fontsize=7)
        ax.set_ylabel("y [m LV95]", fontsize=7)
        
        return im


    # Load Processed Data (Historical Zarr uses 'date')    
    # Create grids
    obs_a  = NDVI_xarray_to_grid(obs_sub,  DATE_A, variable = 'ndvi')
    obs_b  = NDVI_xarray_to_grid(obs_sub,  DATE_B, variable = 'ndvi')    
    proc_a = NDVI_xarray_to_grid(proc_sub, DATE_A, variable = 'ndvi_processed')
    proc_b = NDVI_xarray_to_grid(proc_sub, DATE_B, variable = 'ndvi_processed')
    proc_a_mask = NDVI_xarray_to_grid(proc_sub, DATE_A, variable = 'mask_array')
    proc_b_mask = NDVI_xarray_to_grid(proc_sub, DATE_B, variable = 'mask_array')

    # Plotting
    title_1 = f"Processed – {pd.to_datetime(proc_sub.sel(date=DATE_A).date.values).strftime('%Y-%m-%d')}" #  %Hh%M
    title_2 = f"Processed – {pd.to_datetime(proc_sub.sel(date=DATE_B).date.values).strftime('%Y-%m-%d')}" #  %Hh%M
    title_3 = f"Obs Raw – {pd.to_datetime(obs_sub.sel(date= DATE_A).datetime.values).strftime('%Y-%m-%d %Hh%M')}"
    title_4 = f"Obs Raw – {pd.to_datetime(obs_sub.sel(date= DATE_B).datetime.values).strftime('%Y-%m-%d %Hh%M')}"
    cmap = "RdYlGn"

    # Map 1: NDVI (proc vs obs)
    fig, axes = plt.subplots(2, 2, 
                             figsize=(9, 8), 
                             constrained_layout=True, facecolor='white',
                             sharex=True, sharey=True)
    fig.suptitle(f"{site_name} ({X_coord:.0f}, {Y_coord:.0f})", fontweight="bold")
    _panel(axes[0, 0], proc_a, title_1, cmap=cmap)
    _panel(axes[0, 1], proc_b, title_2, cmap=cmap)
    _panel(axes[1, 0], obs_a,  title_3, cmap=cmap)
    im = _panel(axes[1, 1], obs_b, title_4, cmap=cmap)
    # optimize layout
    for ax in axes.flat: ax.label_outer() # Remove inner tick labels
    #fig.subplots_adjust(wspace=0.05,hspace=0.05) # Reduce spacing between subplots
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label = "NDVI * 10'000")

    # save plot (next to zarr):
    plotpath = (PROC_ZARR+"-TESTSUITE_Fig2"+site_selection+"_"+fname_suffix+".png")
    plt.savefig(plotpath, dpi=180, bbox_inches="tight")
    # save plot (to report folder):
    plotpath = os.path.join("/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig", 
                            os.path.basename(PROC_ZARR)+"-TESTSUITE_Fig2"+site_selection+"_"+fname_suffix+".png")
    plt.savefig(plotpath, dpi=180, bbox_inches="tight")
    plt.close()


    # Map 2: top row: outlier mask, bottom row: NDVI obs
    cmap_mask = plt.get_cmap('PiYG', 5)    # 5 discrete colors
    fig, axes = plt.subplots(3, 2, 
                             figsize=(9, 8/2*3), 
                             gridspec_kw={'hspace':0, 'wspace':0},
                             constrained_layout=True, facecolor='white',
                             sharex=True, sharey=True)
    fig.suptitle(f"{site_name} ({X_coord:.0f}, {Y_coord:.0f})", fontweight="bold")
    _panel(          axes[0, 0], proc_a_mask, title_1, vmin=0, vmax=5, cmap=cmap_mask)
    im_mask = _panel(axes[0, 1], proc_b_mask, title_2, vmin=0, vmax=5, cmap=cmap_mask)
    _panel(          axes[1, 0], proc_a, title_1, cmap=cmap)
    _panel(          axes[1, 1], proc_b, title_2, cmap=cmap)
    _panel(          axes[2, 0], obs_a,  title_3, cmap=cmap)
    im_NDVI = _panel(axes[2, 1], obs_b,  title_4, cmap=cmap)
    # add colorbar
    cbar_NDVI = fig.colorbar(im_NDVI, ax=axes[2, 1],  orientation="vertical", fraction=0.02, pad=0.02, label = "NDVI * 10'000")
    cbar_NDVI = fig.colorbar(im_NDVI, ax=axes[1, 1],  orientation="vertical", fraction=0.02, pad=0.02, label = "NDVI * 10'000")
    cbar_mask = fig.colorbar(im_mask, ax=axes[0, 1],  orientation="vertical", fraction=0.02, pad=0.02, label = "mask")
    cbar_mask.set_ticks(np.array([1,2,3,4,5]) - 0.5) # https://stackoverflow.com/a/18705457
    cbar_mask.set_ticklabels(np.array([1,2,3,4,5]))  # https://stackoverflow.com/a/18705457
    # optimize layout
    for ax in axes.flat: ax.label_outer() # Remove inner tick labels

    # save plot (next to zarr):
    plotpath = (PROC_ZARR+"-TESTSUITE_Fig3"+site_selection+"_"+fname_suffix+".png")
    plt.savefig(plotpath, dpi=180, bbox_inches="tight")
    # save plot (to report folder):
    plotpath = os.path.join("/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig", 
                            os.path.basename(PROC_ZARR)+"-TESTSUITE_Fig3"+site_selection+"_"+fname_suffix+".png")
    plt.savefig(plotpath, dpi=180, bbox_inches="tight")
    plt.close()

    return proc_a

# Plotting 10x10km maps
names_to_process = NAMES[0:5] # remove here the Bitsch nearby site [5], since it is already covered by the Bitsch site (which is [4])
Xcoord_to_process = X_COORDS[0:5]
Ycoord_to_process = Y_COORDS[0:5]
for (nm,x,y) in zip(names_to_process, Xcoord_to_process, Ycoord_to_process):
    print(f"{nm} — ({x},{y}) — {SHORTNAMES[nm]}")
    make_map(x, y, fname_suffix = SHORTNAMES[nm], site_name = nm)

# Plotting 1.2x1.2km maps:
# Drought area (Schaffhausen)
for (nm,x,y) in zip([NAMES[6]], [X_COORDS[6]], [Y_COORDS[6]]):
    print(f"{nm} — ({x},{y}) — {SHORTNAMES[nm]}")
    #make_map(x, y, fname_suffix = SHORTNAMES[nm], site_name = nm,  Lx=1200, Ly=1200, DATE_A="2017-09-24", DATE_B="2018-09-24")
    make_map(x, y, fname_suffix = SHORTNAMES[nm], site_name = nm,  Lx=1200, Ly=1200, DATE_A="2018-08-23", DATE_B="2019-08-25")

# Plotting 1.5x1.5km maps:
# Windthrow area (Airolo): only plot 1500x1500m instead of 10_000x10_000m
for (nm,x,y) in zip([NAMES[7]], [X_COORDS[7]], [Y_COORDS[7]]):
    print(f"{nm} — ({x},{y}) — {SHORTNAMES[nm]}")
    make_map(x, y, fname_suffix = SHORTNAMES[nm], site_name = nm, Lx=1500, Ly=1500, DATE_A = "2019-08-08", DATE_B = "2021-08-12")




# Figure 1 (taken from workflow_implementation/demo/test_all_pixels/7_create_png_for_XY_demoFB.py)
# --- visual comparison with fresh download ----------------------------------
# a) download raw data directly from swisstopo
FLAG_DOWNLOAD = True
for pixel_it in range(0,len(X_COORDS)):
    if FLAG_DOWNLOAD:
        df_raw = download_timeseries_NDVI_singlePixel(
            x=X_COORDS[pixel_it], 
            y=Y_COORDS[pixel_it], 
            start_date = FIGURE01_START_DATE, 
            end_date = FIGURE01_END_DATE)

    # b) subset historic data
    # b1) by coordinate (already done above)

    # b2) by date
    proc_ds_subset2 = proc_ds_subset.sel(date    = slice(pd.to_datetime(FIGURE01_START_DATE), pd.to_datetime(FIGURE01_END_DATE)))
    obs_ds_subset2  = obs_ds_subset.sel(datetime = slice(pd.to_datetime(FIGURE01_START_DATE), pd.to_datetime(FIGURE01_END_DATE)))

    # print(proc_ds_subset2)
    # print(proc_ds_subset2.compute())
    # print(obs_ds_subset2.compute())

    # c) prepare plot
    # Get data
    dates      = proc_ds_subset2["date"].to_numpy()
    ndvi       = proc_ds_subset2["ndvi_processed"].isel(pixel=pixel_it).load().to_numpy() # select first pixel of series
    mask_array = proc_ds_subset2["mask_array"].isel(pixel=pixel_it).load().to_numpy()     # select first pixel of series

    obs_dates  = obs_ds_subset2["datetime"].to_numpy()
    obs_ndvi   = obs_ds_subset2["ndvi"].isel(pixel=pixel_it).load().to_numpy() # select first pixel of series

    print(ndvi)
    print(mask_array)

    # TODO: we did not include medians in historical data cube and need to append it 
    # each time: medians = ds_h["median_ndvi"].isel(date = slice(2800,3265)).load().to_numpy()

    # Filter based on mask_array
    no_obs_to_smooth = mask_array == 0
    no_obs_smoothed = mask_array == 1
    obs_to_smooth = mask_array == 2
    obs_smoothed = mask_array == 3
    outlier_smoothed = mask_array == 4

    # f) make plot
    plt.figure(figsize=(7.2, 4), dpi = 200)

    plt.plot(dates[no_obs_to_smooth], ndvi[no_obs_to_smooth], marker="D", linestyle="None", markersize=2, color ="black",  label = "no obs to smooth") # TODO: what y-values do these have??? They have 32767.
    plt.plot(dates[no_obs_smoothed],  ndvi[no_obs_smoothed],  marker="D", linestyle="None", markersize=2, color ="orange", label = "no obs smoothed")
    plt.plot(dates[obs_to_smooth],    ndvi[obs_to_smooth],    marker="x", linestyle="None", markersize=4, color ="yellow", label = "obs to smooth")
    plt.plot(dates[obs_smoothed],     ndvi[obs_smoothed],     marker="x", linestyle="None", markersize=4, color ="green",  label = "obs smoothed")
    plt.plot(dates[outlier_smoothed], ndvi[outlier_smoothed], marker="x", linestyle="None", markersize=2, color ="red",    label = "outlier smoothed")

    plt.plot(obs_dates, obs_ndvi, 
            marker="x", linestyle="None", markersize=2, color ="black",  label = "raw obs")

    if FLAG_DOWNLOAD:
        # add crosses for raw downloaded observations
        plt.plot(
            df_raw.datetime, 
            df_raw.ndvi_scaled, 
            marker="x", alpha=.5, linestyle="None", markersize=5, color ="blue",
            label = "Raw download")
        # add vertical areas for days with observations
        downloaded_obs_dates = [
            [obs_date.floor("D"), obs_date.ceil("D")]  for obs_date in df_raw.datetime]
        [plt.axvspan(_range[0], _range[1], color='grey', alpha=1.0) for _range in downloaded_obs_dates]

    # TODO: ALSO ADD MEDIANS: plt.plot(dates, medians,color = "black", linestyle="-",label = "median_ndvi")
    plt.ylim(0, 10000)
    # plt.ylim(0, 33000) # TODO: reactivate cropping at 10000
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"{NAMES[pixel_it]} — ({X_COORDS[pixel_it]}, {Y_COORDS[pixel_it]})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # g) output figure
    plotpath = (
        PROC_ZARR+"-"+
        "TESTSUITE_Fig1_"+
        f"{FIGURE01_START_DATE.replace("-","")}to{FIGURE01_END_DATE.replace("-","")}_"+
        f"location_{X_COORDS[pixel_it]}x{Y_COORDS[pixel_it]}"+
        ".png")
    plt.savefig(plotpath)

    plotpath2 = os.path.join(
        "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig", 
        os.path.basename(PROC_ZARR)+"-TESTSUITE_Fig1"+
        f"{FIGURE01_START_DATE.replace("-","")}to{FIGURE01_END_DATE.replace("-","")}_"+
        f"location_{X_COORDS[pixel_it]}x{Y_COORDS[pixel_it]}"+
        ".png")
    plt.savefig(plotpath2)
    plt.close()










            # # Figure 1 (taken from https://github.com/SamanthaBiegel/s2-forest-browning-monitoring/blob/main/notebooks/plot_results.ipynb)

            # indices = [10803621, 71181649, 96658205, 43074504]

            # order = np.argsort(dates)
            # dates_sorted = np.array(dates)[order]
            # # t_sorted = t[order]

            # fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True, dpi=600)
            # axs = axs.flatten()

            # # vmin = np.nanmin([np.nanmin(anomaly_scores[i]) for i in indices])
            # # vmax = -1.5
            # # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            # cmap = plt.cm.magma_r

            # for i, idx in enumerate(indices):
            #     lower = (
            #         double_logistic_function(t_sorted, torch.as_tensor(params_lower[[idx]]).float())
            #         .squeeze()
            #         .numpy()
            #     )
            #     upper = (
            #         double_logistic_function(t_sorted, torch.as_tensor(params_upper[[idx]]).float())
            #         .squeeze()
            #         .numpy()
            #     )
            #     ndvi_vals = np.asarray(ndvi[idx])[order]
            #     ndsi_vals = np.asarray(ndsi[idx])[order]

            #     is_valid = (
            #         (ndvi_vals != -32768)
            #         & (ndvi_vals != 32767)
            #         & (ndsi_vals < 4300)
            #         & ~(np.isnan(ndvi_vals))
            #         & (ndvi_vals <= 10000)
            #         & (ndvi_vals > -1000)
            #     )
            #     ndvi_valid = ndvi_vals[is_valid] / 10000.0
            #     dates_valid = dates_sorted[is_valid]

            #     iqr = np.abs(upper - lower)
            #     low_thr = lower - 1.5 * iqr
            #     high_thr = upper + 1.5 * iqr

            #     ax = axs[i]
            #     ax.plot(dates_sorted, high_thr, ls="--", lw=1, color="tab:green", alpha=0.8)
            #     ax.plot(dates_sorted, low_thr, ls="--", lw=1, color="tab:red", alpha=0.8)
            #     ax.plot(dates_sorted, lower, color="tab:red", lw=1.5)
            #     ax.plot(dates_sorted, upper, color="tab:green", lw=1.5)
            #     ax.fill_between(dates_sorted, lower, upper, color="tab:red", alpha=0.1)

            #     ndvi_interp_low = np.interp(
            #         dates_valid.astype("datetime64[D]").astype(float),
            #         dates_sorted.astype("datetime64[D]").astype(float),
            #         low_thr,
            #     )
            #     ndvi_interp_high = np.interp(
            #         dates_valid.astype("datetime64[D]").astype(float),
            #         dates_sorted.astype("datetime64[D]").astype(float),
            #         high_thr,
            #     )
            #     is_anomaly = ndvi_valid < ndvi_interp_low

            #     scores_valid = anomaly_scores[idx][order][is_valid]
            #     scores_anom = scores_valid[is_anomaly]
            #     colors = cmap(norm(scores_anom))

            #     ax.scatter(
            #         dates_valid[~is_anomaly],
            #         ndvi_valid[~is_anomaly],
            #         color="black",
            #         s=10,
            #         zorder=3,
            #         label="Normal",
            #     )
            #     ax.scatter(
            #         dates_valid[is_anomaly],
            #         ndvi_valid[is_anomaly],
            #         c=colors,
            #         s=15,
            #         zorder=4,
            #         label="Anomaly",
            #     )

            #     ax.set_ylim(-0.1, 1.001)
            #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            #     ax.set_ylabel("NDVI")

            # sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            # fig.colorbar(
            #     sm, ax=axs, orientation="vertical", fraction=0.03, pad=0.02, label="Anomaly score"
            # )

            # plt.tight_layout(rect=[0, 0, 0.88, 1])
            # plt.subplots_adjust(hspace=0.05)

            # plt.savefig("../figs/figure_1.png", bbox_inches="tight", pad_inches=0)

            # plt.show()

