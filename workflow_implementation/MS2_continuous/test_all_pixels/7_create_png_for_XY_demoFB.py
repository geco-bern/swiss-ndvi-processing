import datetime as dt
import numpy as np
#import statsmodels.api as sm
from dask.distributed import Client
# from dask import visualize
# import dask.array as da
import xarray as xr
import argparse
import os, shutil, sys
import matplotlib.pyplot as plt
#import time
#from numcodecs import blosc, Blosc, zarr3
#from zarr.codecs import BloscCodec

# for the download
import dask.array as da
import pystac_client
import rasterio
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

# HOW TO RUN FROM BASH:
# source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
# SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/7_create_png_for_XY_demoFB.py"
# LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels/7_create_png_for_XY_demoFB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"
# # HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_100kmX100km.zarr"
# # HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365.zarr_bkp"
# HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_100kmX100km.zarr"
# HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_10kmX10km.zarr"
# # X_COORD="2720645"; Y_COORD="1118245"
# # X_COORD="2710385"; Y_COORD="1116375"
# X_COORD="2710005"
# Y_COORD="1109995"
# START_DATE="2025-09-01"
# START_DATE="2025-05-01"
# END_DATE="2026-03-31"
# END_DATE="2025-12-31"
# # python -u $SCRIPT_FILE $HISTO_INPUT $X_COORD $Y_COORD $START_DATE $END_DATE > $LOG_FILE  2>&1 &
# # python -u $SCRIPT_FILE $HISTO_INPUT $X_COORD $Y_COORD $START_DATE $END_DATE --add_raw_download > $LOG_FILE  2>&1 &

    #     X_COORD="2710005" # TODO: check if this is indeed a forest pixel otherwise choose other test option
    #     Y_COORD="1109995" # TODO: check if this is indeed a forest pixel otherwise choose other test option
    #     # X_COORD="2644020" # NOTE: Bitsch forest fire
    #     # Y_COORD="1133790" # NOTE: Bitsch forest fire
    # 
    #     

    #     END_DATE="2026-03-31"
    #     FLAG_DOWNLOAD=True


INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr" # TODO: move to data2


# FOR DEVELOPMENT: x=2710005; y=1109995; start_date = START_DATE; end_date = END_DATE
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


if __name__ == "__main__":

    # PARSE ARGUMENTS:
    parser = argparse.ArgumentParser()

    parser.add_argument("HISTO_ZARR_INPUT",  help="Full path to Zarr folder with historic NDVI data")
    parser.add_argument("X_COORD",           help="X-coordinate of pixel to plot.")
    parser.add_argument("Y_COORD",           help="Y-coordinate of pixel to plot.")
    parser.add_argument("START_DATE",        help="Start date of time series to plot.")
    parser.add_argument("END_DATE",          help="End date of time series to plot.")
    parser.add_argument("--add_raw_download", action="store_true", 
                        help="If provided, raw data for given pixel is freshly downloaded to compare to historic NDVI.")
    args = parser.parse_args()

    HISTO_ZARR_INPUT  = args.HISTO_ZARR_INPUT
    X_COORD           = args.X_COORD
    Y_COORD           = args.Y_COORD
    START_DATE        = args.START_DATE
    END_DATE          = args.END_DATE
    FLAG_DOWNLOAD     = args.add_raw_download
    # if running interactively use e.g.:
    #     HISTO_ZARR_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_10kmX10km.zarr"
    #     X_COORD="2710005" # TODO: check if this is indeed a forest pixel otherwise choose other test option
    #     Y_COORD="1109995" # TODO: check if this is indeed a forest pixel otherwise choose other test option
    #     # X_COORD="2644020" # NOTE: Bitsch forest fire
    #     # Y_COORD="1133790" # NOTE: Bitsch forest fire
    # 
    #     START_DATE="2025-09-01"

    #     END_DATE="2026-03-31"
    #     FLAG_DOWNLOAD=True

    X_COORD, Y_COORD = int(X_COORD), int(Y_COORD)

    # a) check validity of inputs
    assert X_COORD % 5 == 0, "X_COORD must end with 5 (located at pixel center)"
    assert Y_COORD % 5 == 0, "Y_COORD must end with 5 (located at pixel center)"
    # TODO: what else to check?

    # b) load data
    N_WORKERS = 2
    MEMORY_PER_WORKER = '20GB'
    N_THREADS_PER_WORKER = 1
    DASK_TEMP_DIR = "/mnt/data2/UniBe-swiss-ndvi/tmp_data7/"
    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=N_THREADS_PER_WORKER,
        memory_limit=MEMORY_PER_WORKER,
        local_directory= DASK_TEMP_DIR,
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':8347')
    print(client, flush = True)
    print(client.dashboard_link, flush = True) # use this dashboard to follow progress

    ds_h = xr.open_zarr(HISTO_ZARR_INPUT, chunks={})


    # c) subset historic data
    # 1) by date
    ds_h_subset = ds_h.sel(date = slice(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE)))
    
    # 2) by coordinate
    #ds_h_subset.x.compute() # from 2710005 to 2719945
    #ds_h_subset.y.compute() # from 1109995 to 1100005
    indexer = ((ds_h_subset.x==X_COORD) & (ds_h_subset.y==Y_COORD))
    indexer = indexer.compute() # in order to use drop=True, we need to compute indexer so that dimension of result is known.
    ds_h_subset2 = ds_h_subset.where(indexer, drop=True)
    #ds_h_subset2.compute()
    #ds_h_subset2.sizes

    print(ds_h_subset2)
    print(ds_h_subset2.compute())

    # d) download raw data directly from swisstopo
    if FLAG_DOWNLOAD:
        df_raw = download_timeseries_NDVI_singlePixel(
            x=X_COORD, 
            y=Y_COORD, 
            start_date = START_DATE, 
            end_date = END_DATE)

    
    # e) prepare plot

    # Get dates
    dates = ds_h_subset2["date"].to_numpy()
    
    # Get data
    ndvi       = ds_h_subset2["ndvi_processed"].load().to_numpy()[0]
    mask_array = ds_h_subset2["mask_array"].load().to_numpy()[0]

    # TODO: we did not include medians in historical data cube and need to append it each time: medians = ds_h["median_ndvi"].isel(date = slice(2800,3265)).load().to_numpy()

    # Filter based on mask_array
    no_obs_to_smooth = mask_array == 0
    no_obs_smoothed = mask_array == 1
    obs_to_smooth = mask_array == 2
    obs_smoothed = mask_array == 3
    outlier_smoothed = mask_array == 4

    # f) make plot
    plt.figure(figsize=(7.2, 4), dpi = 200)

    plt.plot(dates[no_obs_to_smooth], ndvi[no_obs_to_smooth], marker="D", linestyle="None",markersize=2, color ="black",  label = "no obs to smooth") # TODO: what y-values do these have??? They have 32767.
    plt.plot(dates[no_obs_smoothed],  ndvi[no_obs_smoothed],  marker="D", linestyle="None",markersize=2, color ="orange", label = "no obs smoothed")
    plt.plot(dates[obs_to_smooth],    ndvi[obs_to_smooth],    marker="x", linestyle="None",markersize=4, color ="yellow", label = "obs to smooth")
    plt.plot(dates[obs_smoothed],     ndvi[obs_smoothed],     marker="x", linestyle="None",markersize=4, color ="green",  label = "obs smoothed")
    plt.plot(dates[outlier_smoothed], ndvi[outlier_smoothed], marker="x", linestyle="None",markersize=2, color ="red",    label = "outlier smoothed")

    if FLAG_DOWNLOAD:
        # add crosses for raw downloaded observations
        plt.plot(
            df_raw.datetime, 
            df_raw.ndvi_scaled, 
            marker="x", alpha=.5, linestyle="None", markersize=5, color ="blue",
            label = "Raw download")
        # add vertical areas for days with observations
        obs_dates = [
            [obs_date.floor("D"), obs_date.ceil("D")]  for obs_date in df_raw.datetime]
        [plt.axvspan(_range[0], _range[1], color='grey', alpha=1.0) for _range in obs_dates]

    # TODO: ALSO ADD MEDIANS: plt.plot(dates, medians,color = "black", linestyle="-",label = "median_ndvi")
    # plt.ylim(0, 10000)
    plt.ylim(0, 33000) # TODO: reactivate cropping at 10000
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series of location: {(X_COORD, Y_COORD)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # g) output figure
    plotpath = (
        "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig/prova2/"+
        "TESTSUITE_"+
        f"{os.path.basename(HISTO_ZARR_INPUT)}_"+
        f"{START_DATE.replace("-","")}to{END_DATE.replace("-","")}_"+
        f"location_{X_COORD}x{Y_COORD}"+
        ".png")
    plt.savefig(plotpath)
    plt.close()
    
    print("All done")
