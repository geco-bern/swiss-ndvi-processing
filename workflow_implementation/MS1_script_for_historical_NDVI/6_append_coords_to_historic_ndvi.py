# This script finalizes the historic ndvi data set
# by appending x and y coordinates and also by re-chunking it
# to simplify appending new data without re-writing the whole
# zarr data-structure. (It also uses compression to reduce file size.)
# Thus, the script creates an output file with the following structure:

#### The v5 data structure is the following:
# <xarray.Dataset> Size: 1TB
# Dimensions:         (date: 3164, pixel: 105715396)
# Coordinates:
#   * date            (date) datetime64[ns] 25kB 2017-04-03 ... 2025-11-30
#     doy             (date) int32 13kB dask.array<chunksize=(30,), meta=np.ndarray>
#   * pixel           (pixel) int32 423MB 0 1 2 ... 105715393 105715394 105715395
#     x               (pixel) int32 423MB dask.array<chunksize=(10000,), meta=np.ndarray>
#     y               (pixel) int32 423MB dask.array<chunksize=(10000,), meta=np.ndarray>
#     x_idx           (pixel) int32 423MB dask.array<chunksize=(10000,), meta=np.ndarray>
#     y_idx           (pixel) int32 423MB dask.array<chunksize=(10000,), meta=np.ndarray>
# Data variables:
#     ndvi_processed  (pixel, date) int16 669GB dask.array<chunksize=(10000, 365), meta=np.ndarray>
#     mask_array      (pixel, date) bool 334GB dask.array<chunksize=(10000, 365), meta=np.ndarray>
# Attributes:
#     note:     \n    # Define grid underlying PixelID and needed transformatio...

#### The input structure was the following:
# <xarray.Dataset> Size: 1TB
# Dimensions:         (pixel: 105715396, date: 3164)
# Coordinates:
#   * pixel           (pixel) int64 846MB 0 1 2 ... 105715393 105715394 105715395
#   * date            (date) datetime64[ns] 25kB 2017-04-03 ... 2025-11-30
#     doy             (date) int64 25kB dask.array<chunksize=(365,), meta=np.ndarray>
# Data variables:
#     ndvi_processed  (pixel, date) int16 669GB dask.array<chunksize=(5000, 3164), meta=np.ndarray>
#     mask_array      (pixel, date) bool 334GB dask.array<chunksize=(5000, 3164), meta=np.ndarray>

# source /home/fabian/GitHub/geco-bern/swiss-ndvi-processing/.env/ndvi/bin/activate
# LOG_FILE="../swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/6_append_coords_to_historic_ndvi.py_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"
# nohup python -u ../swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/6_append_coords_to_historic_ndvi.py > $LOG_FILE

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from affine import Affine
import pandas as pd
from dask.distributed import Client, LocalCluster
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec
import time

DATE_CHUNKS = 365
PIXEL_CHUNKS = 40000 # TODO make alternative versions with 40k and 10k. Ideally the chunk size should not be too small. 30x500k ~20MB. 365x40k similar, 365x10k is 4x smaller

if __name__ == "__main__":

    N_WORKERS = 35 # it appears dask uses 2x this number
    MEMORY_LIMIT = "100GB"
    cluster = LocalCluster(
            n_workers=N_WORKERS,
            threads_per_worker=1,
            processes=True,
            memory_limit=MEMORY_LIMIT,
            dashboard_address=":8346",
            # local_directory= DASK_TEMP_DIR,
        )
    client = Client(cluster)
    print(client.dashboard_link) # use this dashboard to follow progress for long-running tasks

    # NOTE: this was run on GECO workstation 02:
    IN_HISTORIC_ZARR    = "/data_3/scratch/francesco/ndvi_processed_all_pixels.zarr"
    IN_FIX_FOR_HISTORIC_ZARR = "/data_3/scratch/francesco/mask_array_fixed.zarr" # this was created on 2026-03-22 with /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/quick_fix_mask_array.py

    #OUT_HISTORIC_ZARR_v3 = "/data_3/scratch/francesco/ndvi_historic_v3.zarr"
    OUT_HISTORIC_ZARR_v5 = "/data_3/scratch/francesco/ndvi_historic_v5.zarr"
    FOREST_MASK      = "/data_3/scratch/francesco/forest_mask.npy" # "/data_2/scratch/sbiegel/processed/forest_mask.npy"

    LOOKUP_TABLE     = "/data_3/francesco/lookup_table_median_ndvi.zarr"



    ndvi_hist   = xr.open_zarr(IN_HISTORIC_ZARR)
    ndvi_hist_fix_mask_array = xr.open_zarr(IN_FIX_FOR_HISTORIC_ZARR)
    ndvi_hist['mask_array'] = ndvi_hist_fix_mask_array['mask_array'] # quickfix

    #ndvi_hist3  = xr.open_zarr(OUT_HISTORIC_ZARR_v3)
    forest_mask = np.load(FOREST_MASK)

    # =====================================================
    #  Bring pixel ID to a set of coordinates
    # =====================================================

    # Define grid underlying PixelID and needed transformations

    # Raster info (of downloaded product)
    ### NOW HARDCODE THIS (see 1_extract_swisstopo_dataset.py in continuous workflow for how these were defined)
    bbox_swisstopo_2056 = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
    width_swisstopo     = int((bbox_swisstopo_2056.right - bbox_swisstopo_2056.left) / 10) # 37728
    height_swisstopo    = int((bbox_swisstopo_2056.top - bbox_swisstopo_2056.bottom) / 10) # 24542
    ### END HARDCODING
    ### NOW HARDCODE THIS (see 1_extract_swisstopo_dataset.py in continuous workflow for how these were defined)
    ref_meta = {
        'transform': Affine(10.0, 0.0,  bbox_swisstopo_2056.left,   # Affine(10.0, 0.0, np.float64(2474090.0),
                            0.0, -10.0, bbox_swisstopo_2056.top),   #        0.0, -10.0, np.float64(1310530.0)), 
        'crs': CRS.from_wkt('PROJCS["CH1903+ / LV95",GEOGCS["CH1903+",DATUM["CH1903+",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],AUTHORITY["EPSG","6150"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4150"]],PROJECTION["Hotine_Oblique_Mercator_Azimuth_Center"],PARAMETER["latitude_of_center",46.9524055555556],PARAMETER["longitude_of_center",7.43958333333333],PARAMETER["azimuth",90],PARAMETER["rectified_grid_angle",90],PARAMETER["scale_factor",1],PARAMETER["false_easting",2600000],PARAMETER["false_northing",1200000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","2056"]]'), 
        'width': np.float64(width_swisstopo), 
        'height': np.float64(height_swisstopo)}
    #unused here: # load forest mask from disk
    #unused here: forest_mask_zarr = zarr.open("/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/data/forest_mask_bits.zarr", mode="r")
    #unused here: forest_mask_shape = (height_swisstopo, width_swisstopo)
    #unused here: forest_mask = np.unpackbits(forest_mask_zarr["bits"][:])[:np.prod(forest_mask_shape)].reshape(forest_mask_shape)
    #unused here: np.array_equiv(forest_mask2, forest_mask) # True, this confirmed that recovery was good.
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

                    # Test how to use this transformation definition:
                    # rasterio.transform.xy(ref_meta['transform'], [0], [0])    # returns center coordinates of first upper left pixel
                    # rasterio.transform.xy(ref_meta['transform'], [10], [10])  # returns center coordinates of tenth upper left pixel (i.e. is more east and more south)
                    # rasterio.transform.xy(ref_meta['transform'], [0, 10], [0, 10]) # this goes from pixel index to coordinate

                    # rows, cols = np.nonzero(forest_mask) # forest_mask = np.load(FOREST_MASK) # TODO: FOREST_MASK is unused
                    # rows, cols = np.nonzero([[1, 0, 1],[0, 0, 0],[0, 0, 0]]) # returns [0 0] and [0 2]
                    # ids = np.arange(len(rows))
                    # xs, ys = rasterio.transform.xy(ref_meta['transform'], rows, cols) # returns:  (array([2474095., 2474115.]), array([1310525., 1310525.]))
                    # rasterio.transform.rowcol(ref_meta['transform'], xs, ys)          # recovers: (array([0, 0]), array([0, 2]))

                    # coord_lookup = pd.DataFrame({
                    #     'pixelID': ids,
                    #     'x': xs,
                    #     'y': ys
                    # })

                    # coords = list(zip(xs, ys))
                    # plt.plot(xs, ys)
                    # plt.plot(coords)

    rows, cols = np.nonzero(forest_mask)
    ids = np.arange(len(rows))
    xs, ys = rasterio.transform.xy(ref_meta['transform'], rows, cols)

    coord_lookup = pd.DataFrame({
        'pixel': ids,
        'x': xs,
        'y': ys,
        'x_idx': rows,
        'y_idx': cols,
    }).set_index('pixel')

    # coord_lookup

    # Check result:
    # fig = plt.figure()
    # pl = plt.scatter(xs, ys)
    # plt.savefig('coords.png')

    # rows and cols can also be derived with x,y and ref_meta['transform']:
    # row_idx, cols_idx = rasterio.transform.rowcol(
    #     ref_meta['transform'], 
    #     ndvi_hist_v5.x.values, 
    #     ndvi_hist_v5.y.values)
    # tiff_dim_height = rows_idx.max() + 1
    # tiff_dim_width = cols_idx.max() + 1

    # =====================================================
    #  Append coordinates to data set (and control number types)
    # =====================================================

    # Align lookup by pixel values
    pixel_coords   = ndvi_hist.pixel.values
    coord_lookup_aligned = coord_lookup.loc[pixel_coords]

    ndvi_hist_v5 = ndvi_hist.assign_coords(
        # change number types of dimensions (pixel is that way 420MB instead of 840MB)
        pixel = ('pixel', ndvi_hist.pixel.values.astype(np.int32)),
        doy   = ('date', ndvi_hist.doy.values.astype(np.int32)),
        # and add coordinates and indices in regular grid
        x=('pixel', coord_lookup_aligned['x'].values.astype(np.int32)), # or uint32
        y=('pixel', coord_lookup_aligned['y'].values.astype(np.int32)), # or uint32
        x_idx=('pixel', coord_lookup_aligned['x_idx'].values.astype(np.int32)), # or uint32
        y_idx=('pixel', coord_lookup_aligned['y_idx'].values.astype(np.int32))  # or uint32
    )

    ndvi_hist_v5.attrs['pixel_definition'] = reference_summary_msg
    # ndvi_hist_v5.attrs['description'] = 'NDVI (scaled int16: -10000 to 10000)'
    #ndvi_hist_v5.attrs['nodata'] = NO_COVERAGE
    #ndvi_hist_v5.attrs['cloud_shadow'] = INVALID


    # =====================================================
    #  Write zarr file
    # =====================================================
    # Note that we DO want to chunk in date dimension since we want to append data.
    # When appeding, whole chunks have to be rewritten, therefore we keep chunks small to keep rewrite small.
    # (see https://discourse.pangeo.io/t/extremely-slow-rechunking-of-zarr-store-with-xarray/1838/6)
    # "For xarray/Dask users: Use region to write to limited regions of existing arrays.", (https://stackoverflow.com/a/78268009)
    # 
    # Regarding zarr chunk size we aim for something bigger than 16MB for performance reasons
    # and ideally also rather large to reduce number of files (NOTE: we do NOT use sharding)
    # currently: chunks of 3164*5000 for a int16 (16 bit = 2 Byte) result in chunks that are 32MB (uncompressed)
    # aim      : chunks of 30*500_000 for a int16 (16bit = 2 Byte) result in chunks that are 30 MB
    out_ds = ndvi_hist_v5

    # # Explicit encoding: no compressor for each data var
    # encoding = {v: {"compressors": None} for v in out_ds.data_vars}
    # t0=time.perf_counter()
    # out_ds.chunk(
    #     {"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS} # for v2 this used to be 5000 and -1
    #     ).to_zarr(
    #         OUT_HISTORIC_ZARR_v5,
    #         mode="w",
    #         consolidated=True,
    #         compute=True,
    #         encoding=encoding,
    #         zarr_format=3
    #         )
    # print(f"Uncompressed storage elapsed: {time.perf_counter()-t0:.3f}s") # NOTE: for 5000x3164 this takes 2633.702 secs to write (with 40 workers)
    # # NOTE: for 500000x30 this takes 3923 secs to write (with 35 workers)

    # # Write compressed zarr
    # Test out 4 different compression method 
    # (for testing insert isel(pixel=range(0,100000)) before `.chunk`)
    # (remove for final run with a single compression method)
    for it in range(5,6): # only run case 5 which appears to be fast and yield good compression
        match it:
            # case 0:
            #     OUT_HISTORIC_ZARR_v5_compr = "/data_3/scratch/francesco/ndvi_historic_v5_uncompressed.zarr"
            #     compressors = None
            # case 1:
            #     OUT_HISTORIC_ZARR_v5_compr = "/data_3/scratch/francesco/ndvi_historic_v5_compressed.zarr"
            #     compressors = zarr3.Blosc() # see https://github.com/pydata/xarray/issues/9987#issuecomment-2631471771
            # case 2:
            #     OUT_HISTORIC_ZARR_v5_compr = "/data_3/scratch/francesco/ndvi_historic_v5_compressed_zstd31.zarr"
            #     compressors = zarr3.Blosc(cname="zstd", clevel=3, shuffle=1)
            # case 3:
            #     OUT_HISTORIC_ZARR_v5_compr = "/data_3/scratch/francesco/ndvi_historic_v5_compressed_zstd91.zarr"
            #     compressors = zarr3.Blosc(cname="zstd", clevel=9, shuffle=1)
            # case 4:
            #     OUT_HISTORIC_ZARR_v5_compr = "/data_3/scratch/francesco/ndvi_historic_v5_compressed_zstd92.zarr"
            #     compressors = zarr3.Blosc(cname="zstd", clevel=9, shuffle=2)
            case 5:
                # OUT_HISTORIC_ZARR_v5_compr = "/data_3/scratch/francesco/ndvi_historic_v5_compressed_zstd32.zarr"
                OUT_HISTORIC_ZARR_v5_compr = f"/data_3/scratch/francesco/ndvi_historic_v5_chk_{PIXEL_CHUNKS}_{DATE_CHUNKS}.zarr"
                compressors = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)

        # Explicit encoding: simple compressor for each data var
        encoding_compr = {v: {"compressors": compressors} for v in out_ds.data_vars}
        t0=time.perf_counter()
        out_ds.chunk(           # for  testing add before .chunk(): .isel(date = range(-100,-0))
            {"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS}
            ).to_zarr(
                OUT_HISTORIC_ZARR_v5_compr, # for  testing add .replace(".zarr", "_100days.zarr"),
                mode="w",
                # consolidated=True,
                compute=True,
                encoding=encoding_compr,
                zarr_format=3)
        print(f"Compress iteration {it}: Elapsed time for full dataset: {time.perf_counter()-t0:.3f}s")
        # NOTE: for 5000x3164 this takes 2633 secs to write (with 40 workers)
        # NOTE: for 500000x30 this takes 2699 secs to write (with 35 workers)
        # NOTE: for 10_000x365 this takes xxxx secs to write (with 35 workers)

        # Also genereate a spatial subsets with a bounding box:
        for it_subset in range(0,4):
            match it_subset:
                case 0:
                    xmin, xmax = 2600000, 2601000
                    ymin, ymax = 1196000, 1197000
                    OUT_HISTORIC_subset = OUT_HISTORIC_ZARR_v5_compr.replace(".zarr", "_1kmX1km.zarr")
                case 1:
                    xmin, xmax = 2710000, 2720000 # focus on Ticino 10x10km
                    ymin, ymax = 1100000, 1110000 # focus on Ticino 10x10km
                    OUT_HISTORIC_subset = OUT_HISTORIC_ZARR_v5_compr.replace(".zarr", "_10kmX10km.zarr")
                case 2:
                    xmin, xmax = 2650000, 2750000 # focus on Ticino 100x100km
                    ymin, ymax = 1070000, 1170000 # focus on Ticino 100x100km
                    OUT_HISTORIC_subset = OUT_HISTORIC_ZARR_v5_compr.replace(".zarr", "_100kmX100km.zarr")
        
            t0=time.perf_counter()
            pixels_subset_mask = (
                (out_ds.x.data >= xmin) &
                (out_ds.x.data <= xmax) &
                (out_ds.y.data >= ymin) &
                (out_ds.y.data <= ymax)
            )
            # sum(pixels_subset_mask) # these are 79 pixesl
            pixels_subset_idx = pixels_subset_mask.nonzero()[0]
            out_ds_subset = out_ds.isel(pixel=pixels_subset_idx)
            
            out_ds_subset.chunk(           # for  testing add before .chunk(): .isel(pixel = range(0,1000000))
                {"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS}
                ).to_zarr(
                    OUT_HISTORIC_subset,
                    mode="w",
                    consolidated=True,
                    compute=True,
                    encoding=encoding_compr,
                    zarr_format=3)
            print(f"It {it}: Elapsed: {time.perf_counter()-t0:.3f}s")
        # NOTE: for 500000x30 chunking (but only 79 pixels) this takes 1.8 secs to write (with 35 workers)
        # NOTE: for 500000x30 chunking (but only 4200 pixels) this takes 5.5 secs to write (with 35 workers)
        # NOTE: for 10_000x365 this takes xxxx secs to write (with 35 workers)


    # =====================================================
    #  Explore these data sets
    # =====================================================
    #xr.open_zarr("/data_3/scratch/francesco/ndvi_historic_v5_compressed_zstd32.zarr")
    #xr.open_zarr("/data_3/scratch/francesco/ndvi_historic_v5_compr.zarr")
    #xr.open_zarr("/data_3/scratch/francesco/ndvi_historic_v5_compr_1000mX1000m.zarr")

    # ndvi_hist3    = xr.open_zarr(OUT_HISTORIC_ZARR_v5)
    # ndvi_hist3    = xr.open_zarr(OUT_HISTORIC_ZARR_v5_compr)
    # lookup        = xr.open_zarr(LOOKUP_TABLE)

    # # explore these data sets
    # ndvi_hist["mask_array"]

    # len(forest_mask)
    # # np.unique(forest_mask) # returns [0 1]

    # first_date = ndvi_hist["date"].isel(date = 0).values
    # first_pixel = min(ndvi_hist["pixel"].values) # 0
    # last_pixel = max(ndvi_hist["pixel"].values)  # 105715395

    # =====================================================
    #  Transfer the data to swisstopo-tunder
    # =====================================================

    # do manually in a terminal:
    # ssh dash
    # tmux
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v2.zarr fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v5.zarr fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v5_compr.zarr fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v5_compr_100days.zarr fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v5_compr_1000mX1000m.zarr fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v5_compr_100kmX100km.zarr fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/
    # rsync --dry-run -avz --human-readable --progress -i -e 'ssh -p 22' /data_3/scratch/francesco/ndvi_historic_v5* fabian-bernhard@tunder.dev.admin.ch:/mnt/data1/UniBe-swiss-ndvi/input_data/