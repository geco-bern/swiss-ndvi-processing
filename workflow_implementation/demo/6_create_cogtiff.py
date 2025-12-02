from datetime import datetime, date
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
import zarr
import math

INPUT_ZARR = "data_for_demo/processed_ndvi.zarr"
COG_TIFF_FOLDER = "data_for_demo/output_cogtiff/"

ds = xr.open_zarr(INPUT_ZARR)

pixel = ds["pixel"].values

# Just for the first time, select the spinup (2018-01-01)

select_date_str = "2018-02-01"

select_date = np.datetime64(select_date_str)

select_date_arr = np.array([select_date])

mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

for day in select_date_arr:

    ndvi_layer =  ds["ndvi_processed"].sel(date = day).values

    # Raster info
    height, width = 24542, 37728
    left, bottom = 2474090.0, 1065110.0
    px = 10.0
    top = bottom + height * px

    # ----- center cooridnates  -----
    center_x, center_y = 2694491.82, 1126023.20
    # Rectangle corners (UL and BR)
    UL_x, UL_y = center_x - 300, center_y - 300 
    BR_x, BR_y = center_x + 300, center_y + 300


    # ----- compute pixel window (row 0 = top) -----
    x_min, x_max = min(UL_x, BR_x), max(UL_x, BR_x)
    y_min, y_max = min(UL_y, BR_y), max(UL_y, BR_y)

    col_min = int(math.floor((x_min - left) / px))
    col_max = int(math.floor((x_max - left) / px))

    row_min = int(math.floor((top - y_max) / px))
    row_max = int(math.floor((top - y_min) / px))

    # clip to bounds
    col_min = max(0, min(width - 1, col_min))
    col_max = max(0, min(width - 1, col_max))
    row_min = max(0, min(height - 1, row_min))
    row_max = max(0, min(height - 1, row_max))

    win_cols = col_max - col_min + 1
    win_rows = row_max - row_min + 1
    print(f"Window cols {col_min}..{col_max} ({win_cols}), rows {row_min}..{row_max} ({win_rows})")

    # ----- load mask -----
    mask = np.load(mask_path)
    assert mask.shape == (height, width), f"Mask shape {mask.shape} != raster {(height, width)}"

    mask_flat = mask.ravel(order="C")
    masked_positions = np.flatnonzero(mask_flat)
    n_masked = masked_positions.size
    print(f"Mask has {n_masked} True pixels.")

    # build index map from full array -> masked array
    idx_map = np.full(mask_flat.shape[0], -1, dtype=np.int64)
    idx_map[masked_positions] = np.arange(n_masked, dtype=np.int64)

    # ----- compute flat indices in window -----
    rows = np.arange(row_min, row_max + 1, dtype=np.int64)
    cols = np.arange(col_min, col_max + 1, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    full_flat_idx = (rr * width + cc).ravel()

    masked_idx_in_window = idx_map[full_flat_idx]
    is_masked = masked_idx_in_window >= 0
    n_masked_in_window = is_masked.sum()
    print(f"Pixels in window: {full_flat_idx.size}, masked pixels: {n_masked_in_window}")

    sel = masked_idx_in_window[is_masked].tolist()

    values = np.empty(n_masked_in_window, dtype=float)
    window = np.full(win_rows * win_cols, np.nan, dtype=float)


    window[is_masked] = ndvi_layer
    window = window.reshape((win_rows, win_cols))


    arr = window.astype('int16')


    x_min = 2694491 - 300
    y_max = 1126023 - 300
    pixel_width = 10
    pixel_height = 10

    transform = from_origin(x_min, y_max, pixel_width, pixel_height)

    COG_TIFF_OUTPUT = COG_TIFF_FOLDER + day.astype(str) + ".tif"

    with rasterio.open(
        COG_TIFF_OUTPUT,
        'w',
        driver='COG',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs='EPSG:2056', 
        transform=transform,
        nodata=np.nan,
        compress='deflate',
        tiled=True
    ) as dst:
        dst.write(arr, 1) 
