from affine import Affine
import numpy as np
import rasterio
import xarray as xr

# below function to be used with 4_plot_historic_tiff.py and 3_check_historical_ndvi.py
# NDVI_historic, curr_date, variable = proc_sub, DATE_A, 'ndvi_processed'
# NDVI_historic, curr_date, variable = NDVI_historic, curr_date, 'mask_array'
def NDVI_xarray_to_grid(NDVI_historic, curr_date, variable = 'ndvi_processed'):

    # Starting from v4 of our format we have access to trans in the NDVI_historic.note:
    # Instructions are stored: in: NDVI_historic.transform_instr, telling you to do:
    trans = Affine(*NDVI_historic.attrs['transform_coeffs'][0:6])
    
    # Starting from v4 we have access to x_idx and y_idx.
    # Compute minimal bounding window that contains all pixels, then shift indices
    # so the grid uses a compact array sized to that window.
    cols = NDVI_historic.y_idx.values.astype(int) # TODO: this fixes to rotation, but ideally we should fix it in the format definition
    rows = NDVI_historic.x_idx.values.astype(int) # TODO: this fixes to rotation, but ideally we should fix it in the format definition
    #y_coords = NDVI_historic.y.values.astype(int) # NOTE: this is correct, not affected by the rotation
    #x_coords = NDVI_historic.x.values.astype(int) # NOTE: this is correct, not affected by the rotation

    # bounding box in full-raster coordinates
    min_row = int(rows.min())
    min_col = int(cols.min())
    max_row = int(rows.max())
    max_col = int(cols.max())

    # size of the output window
    height = max_row - min_row + 1
    width  = max_col - min_col + 1

    # local indices inside the compact window
    local_rows = (rows - min_row).astype(int)
    local_cols = (cols - min_col).astype(int)

    # Compute affine transform for the compact window from the actual x/y coordinates
    if trans is None:
        raise RuntimeError("affine transform 'trans' not found in NDVI_historic.attrs['transform_coeffs] and is required")

    # derive ordered unique coordinates and pixel size
    ux = np.unique(NDVI_historic.x.values)
    uy = np.unique(NDVI_historic.y.values)
    if ux.size < 2 or uy.size < 2:
        raise RuntimeError("Not enough unique x/y coordinates to derive pixel size")

    dx = abs(trans[0])
    dy = abs(trans[4])

    # origin for Affine.from_origin is top-left: (min_x - dx/2, max_y + dy/2)
    origin_x = float(ux.min() - dx / 2.0)
    origin_y = float(uy.max() + dy / 2.0)

    # create transform with pixel-size in x (dx) and negative y (so row index increases downward, similar to trans[4])
    window_trans = rasterio.Affine(dx, 0.0, origin_x, 0.0, -dy, origin_y)

    # Initialize compact window grid filled with NaN for tiff to be filled with values
    grid = np.full((height, width), np.nan) # TODO: add again: , dtype=np.int16
    grid_x_coords = (window_trans.c + window_trans.a * (np.arange(width ) + 0.5)).astype(int) # origin_x + dx * [...]
    grid_y_coords = (window_trans.f + window_trans.e * (np.arange(height) + 0.5)).astype(int) # origin_y + dy * [...]

    # Fill compact window grid using local indices
    grid[local_rows, local_cols] = NDVI_historic.sel(date=curr_date)[variable].values
    # grid_ndvi[local_rows, local_cols] = NDVI_historic.sel(date=curr_date)['ndvi_processed'].values
    # grid_mask[local_rows, local_cols] = NDVI_historic.sel(date=curr_date)['mask_array'].values
        # mask_array == 0: the data is not an observation and is yet to be smoothed
        # mask_array == 1: the data is not an observation and is smoothed
        # mask_array == 2: the data is an observation and is yet to be smoothed
        # mask_array == 3: the data is an observation and is smoothed
        # mask_array == 4: the data is an observation and is an outlier

    # Transform back into a xarray/rioxarray DataArray that spans the compact x-y-grid
    ds_curr_date_gridded = xr.DataArray(grid, dims=("y", "x"), 
                                        coords={"x": grid_x_coords,"y": grid_y_coords})
    ds_curr_date_gridded = ds_curr_date_gridded.rio.write_transform(window_trans)
    ds_curr_date_gridded = ds_curr_date_gridded.rio.write_crs("EPSG:2056")

    return (ds_curr_date_gridded)
