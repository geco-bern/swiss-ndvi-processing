# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py> /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tif.log 2>&1 &

import numpy as np
import xarray as xr
import zarr
import dask.array as da


# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_create_cogtiff.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/6_tiff.log 2>&1 &


# Paths
INPUT_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/05_processed_ndvi.zarr"
OUTPUT_NC = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/2018-06-01.nc"

store = zarr.open(INPUT_ZARR, mode='r')
print(f"arrays: {list(store.keys())}")


target_date_idx = 180
print(f"Loading ONLY date index {target_date_idx}...")

# CORRECT SLICING for (pixels, dates) shape
ndvi_array = store['ndvi_processed']

# **RIGHT SLICE**: all pixels, date 365
ndvi_single_date = da.from_array(ndvi_array[:, target_date_idx], chunks=100000)
x_da = da.from_array(store['x'][:], chunks=100000)
y_da = da.from_array(store['y'][:], chunks=100000)
date_da = da.from_array(store['date'][:], chunks='auto')

mask_array = store['mask_array']

mask_array_single_date = da.from_array(mask_array[:, target_date_idx], chunks=100000)


print(f"NDVI slice shape: {ndvi_single_date.shape}")  # Should be (105715396,)
print(f"X/Y shape: {x_da.shape}")

# Build single-date Dataset
ds_single = xr.Dataset({
    'ndvi': (['pixel'], ndvi_single_date),
    "mask": (['pixel'], ndvi_single_date),
    'x': (['pixel'], x_da),
    'y': (['pixel'], y_da)
})

selected_date = int(date_da[target_date_idx].compute().item())
print(f"Date {target_date_idx} = day {selected_date}")
print(f"Pixels: {len(ds_single.pixel):,}")

# Metadata
ds_single.attrs = {
    'crs': 'EPSG:2056',
    'date_idx': target_date_idx,
    'date_value': selected_date,
    'total_pixels': len(ds_single.pixel)
}

# Your exact dtypes
encoding = {
    'ndvi': {'zlib': True, 'complevel': 5, 'dtype': np.int16},
    'mask': {'zlib': True, 'complevel': 5, 'dtype': np.int8},
    'x': {'zlib': True, 'dtype': np.int32},
    'y': {'zlib': True, 'dtype': np.int32}
}

ds_single.to_netcdf(OUTPUT_NC, encoding=encoding)
print("saved")
