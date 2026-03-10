import xarray as xr
import numpy as np

historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_processed_all_pixels_v3_compr.zarr" # TODO: is this the main file that is extended? So in the full workflow this would be circular, i.e. 04_merged_ndvi.zarr ?
historical_ndvi = xr.open_zarr(historical_ndvi_src, chunks={})
last_date = historical_ndvi.date.tail(1).values[0]
print(np.datetime_as_string(last_date, unit='D')) # prints YYYY-MM-DD