import xarray as xr
import numpy as np

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

historical_ndvi_src = "/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v3_compr.zarr"
historical_ndvi = xr.open_zarr(historical_ndvi_src, chunks={})
last_date = historical_ndvi.date.tail(1).values[0]
print(np.datetime_as_string(last_date, unit='D')) # prints YYYY-MM-DD