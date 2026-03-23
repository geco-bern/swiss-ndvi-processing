import xarray as xr
import numpy as np
import argparse

# HOW TO RUN FROM BASH:
# source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
# SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/0_2_get_last_date.py"
# HISTO_INPUT="/mnt/data1/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr"
# python -u $SCRIPT_FILE $HISTO_INPUT

parser = argparse.ArgumentParser()
parser.add_argument("HISTO_INPUT",   help="Full path to Zarr folder with historic NDVI data")
args = parser.parse_args()

HISTO_ZARR           = args.HISTO_INPUT
# if running interactively use e.g.:
    # HISTO_ZARR = "/mnt/data1/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_1000mX1000m.zarr"
    # HISTO_ZARR = "/mnt/data1/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_1kmX1km.zarr"
import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

historical_ndvi = xr.open_zarr(HISTO_ZARR, chunks={})
last_date = historical_ndvi.date.tail(1).values[0]
print(np.datetime_as_string(last_date, unit='D')) # prints YYYY-MM-DD