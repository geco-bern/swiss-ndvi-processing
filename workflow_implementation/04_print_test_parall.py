# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/04_print_test_parall.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/plot_paralles.log &

# print the result and check
# the pixel input are visible in the 

from datetime import datetime, date, timedelta 
import zarr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def zarr_date_to_date(zarr_date):
    zarr_date = unwrap_scalar(zarr_date)

    if isinstance(zarr_date, bytes):
        return datetime.strptime(zarr_date.decode("utf-8"), "%Y-%m-%d").date()
    elif isinstance(zarr_date, np.datetime64):
        return zarr_date.astype('M8[D]').astype(datetime).date()
    elif isinstance(zarr_date, datetime):
        return zarr_date.date()
    elif isinstance(zarr_date, date):
        return zarr_date
    else:
        raise TypeError(f"Unknown date type: {type(zarr_date)}")

def unwrap_scalar(x):

    while isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    return x

def _parse_zarr_date(d_val):
    if isinstance(d_val, (bytes, np.bytes_)):
        return pd.to_datetime(d_val.decode("utf-8"), errors="coerce")
    else:
        s = str(d_val)
        if s.startswith("np.bytes_("):
            m = re.search(r"b['\"]([^'\"]+)['\"]", s)
            if m:
                s = m.group(1)
        return pd.to_datetime(s, errors="coerce")

root = zarr.open("/data_2/scratch/francesco/zarr_demo_daily_output", mode="r")
#original_root = zarr.open("/data_2/scratch/francesco/zarr_demo_daily/", mode="r")



dates_zarr = root["dates"]


dates = []
for i in range(dates_zarr.shape[0]):
    d_arr = dates_zarr.get_basic_selection((i,))
    d_val = d_arr[()] if isinstance(d_arr, np.ndarray) and d_arr.shape == () else d_arr[0]
    d_dt = _parse_zarr_date(d_val)
    if pd.notna(d_dt):
        dates.append(d_dt)

dates = sorted(list(set(dates)))

# Access the actual NDVI array
ndvi_arr = root["ndvi"]
#observed_ndvi = original_root["ndvi"]

# Extract NDVI time series
base_date = zarr_date_to_date(dates[0])
date_list = [zarr_date_to_date(d) for d in dates[:1500]]

import matplotlib
matplotlib.use("Agg")   

import matplotlib.pyplot as plt
import os

out_dir = "/home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/figure/test_parallelization/"
os.makedirs(out_dir, exist_ok=True)  

pixels = [564891, 316517, 924896, 475939, 136766, 384152 ,100670, 210095, 562418, 471642,
 338199, 310485, 701026,  65215, 936349
 ]



for pixel in pixels:
    ndvi_series = ndvi_arr[:1500, pixel] / 10000.0

    plt.figure(figsize=(10, 5))
    plt.plot(date_list, ndvi_series, lw=1, label="NDVI") 
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series for Pixel {pixel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"output Pixel_{pixel}.png")  
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close() 
    print(f"Saved {filename}")
