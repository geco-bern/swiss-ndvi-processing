import datetime as dt
import numpy as np
import xarray as xr
import argparse
import os, shutil, sys
import matplotlib.pyplot as plt

def create_png(pixel,path):


    processed = ds.isel(pixel = pixel)
    historical = ds_h.isel(pixel = pixel)
    ndvi = processed["ndvi_processed"].isel(date = slice(2800,3265)).load().to_numpy()
    #    ndvi = historical["ndvi"].isel(date = slice(2800,3265)).load().to_numpy()
    mask_arrray = processed["mask_array"].isel(date = slice(2800,3265)).load().to_numpy()
    medians = historical["median_ndvi"].isel(date = slice(2800,3265)).load().to_numpy()

    # Get dates
    dates = processed["date"].isel(date = slice(2800,3265)).to_numpy()

    # filter based on mask_array
    no_obs_to_smooth = mask_arrray == 0
    no_obs_smoothed = mask_arrray == 1
    obs_to_smooth = mask_arrray == 2
    obs_smoothed = mask_arrray == 3
    outlier_smoothed = mask_arrray == 4

    plt.figure(figsize=(20, 8))

    plt.plot(dates[no_obs_to_smooth], ndvi[no_obs_to_smooth], marker="D", linestyle="None",markersize=2, color ="black", label = "no obs to smooth")
    plt.plot(dates[no_obs_smoothed], ndvi[no_obs_smoothed], marker="D", linestyle="None",markersize=2, color ="orange", label = "no obs smoothed")
    plt.plot(dates[obs_to_smooth], ndvi[obs_to_smooth], marker="D", linestyle="None",markersize=2, color ="yellow", label = "obs to smooth")
    plt.plot(dates[obs_smoothed], ndvi[obs_smoothed], marker="D", linestyle="None",markersize=2, color ="green", label = "obs smoothed")
    plt.plot(dates[outlier_smoothed], ndvi[outlier_smoothed], marker="D", linestyle="None",markersize=2, color ="red", label = "outlier smoothed")

    plt.plot(dates, medians,color = "black", linestyle="-",label = "median_ndvi")
    plt.ylim(0, 10000)
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series of pixel {pixel}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path)
    plt.close()

    

# python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/7_create_png_for_demo.py


input_zarr_src = "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/data_for_demo_2/"
historical_data = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged_small_FB2026-03-17.zarr"

ds = xr.open_zarr(input_zarr_src, chunks={})
ds_h = xr.open_zarr(historical_data, chunks={})

out_dir = "./report/fig/prova"
os.makedirs(out_dir, exist_ok=True)

for i in np.arange(0,100):
    path = f"./report/fig/prova/timeserie_of_pixel_{i}.png"
    create_png(i,path)