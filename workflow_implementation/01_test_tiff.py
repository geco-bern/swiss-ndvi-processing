# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/01_test_tiff.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/test_tiff.log 

import zarr
import numpy as np
import rasterio
from rasterio.transform import from_origin
import random
import os
import gc

# --- Configuration ---
subset_path = "/data_2/scratch/francesco/zarr_demo/"
out_dir = "/home/francesco/data_scratch/swiss-ndvi-processing/demo/output/tiffs"
os.makedirs(out_dir, exist_ok=True)

# --- Raster info (match your subset window) ---
left = 2474090.0
bottom = 1065110.0
height, width = 1301, 1301
px = 10.0
top = bottom + height * px

# --- Open subset ---
z = zarr.open_group(subset_path, mode="r")
ndvi = z["ndvi"][:]  # shape: (time, pixels)
dates = [d.decode("utf-8") for d in z["dates"][:]]

# --- Number of random TIFFs to export ---
n_random = min(100, len(dates))  # exports up to 100 or less if fewer dates
random_indices = random.sample(range(len(dates)), n_random)

print(f"🎲 Exporting {n_random} random NDVI dates...")

# --- Loop through random dates ---
for i in random_indices:
    date = dates[i]
    print(f"🗓️  Exporting NDVI for date: {date}")

    # Extract NDVI (1D flattened masked subset)
    ndvi_flat = ndvi[i, :height * width]
    ndvi_img = np.full((height, width), np.nan, dtype=np.float32)
    ndvi_img.flat[:ndvi_flat.size] = ndvi_flat / 10000.0  # scale to float

    # Define raster transform
    transform = from_origin(left, top, px, px)

    # Output filename
    out_tiff = os.path.join(out_dir, f"ndvi_{date}.tif")

    # --- Write and close explicitly ---
    dst = rasterio.open(
        out_tiff,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs="EPSG:2056",
        transform=transform,
        compress="deflate"
    )
    dst.write(ndvi_img, 1)
    dst.close()  # ✅ explicitly close the file

    # Free memory
    del ndvi_img, ndvi_flat, dst
    gc.collect()

    print(f"✅ Saved {out_tiff}")

print("🎉 Done! All NDVI GeoTIFFs exported.")
