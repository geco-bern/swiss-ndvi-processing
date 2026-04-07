"""
2_zarr_to_xarray.py

Converts the temporary zarr store (produced by step 1) into a structured
xarray Dataset and writes it to the final OUTPUT_ZARR path.

Run this when step 1 crashed AFTER all downloads completed, i.e. when
OUTPUT_ZARR_TEMP exists and is intact but OUTPUT_ZARR was never written.

HOW TO RUN:
    source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
    python -u 2_zarr_to_xarray.py
"""

import xarray as xr
import dask.array as da
import numpy as np
import zarr
import rasterio
import pandas as pd
from affine import Affine
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
import os
import json

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

#  nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_folder/1.1_fix_satellite_imgaes.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/MS1_script_for_historical_NDVI/new_folder/log/fix.log &


# ==============================================================================
# CONFIGURE — must match exactly what was used in step 1
# ==============================================================================
OUTPUT_ZARR_TEMP = "/data_3/scratch/francesco/processed/new_ndvi_dataset_spatial_tmp_short.zarr"
OUTPUT_ZARR      = "/data_3/scratch/francesco/processed/new_ndvi_dataset_spatial.zarr"
FOREST_MASK_ZARR = "/data_3/francesco/processed/forest_mask_bits.zarr"
PIXEL_CHUNKS     = 10000

NO_COVERAGE = 2**15 - 1   #  32767 — pixels with no data for a given time step
INVALID     = -2**15      # -32768 — filtered pixels (e.g. cloud shadows)

# ==============================================================================
# Hardcoded spatial reference (identical to step 1)
# ==============================================================================
bbox_swisstopo_2056 = BoundingBox(left=2474090.0, bottom=1065110.0,
                                  right=2851370.0, top=1310530.0)
width_swisstopo  = int((bbox_swisstopo_2056.right - bbox_swisstopo_2056.left) / 10)  # 37728
height_swisstopo = int((bbox_swisstopo_2056.top   - bbox_swisstopo_2056.bottom) / 10) # 24542

ref_meta = {
    'transform': Affine(10.0, 0.0,  bbox_swisstopo_2056.left,
                        0.0, -10.0, bbox_swisstopo_2056.top),
    'crs': CRS.from_epsg(2056),
    'width':  np.float64(width_swisstopo),
    'height': np.float64(height_swisstopo),
}

# ==============================================================================
# Load forest mask
# ==============================================================================
print("Loading forest mask...", flush=True)
forest_mask_zarr  = zarr.open(FOREST_MASK_ZARR, mode="r")
forest_mask_shape = (height_swisstopo, width_swisstopo)
forest_mask = (
    np.unpackbits(forest_mask_zarr["bits"][:])
    [:np.prod(forest_mask_shape)]
    .reshape(forest_mask_shape)
)

global_forest_pixelIDs = np.flatnonzero(forest_mask == 1)
N = len(global_forest_pixelIDs)

reference_summary_msg = (
    f"Total of global grid used for pixel ID (based on forest mask): "
    f"\nBox: {bbox_swisstopo_2056}"
    f"\nGrid: {forest_mask.shape} = {forest_mask.size:_} pixels"
    f", of which {N:_} are identified as forest pixels."
)
print(reference_summary_msg, flush=True)

# ==============================================================================
# Open the temporary zarr store written by step 1
# ==============================================================================


# Reconstruct missing zarr v3 metadata for each array
T = 1180  # number of timesteps — check with: ls .../ndvi/c/ | wc -l
N = 105_715_396  # number of forest pixels

array_specs = {
    "ndvi":     {"shape": [T, N], "dtype": "int16",  "fill_value": NO_COVERAGE},
    "ndsi":     {"shape": [T, N], "dtype": "int16",  "fill_value": NO_COVERAGE},
    "timestep": {"shape": [T],    "dtype": "int64",  "fill_value": -9223372036854775808},
}
chunks = {
    "ndvi":     [1, N],
    "ndsi":     [1, N],
    "timestep": [1],
}

for name, spec in array_specs.items():
    meta_path = f"{OUTPUT_ZARR_TEMP}/{name}/zarr.json"
    if not os.path.exists(meta_path):
        print(f"Writing missing metadata for {name}...", flush=True)
        meta = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": spec["shape"],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": chunks[name]}
            },
            "chunk_key_encoding": {"name": "default", "separator": "/"},
            "data_type": spec["dtype"],
            "fill_value": spec["fill_value"],
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "blosc", "configuration": {
                    "cname": "zstd", "clevel": 3, "shuffle": "bitshuffle"
                }}
            ],
            "attributes": {}
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)


print(f"Opening temp zarr store: {OUTPUT_ZARR_TEMP}", flush=True)
# zarr v3 bare arrays: each array was written as its own subdirectory,
# so open each subdirectory path directly as an array
ndvi_arr  = zarr.open_array(store=f"{OUTPUT_ZARR_TEMP}/ndvi",     mode="r")
ndsi_arr  = zarr.open_array(store=f"{OUTPUT_ZARR_TEMP}/ndsi",     mode="r")
times_arr = zarr.open_array(store=f"{OUTPUT_ZARR_TEMP}/timestep", mode="r")

ndvi_da  = da.from_array(ndvi_arr,  chunks=(1, ndvi_arr.shape[1]))
ndsi_da  = da.from_array(ndsi_arr,  chunks=(1, ndsi_arr.shape[1]))
times_da = da.from_array(times_arr, chunks=(1,)).astype("datetime64[ns]").compute()

print(f"  Shape ndvi : {ndvi_da.shape}", flush=True)
print(f"  Shape ndsi : {ndsi_da.shape}", flush=True)
print(f"  Timesteps  : {len(times_da)}", flush=True)

# ==============================================================================
# Build xarray Dataset
# ==============================================================================
print("Building xarray Dataset...", flush=True)

ndvi_xr = xr.DataArray(
    ndvi_da,
    dims=("datetime", "pixel"),
    coords={
        "pixel":    np.arange(ndvi_da.shape[1], dtype=np.int32),
        "datetime": times_da,
    },
    name="ndvi",
).chunk({"pixel": PIXEL_CHUNKS, "datetime": -1})

ndsi_xr = xr.DataArray(
    ndsi_da,
    dims=("datetime", "pixel"),
    coords={
        "pixel":    np.arange(ndsi_da.shape[1], dtype=np.int32),
        "datetime": times_da,
    },
    name="ndsi",
).chunk({"pixel": PIXEL_CHUNKS, "datetime": -1})

ds_out = xr.Dataset({"ndvi": ndvi_xr, "ndsi": ndsi_xr})

# Add a day-level coord for grouping (keeps original 'datetime' untouched)
ds_out = ds_out.assign_coords(date=ds_out.datetime.dt.floor("D"))

# ==============================================================================
# Add x, y, x_idx, y_idx coordinates per pixel
# ==============================================================================
print("Computing pixel coordinates...", flush=True)
trans = ref_meta["transform"]
rows, cols = np.nonzero(forest_mask)
ids = np.arange(len(rows))
xs, ys = rasterio.transform.xy(trans, rows, cols)

coord_lookup = pd.DataFrame({
    "pixel": ids,
    "x":     xs,
    "y":     ys,
    "x_idx": rows,
    "y_idx": cols,
}).set_index("pixel")

pixel_coords         = ds_out.pixel.values
coord_lookup_aligned = coord_lookup.loc[pixel_coords]

ds_out2 = ds_out.assign_coords(
    pixel  = ("pixel", ds_out.pixel.values.astype(np.int32)),
    x      = ("pixel", coord_lookup_aligned["x"].values.astype(np.int32)),
    y      = ("pixel", coord_lookup_aligned["y"].values.astype(np.int32)),
    x_idx  = ("pixel", coord_lookup_aligned["x_idx"].values.astype(np.int32)),
    y_idx  = ("pixel", coord_lookup_aligned["y_idx"].values.astype(np.int32)),
)

ds_out2.attrs["transform_note"]   = str(trans)
ds_out2.attrs["transform_coeffs"] = tuple(float(v) for v in trans)
ds_out2.attrs["transform_instr"]  = "from affine import Affine; t = Affine(*ds.attrs['transform_coeffs'][0:6])"
ds_out2.attrs["description_ndvi"] = "NDVI (scaled int16: -10000 to 10000)"
ds_out2.attrs["description_ndsi"] = "NDSI (scaled int16: -10000 to 10000)"
ds_out2.attrs["nodata"]           = NO_COVERAGE
ds_out2.attrs["cloud_shadow"]     = INVALID
ds_out2.attrs["pixel_definition"] = reference_summary_msg

# ==============================================================================
# Write final output
# ==============================================================================
print(f"Writing to: {OUTPUT_ZARR}", flush=True)
encoding = {
    "ndvi": {
        "_FillValue": NO_COVERAGE,
        "dtype": "int16",
    },
    "ndsi": {
        "_FillValue": NO_COVERAGE,
        "dtype": "int16",
    },
}


ds_out2.to_zarr(
    OUTPUT_ZARR,
    mode="w",
    consolidated=True,
    compute=True,
    encoding=encoding,
)

test = xr.open_zarr(OUTPUT_ZARR, chunks={"date": -1, "pixel": 10000}, mask_and_scale= False)

ndvi_2 = test["ndvi"].isel(pixel = 0).load().to_numpy()

print(ndvi_2)