import os
import subprocess
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import zarr
import shutil
from whitebox.whitebox_tools import WhiteboxTools
from config import REF_TRANSFORM, REF_CRS, REF_WIDTH, REF_HEIGHT, FOREST_MASK, CHUNK_SIZE

TMPDIR = "/data_2/scratch/sbiegel/tmp"
OUT_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr"
DEM_2M = "/data_2/scratch/sbiegel/processed/dem2m.tif"
DEM_10M = os.path.join(TMPDIR, "dem10m.tif")

wbt = WhiteboxTools()
wbt.set_working_dir(TMPDIR)

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)

def resample_raster(src_path, dst_path):
    nodata_val = -9999

    cmd = (
        f"gdalwarp "
        f"-tr 10.0 10.0 " # target resolution
        f"-t_srs EPSG:2056 " # target CRS
        f"-r average " # resampling method
        f"-dstnodata {nodata_val} "
        f"-co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=YES "
        f"-overwrite {src_path} {dst_path}"
    )

    run_cmd(cmd)

def create_topographic_features_2m(dem2m_path):
    run_cmd(f"gdaldem slope {dem2m_path} {TMPDIR}/slope2m.tif -s 1.0 -compute_edges")
    run_cmd(f"gdaldem aspect {dem2m_path} {TMPDIR}/aspect2m.tif -zero_for_flat -compute_edges")
    run_cmd(f"gdaldem TRI {dem2m_path} {TMPDIR}/tri2m.tif -compute_edges")
    run_cmd(f"gdaldem roughness {dem2m_path} {TMPDIR}/roughness2m.tif -compute_edges")

def resample_features_to_10m():
    features = ["slope", "tri", "roughness", "northing", "easting"]
    for feat in features:
        resample_raster(f"{TMPDIR}/{feat}2m.tif", f"{TMPDIR}/{feat}10m.tif")

def create_hydrological_features_10m(dem10m_path):
    filled = f"{TMPDIR}/filled10m.tif"
    d8flow = f"{TMPDIR}/d8flow10m.tif"
    slope = f"{TMPDIR}/slope10m.tif"
    d8slope = f"{TMPDIR}/d8slope10m.tif"
    accum = f"{TMPDIR}/accum10m.tif"
    twi = f"{TMPDIR}/twi10m.tif"

    wbt.fill_depressions(dem10m_path, filled)
    run_cmd(f"mpiexec -n 48 d8flowdir -fel {filled} -p {d8flow} -sd8 {d8slope}")
    run_cmd(f"mpiexec -n 48 aread8 -p {d8flow} -ad8 {accum}")

    wbt.wetness_index(accum, slope, twi)

    wbt.profile_curvature(dem10m_path, f"{TMPDIR}/profile_curv10m.tif")
    wbt.plan_curvature(dem10m_path, f"{TMPDIR}/plan_curv10m.tif")
    wbt.mean_curvature(dem10m_path, f"{TMPDIR}/mean_curv10m.tif")

def create_northing_easting(aspect_path, northing_path, easting_path):
    run_cmd(
        f"gdal_calc.py -A {aspect_path} --outfile={northing_path} "
        f'--calc="sin(A * pi / 180)" --NoDataValue=-9999 --type=Float32 '
        f'--co="COMPRESS=DEFLATE" --co="TILED=YES" --co="BIGTIFF=YES" --overwrite --quiet'
    )
    run_cmd(
        f"gdal_calc.py -A {aspect_path} --outfile={easting_path} "
        f'--calc="cos(A * pi / 180)" --NoDataValue=-9999 --type=Float32 '
        f'--co="COMPRESS=DEFLATE" --co="TILED=YES" --co="BIGTIFF=YES" --overwrite --quiet'
    )

def write_features_to_zarr(tmpdir, forest_mask, zarr_path):
    group = zarr.open_group(zarr_path, mode="a", zarr_format=3)
    feat_grp = group.require_group("features")

    flat_idx = np.flatnonzero(forest_mask)
    N = flat_idx.size

    features = [
        "slope10m", "northing10m", "easting10m", "tri10m", "roughness10m",
        "twi10m", "mean_curv10m", "profile_curv10m", "plan_curv10m", "accum10m"
    ]

    for feat in features:
        path = os.path.join(tmpdir, f"{feat}.tif")
        with rasterio.open(path) as src:
            arr2d = src.read(1)
            dtype = arr2d.dtype
            nodata = src.nodata if src.nodata is not None else -9999
            arr1d = arr2d.ravel()[flat_idx]

        zarr_array = feat_grp.create_array(
            name=feat.replace("10m", ""),
            shape=(N,),
            dtype=np.dtype(dtype),
            fill_value=nodata,
            chunks=(CHUNK_SIZE,),
            overwrite=True
        )
        zarr_array[:] = arr1d

if __name__ == "__main__":
    forest_mask = np.load(FOREST_MASK)
    resample_raster(DEM_2M, DEM_10M)
    print("DEM resampled to 10m resolution", flush=True)
    create_topographic_features_2m(DEM_2M)
    print("Topographic features created at 2m resolution", flush=True)
    create_northing_easting(f"{TMPDIR}/aspect2m.tif", f"{TMPDIR}/northing2m.tif", f"{TMPDIR}/easting2m.tif")
    print("Topographic features created at 2m resolution", flush=True)
    resample_features_to_10m()
    print("Topographic features resampled to 10m resolution", flush=True)
    create_hydrological_features_10m(DEM_10M)
    print("Hydrological features created at 10m resolution", flush=True)
    write_features_to_zarr(TMPDIR, forest_mask, OUT_ZARR)
    print("All features written to Zarr store", flush=True)