import zarr
import rasterio
from rasterio.windows import Window
from affine import Affine
from tqdm import tqdm
from config import REF_BBOX as bbox_swisstopo_2056

if __name__ == "__main__":
    root = zarr.open(
        "/data_2/scratch/sbiegel/processed/full_dem_2m.zarr",
        mode="r",
        zarr_format=3
    )

    dem2m = root["dem_2m"]

    ny, nx = dem2m.shape
    chunk_h, chunk_w = dem2m.chunks

    ref_bounds   = bbox_swisstopo_2056
    ref_h_2m     = int((ref_bounds.top - ref_bounds.bottom) / 2.0)
    ref_w_2m     = int((ref_bounds.right - ref_bounds.left) / 2.0)
    transform_2m = Affine(2.0, 0.0, ref_bounds.left,
                        0.0,-2.0, ref_bounds.top)

    profile = {
        "driver":     "COG",
        "dtype":      dem2m.dtype,
        "count":      1,
        "crs":        "EPSG:2056",
        "transform":  transform_2m,
        "height":     ref_h_2m,
        "width":      ref_w_2m,
        "tiled":      True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress":   "deflate",
        "BIGTIFF":   "YES",
    }

    cog_path = "/data_2/scratch/sbiegel/processed/dem2m_2.tif"
    with rasterio.open(cog_path, "w", **profile) as dst:

        nchunks_y = (ny + chunk_h - 1) // chunk_h
        nchunks_x = (nx + chunk_w - 1) // chunk_w

        for ci in tqdm(range(nchunks_y), desc="row‐chunks"):
            for cj in range(nchunks_x):
                i0 = ci * chunk_h
                j0 = cj * chunk_w
                i1 = min(i0 + chunk_h, ny)
                j1 = min(j0 + chunk_w, nx)
                h  = i1 - i0
                w  = j1 - j0

                tile = dem2m[i0:i1, j0:j1]

                window = Window(j0, i0, w, h)
                dst.write(tile, 1, window=window)