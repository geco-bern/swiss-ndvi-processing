"""
Transpose Swisstopo NDVI/NDSI dataset from (T, N) to (N, T) layout using Dask.
"""
# "Run Python File" in VSCode

import dask.array as da
from dask.distributed import Client, LocalCluster
import shutil
import os

# SOURCE_ZARR = "/data_3/scratch/francesco/processed/ndvi_dataset_spatial.zarr" # TODO ==> appears renamed to all_ndvi_dataset_spatial.zarr
# TRANSPOSED_ZARR = "/data_3/scratch/francesco/processed/ndvi_dataset_temporal.zarr"
SOURCE_ZARR = "../../data/output/01_all_ndvi_dataset_spatial.zarr"    # from the script 1, will be deleted
TRANSPOSED_ZARR = "../../data/output/02-03_ndvi_dataset_temporal.zarr"
DASK_TEMP_DIR = "../../data/temporary"

#if os.path.exists(TRANSPOSED_ZARR):
#    shutil.rmtree(TRANSPOSED_ZARR)

N_WORKERS = 40
MEMORY_LIMIT = "100GB"

def transpose_zarr(source_zarr, target_zarr, component="ndvi"):

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_LIMIT,
        local_directory=DASK_TEMP_DIR,
    )
    client = Client(cluster)
    print(client.dashboard_link) # use this dashboard to follow progress

    src = da.from_zarr(source_zarr, component=component)
    T, N = src.shape

    # transpose to (N, T)
    dst = src.T

    dst_rechunked = dst.rechunk(chunks=(4000, T))

    dst_rechunked.to_zarr(
        target_zarr,
        component=component,
        overwrite=True,
        compute=True
    )

    client.close()

if __name__ == "__main__":
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="ndvi")
    print("done ndvi")
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="ndsi")
    print("done ndsi")

    #if os.path.exists(SOURCE_ZARR):
    #    shutil.rmtree(SOURCE_ZARR)

    shutil.rmtree(DASK_TEMP_DIR)
