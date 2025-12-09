"""
Transpose Swisstopo NDVI/NDSI dataset from (T, N) to (N, T) layout using Dask.
"""
# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/demo/2_transpose_swisstopo_dataset.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/transpose_swisstopo.log &

import dask.array as da
from dask.distributed import Client, LocalCluster
import shutil
import os

SOURCE_ZARR = "/data_3/scratch/fabian/data-swiss-ndvi-processing_redownload_Apr2017-Nov2025/ndvi_dataset_spatial_Apr2017-Nov2025.zarr"
TRANSPOSED_ZARR = "/data_3/scratch/francesco/processed/all_ndvi_dataset_temporal.zarr"
DASK_LOCAL_DIRECTORY = "/data_3/francesco/dask_worker_space"

if os.path.exists(TRANSPOSED_ZARR):
    shutil.rmtree(TRANSPOSED_ZARR)

def transpose_zarr(source_zarr, target_zarr, component="ndvi"):

    cluster = LocalCluster(
        n_workers=40,
        threads_per_worker=1,
        processes=True,
        memory_limit="100GB",
        local_directory=DASK_LOCAL_DIRECTORY,
    )
    client = Client(cluster)

    print(client.dashboard_link)

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

if __name__ == "__main__":
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="ndvi")
    print("done ndvi")
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="ndsi")
    print("done ndsi")

    shutil.rmtree(DASK_LOCAL_DIRECTORY)

