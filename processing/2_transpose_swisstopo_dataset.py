"""
Transpose Swisstopo NDVI/NDSI dataset from (T, N) to (N, T) layout using Dask.
"""
import dask.array as da
from dask.distributed import Client, LocalCluster

from config import TEMPORAL_DATASET_ZARR, SPATIAL_DATASET_ZARR, DASK_LOCAL_DIRECTORY

def transpose_zarr(source_zarr, target_zarr, component="ndvi"):
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        processes=True,
        memory_limit="10GB",
        local_directory=DASK_LOCAL_DIRECTORY,
    )
    client = Client(cluster)

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
    transpose_zarr(SPATIAL_DATASET_ZARR, TEMPORAL_DATASET_ZARR, component="ndvi")
    transpose_zarr(SPATIAL_DATASET_ZARR, TEMPORAL_DATASET_ZARR, component="ndsi")