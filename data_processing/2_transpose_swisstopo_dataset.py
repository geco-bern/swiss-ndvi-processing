import dask.array as da
from dask.distributed import Client, LocalCluster

SOURCE_ZARR = "/data_2/scratch/sbiegel/processed/new_ndvi_timeseries.zarr"
TRANSPOSED_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr"
DASK_LOCAL_DIRECTORY = "/data_2/scratch/sbiegel/dask_worker_space"

def transpose_zarr(source_zarr, target_zarr):
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        processes=True,
        memory_limit="10GB",
        local_directory=DASK_LOCAL_DIRECTORY,
    )
    client = Client(cluster)

    src = da.from_zarr(source_zarr, component="ndvi")
    T, N = src.shape

    # transpose to (N, T)
    dst = src.T

    dst_rechunked = dst.rechunk(chunks=(4000, T))

    dst_rechunked.to_zarr(
        target_zarr,
        component="ndvi",
        overwrite=True,
        compute=True
    )

if __name__ == "__main__":
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR)