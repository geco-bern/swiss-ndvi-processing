"""
Transpose Swisstopo NDVI/NDSI dataset from (T, N) to (N, T) layout using Dask.
"""
# "Run Python File" in VSCode

import dask.array as da
from dask.distributed import Client, LocalCluster

INPUT_ZARR_TEMP = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_01_downloadedA.zarr" # or store into /var/tmp/
OUTPUT_ZARR_TEMP = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_ndvi_02-03_downloadedB.zarr" # or store into /var/tmp/

DASK_TEMP_DIR = "/mnt/data1/UniBe-swiss-ndvi/tmp_data"
os.makedirs(DASK_TEMP_DIR, exist_ok=True)

N_WORKERS = 40
MEMORY_LIMIT = "300GB"

def transpose_zarr(src, trgt, component="ndvi"):

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_LIMIT,
        local_directory=DASK_TEMP_DIR,
    )
    client = Client(cluster)
    print(client.dashboard_link) # use this dashboard to follow progress

    src = da.from_zarr(src, component=component)
    T, N = src.shape

    # TODO: transpose and rechunk directly after download to remove this whole script
    # transpose to (N, T)
    dst = src.T

    dst_rechunked = dst.rechunk(chunks=(4000, T))

    dst_rechunked.to_zarr(
        trgt,
        component=component,
        overwrite=True,
        compute=True
    )

    client.close()
    cluster.close()

if __name__ == "__main__":

    transpose_zarr(INPUT_ZARR_TEMP, OUTPUT_ZARR_TEMP, component="ndvi")
    print("done ndvi")

    transpose_zarr(INPUT_ZARR_TEMP, OUTPUT_ZARR_TEMP, component="ndsi")
    print("done ndsi")
