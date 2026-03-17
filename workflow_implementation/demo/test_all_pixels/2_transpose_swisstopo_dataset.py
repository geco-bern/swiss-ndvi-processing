"""
Transpose Swisstopo NDVI/NDSI dataset from (T, N) to (N, T) layout using Dask.
"""
# "Run Python File" in VSCode

import dask.array as da
from dask.distributed import Client, LocalCluster
import argparse

def transpose_zarr(src, trgt, component="ndvi"):

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
    # PARSE ARGUMENTS:
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_ZARR", help="Full path of Zarr folder, downloaded with script 1")
    args = parser.parse_args()

    INPUT_ZARR_TEMP = args.INPUT_ZARR
    # if running interactively use e.g.:
    #   INPUT_ZARR_TEMP = "/mnt/data1/UniBe-swiss-ndvi/data/tmp_2026-03-17_16h27_ndvi_01_downloadedA_2025-11-30_2025-12-12.zarr"
    OUTPUT_ZARR_TEMP = INPUT_ZARR_TEMP.replace("01_downloadedA","02-03_downloadedB")

    DASK_TEMP_DIR = "/mnt/data1/UniBe-swiss-ndvi/tmp_data"
    os.makedirs(DASK_TEMP_DIR, exist_ok=True)

    N_WORKERS = 40
    MEMORY_LIMIT = "300GB"
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_LIMIT,
        dashboard_address=":8342",
        local_directory=DASK_TEMP_DIR,
    )
    client = Client(cluster)
    print(client, flush = True)
    print(client.dashboard_link, flush = True) # use this dashboard to follow progress

    transpose_zarr(INPUT_ZARR_TEMP, OUTPUT_ZARR_TEMP, component="ndvi")
    print("done ndvi")

    transpose_zarr(INPUT_ZARR_TEMP, OUTPUT_ZARR_TEMP, component="ndsi")
    print("done ndsi")
