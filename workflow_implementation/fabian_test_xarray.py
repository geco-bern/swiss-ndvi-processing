# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/fabian_test_xarray.py > /home/francesco/data_scratch/swiss-ndvi-processing/workflow_implementation/output/log/test_xarray.log &

#!/usr/bin/env python3
"""
Benchmark Dask + Xarray performance on NDVI Zarr dataset.
Logs timing results for various access and compute patterns.
"""

import timeit
import socket
from dask.distributed import Client
import xarray as xr
from matplotlib import pyplot as plt


def find_free_port(default_port=1234):
    """Find an available dashboard port (use default if possible)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", default_port))
        port = s.getsockname()[1]
    except OSError:
        s.bind(("", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    return port


def benchmark(ds_main):
    """Run timing benchmarks and print results."""
    timings = {}

    print("\n--- Starting Benchmarks ---")


    timings["load_pixel_time_100_timestep"] = timeit.timeit(
        lambda: ds_main["ndvi"].isel(pixel=slice(0,1))[0:100].load(), number=1
    )


    timings["load_1_pixel_full_time"] = timeit.timeit(
        lambda: ds_main["ndvi"].isel(pixel=slice(0,1)).load(), number=1
    )

    timings["load_1000_pixel_full_time"] = timeit.timeit(
        lambda: ds_main["ndvi"].isel(pixel=slice(0,1000)).load(), number=1
    )


    timings["mean_1_pixel_7_values"] = timeit.timeit(
        lambda: ds_main["ndvi"].isel(pixel=slice(1, 2))[0:8].mean().compute(), number=1
    )

    timings["mean_1_pixel_full_timeserie"] = timeit.timeit(
        lambda: ds_main["ndvi"].isel(pixel=slice(1, 2)).mean().compute(), number=1
    )

    print("\n--- Benchmark Results (seconds) ---")
    for k, v in timings.items():
        print(f"{k:20s}: {v:.4f}")

    print("\nAll timings complete.")
    return timings


def main():
    INPUT_DIR = "/data_3/scratch/francesco/zarr_demo_pixel_chunked_small.zarr/"
    N_WORKERS = 1

    port = find_free_port(1234)

    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit="24GB",
        processes=True,
        dashboard_address=f":{port}",
    )

    ds_main = xr.open_zarr(INPUT_DIR, chunks="auto")

    # Run benchmarks
    benchmark(ds_main)


if __name__ == "__main__":
    main()
