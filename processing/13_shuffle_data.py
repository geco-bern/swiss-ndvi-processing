"""
Filter and shuffle NDVI dataset for training neural networks.
Set NDVI to missing where NDSI indicates snow/ice.
"""
import zarr
import numpy as np
import math
import time
from tqdm import tqdm

SOURCE_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
TARGET_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_filtered_shuffled.zarr"

def filter_and_shuffle_ndvi(source_zarr, target_zarr,
                            block_rows=8192,  # multiple of 8192
                            chunk_rows=8192,
                            ndsi_min=4300,
                            ndsi_max=10000,
                            seed=42):
    rng = np.random.default_rng(seed)

    src_ndvi = zarr.open(f"{source_zarr}/ndvi", mode="r")
    src_ndsi = zarr.open(f"{source_zarr}/ndsi", mode="r")
    src_features = zarr.open(f"{source_zarr}/merged_features", mode="r")

    N, T = src_ndvi.shape
    n_blocks = math.ceil(N / block_rows)

    root = zarr.open_group(target_zarr, mode="w")
    ndvi_tgt = root.create_array(
        "ndvi",
        shape=(N, T),
        chunks=(chunk_rows, T),
        dtype=np.int16,
    )
    features_tgt = root.create_array(
        "merged_features",
        shape=src_features.shape,
        chunks=(chunk_rows, src_features.shape[1]),
        dtype=src_features.dtype,
    )

    perm = rng.permutation(N)
    print(f"Generated global permutation of {N:,} rows")

    for bi in tqdm(range(n_blocks)):
        start = bi * block_rows
        end = min((bi + 1) * block_rows, N)
        idx = perm[start:end]
        idx.sort()
        print("idx sorted")

        ndvi_block = src_ndvi[idx, :]
        print("ndvi_block loaded")
        ndsi_block = src_ndsi[idx, :]
        print("ndsi_block loaded")
        features_block = src_features[idx, :]
        print("features_block loaded")

        mask = (ndsi_block > ndsi_min) & (ndsi_block < ndsi_max)
        ndvi_block[mask] = np.int16(-2**15)
        print("ndvi_block filtered")

        local_perm = rng.permutation(len(idx))
        ndvi_block = ndvi_block[local_perm]
        features_block = features_block[local_perm, :]
        print("ndvi_block shuffled")

        ndvi_tgt[start:end] = ndvi_block
        print("ndvi_block written")
        features_tgt[start:end, :] = features_block

if __name__ == "__main__":
    filter_and_shuffle_ndvi(SOURCE_ZARR, TARGET_ZARR)