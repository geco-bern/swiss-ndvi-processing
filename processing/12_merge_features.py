"""
Merge all feature arrays into a single Zarr array.
"""
from dask.distributed import Client, LocalCluster
import dask.array as da
import zarr
import os

SOURCE_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
TARGET_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
DASK_LOCAL_DIRECTORY = "/data_2/scratch/sbiegel/dask_worker_space"

def merge_features_to_single_array(source_zarr_path, target_zarr_path):
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        processes=True,
        memory_limit="10GB",
        local_directory=DASK_LOCAL_DIRECTORY,
    )
    client = Client(cluster)

    root = zarr.open(source_zarr_path, mode="r", zarr_format=3)
    features_group = root["features"]
    all_keys = list(features_group.array_keys())

    if "habitat" not in all_keys or "tree_species" not in all_keys:
        raise ValueError("Both 'tree_species' and 'habitat' must exist in features group.")

    other_keys = [k for k in all_keys if k not in ("tree_species", "habitat")]
    feature_order = other_keys + ["tree_species", "habitat"]

    arrays = []
    feature_column_map = {}
    col_idx = 0
    for name in feature_order:
        arr = da.from_zarr(os.path.join(source_zarr_path, "features", name))
        if arr.ndim == 1:
            arr = arr[:, None]
        n_cols = arr.shape[1]
        if n_cols == 1:
            feature_column_map[name] = [col_idx]
        else:
            feature_column_map[name] = list(range(col_idx, col_idx + n_cols))
        col_idx += n_cols
        if name == 'forest_mix_rate':
            nodata_value = -128
            arr_masked = da.where(arr == nodata_value, da.nan, arr)
            median_value = da.nanmedian(arr_masked, axis=0).compute()
            arr = da.where(arr == nodata_value, median_value, arr).astype("float32")
            arr = arr / 100.0
        arrays.append(arr)

    merged = da.concatenate(arrays, axis=1)
    merged_rechunked = merged.rechunk((4000, merged.shape[1]))

    merged_rechunked.to_zarr(
        target_zarr_path,
        component="merged_features",
        overwrite=True,
        compute=True,
    )

    store = zarr.open(target_zarr_path, mode="a", zarr_format=3)
    merged_array = store["merged_features"]

    merged_array.attrs["feature_columns"] = feature_column_map

if __name__ == "__main__":
    merge_features_to_single_array(SOURCE_ZARR, TARGET_ZARR)