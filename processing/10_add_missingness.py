"""
Compute and add NDVI missingness information to the Zarr dataset.
"""
import zarr
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import TEMPORAL_DATASET_ZARR, CHUNK_SIZE

root = zarr.open_group(TEMPORAL_DATASET_ZARR, mode="a")
ndvi = root["ndvi"]
ndsi = root["ndsi"]
dates = pd.to_datetime([d.decode("utf-8") for d in root["dates"][:]])

# Fractional-year bins
dtindex = pd.DatetimeIndex(dates)
doy = dtindex.dayofyear.to_numpy()
is_leap = dtindex.is_leap_year.astype(int)
t = (doy - 1) / (365 + is_leap)

n_samples, total_days = ndvi.shape
batch_size = 4000

n_bins = 365
bin_edges = np.linspace(0, 1, n_bins + 1)
missing_counts_by_bin = np.zeros(n_bins, dtype=np.int64)
sample_counts_by_bin = np.zeros(n_bins, dtype=np.int64)

for i in tqdm(range(0, n_samples, batch_size)):
    i_end = min(i + batch_size, n_samples)
    batch = ndvi[i:i_end, :]
    batch_ndsi = ndsi[i:i_end, :].astype(np.float32) / 10000.0
    batch_nan = (
        np.isnan(batch)
        | (batch == -2**15)
        | (batch == 2**15 - 1)
        | ((batch_ndsi > 0.43) & (batch_ndsi <= 1.0))
    )

    # Bin fractional times
    bin_ids = np.digitize(t, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        missing_counts_by_bin[b] += batch_nan[:, mask].sum()
        sample_counts_by_bin[b] += mask.sum() * batch.shape[0]

missingness_by_bin = missing_counts_by_bin / sample_counts_by_bin
missingness = missingness_by_bin[np.digitize(t, bin_edges) - 1]

if "missingness" in root:
    del root["missingness"]
root.create_array(
    "missingness",
    shape=missingness.shape,
    dtype=missingness.dtype,
    chunks=(CHUNK_SIZE,),
    overwrite=True,
)
root["missingness"][:] = missingness