# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/area_visulization/highland_broadleaf_copy.py > output_high_broadleaf_3.log 2>&1 &

import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import torch
import pandas as pd
from scipy.signal import savgol_filter

# ----- Config -----
zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"
out_gif = "ndvi_lowland_broadleaf_non_gapfilled.gif"

z = zarr.open(zarr_path, mode="r")

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# Rectangle corners (UL and BR)
UL_x, UL_y = 2694027.49, 1123239.84
BR_x, BR_y = 2693027.49, 1122239.84

# ----- compute pixel window (row 0 = top) -----
x_min, x_max = min(UL_x, BR_x), max(UL_x, BR_x)
y_min, y_max = min(UL_y, BR_y), max(UL_y, BR_y)

col_min = int(math.floor((x_min - left) / px))
col_max = int(math.floor((x_max - left) / px))

row_min = int(math.floor((top - y_max) / px))
row_max = int(math.floor((top - y_min) / px))

# clip to bounds
col_min = max(0, min(width - 1, col_min))
col_max = max(0, min(width - 1, col_max))
row_min = max(0, min(height - 1, row_min))
row_max = max(0, min(height - 1, row_max))

win_cols = col_max - col_min + 1
win_rows = row_max - row_min + 1
print(f"Window cols {col_min}..{col_max} ({win_cols}), rows {row_min}..{row_max} ({win_rows})")

# ----- load mask -----
mask = np.load(mask_path)
assert mask.shape == (height, width), f"Mask shape {mask.shape} != raster {(height, width)}"

mask_flat = mask.ravel(order="C")
masked_positions = np.flatnonzero(mask_flat)
n_masked = masked_positions.size
print(f"Mask has {n_masked} True pixels.")

# build index map from full array -> masked array
idx_map = np.full(mask_flat.shape[0], -1, dtype=np.int64)
idx_map[masked_positions] = np.arange(n_masked, dtype=np.int64)

# ----- compute flat indices in window -----
rows = np.arange(row_min, row_max + 1, dtype=np.int64)
cols = np.arange(col_min, col_max + 1, dtype=np.int64)
rr, cc = np.meshgrid(rows, cols, indexing="ij")
full_flat_idx = (rr * width + cc).ravel()

masked_idx_in_window = idx_map[full_flat_idx]
is_masked = masked_idx_in_window >= 0
n_masked_in_window = is_masked.sum()
print(f"Pixels in window: {full_flat_idx.size}, masked pixels: {n_masked_in_window}")

if n_masked_in_window == 0:
    raise RuntimeError("No masked pixels in window!")

# ----- open Zarr -----
N, T = z.shape
assert N == n_masked, f"Zarr first-dim {N} != mask True count {n_masked}"

# ----- plotting extent -----
extent = (
    left + col_min * px,
    left + (col_max + 1) * px,
    top - (row_max + 1) * px,
    top - row_min * px,
)

# Limit to first timesteps
d_frames = min(1000, T)


# ----- seasonal cycle fitting -----
ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/data/prova.zarr")
ndvi = ds["ndvi"]
dates = ds["dates"]
params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)

# convert dates to doy
dates_pd = pd.to_datetime(dates)
df = pd.DataFrame({"date": dates_pd})
df_sorted = df.sort_values(by="date")
dates_sorted = df_sorted["date"].values
dates_pd_sorted = pd.to_datetime(dates_sorted)
doy = dates_pd_sorted.dayofyear.values
doy = torch.tensor(doy, dtype=torch.float32)
T_SCALE = 1.0 / 365.0
t = doy.unsqueeze(0).repeat(params_lower.shape[0], 1) * T_SCALE

# Define double logistic function
def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.nn.functional.sigmoid(
        -2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10)
    )
    sigmoid_sen_eos = torch.nn.functional.sigmoid(
        -2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10)
    )
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

lower = double_logistic_function(t[[0]], params_lower[[0]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[0]]).squeeze().cpu().numpy()

median_iqr = upper - (upper - lower) / 2

param_iqr = 1.5
bottom_iqr = 0.9
upper_iqr = 0.1
window_length = 14
polyorder = 2

# ----- open new zarr output -----
#z_out = zarr.open("/home/francesco/data_scratch/swiss-ndvi-processing/lowland_broadleaf.zarr/ndvi")
ndvi_chunk = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)
ndvi_chunk_smoothed = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)

# ----- gapfilling loop -----
# ----- GAPFILLING FUNCTION -----


# ----- GAPFILLING LOOP -----
ndvi_chunk = np.empty((n_masked_in_window, T), dtype=np.float32)
outlier_matrix = np.zeros((n_masked_in_window, T), dtype=bool)





sel = masked_idx_in_window[is_masked].tolist()


random_pixels = sel[0] # pick one pixel index to check

ndvi_series = z[random_pixels, :]

ndvi_gapfilled, outlier_arr = gapfill_timeseries(ndvi_series, lower, upper)

plot_results(
        pixel_idx=random_pixels,
        ndvi_series=ndvi_series / 10000.0,  # normalize for plotting
        ndvi_gapfilled=ndvi_gapfilled,
        outlier_arr=outlier_arr,
        lower=lower,
        upper=upper,
        dates=dates_sorted,   # your precomputed time axis
        save_path=f"pixel_prova.png"
    )

print("done")

ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/sample_seasonal_cycle_parameter_preds.zarr")

lower = double_logistic_function(t[[0]], params_lower[[91]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[91]]).squeeze().cpu().numpy()

ndvi_gapfilled, outlier_arr = gapfill_timeseries(ndvi_series, lower, upper)

plot_results(
        pixel_idx=random_pixels,
        ndvi_series=ndvi_series / 10000.0,  # normalize for plotting
        ndvi_gapfilled=ndvi_gapfilled,
        outlier_arr=outlier_arr,
        lower=lower,
        upper=upper,
        dates=dates_sorted,   # your precomputed time axis
        save_path=f"pixel_prova_2.png"
    )