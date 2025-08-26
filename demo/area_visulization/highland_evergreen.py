# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/area_visulization/highland_evergreen.py > output_high_ever.log 2>&1 &

import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import torch
import pandas as pd

# ----- Config -----
zarr_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr/ndvi"
mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"

z = zarr.open(zarr_path, mode="r")

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

# Rectangle corners (UL and BR)
UL_x, UL_y = 2782037.00, 1183475.00
BR_x, BR_y = 2783037.00, 1184475.00

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
ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/sample_seasonal_cycle_parameter_preds.zarr")
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

lower = double_logistic_function(t[[0]], params_lower[[91]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[91]]).squeeze().cpu().numpy()

median_iqr = upper - (upper - lower) / 2

param_iqr = 1.5
bottom_iqr = 0.9
upper_iqr = 0.1

# ----- open new zarr output -----
#z_out = zarr.open("/home/francesco/data_scratch/swiss-ndvi-processing/lowland_broadleaf.zarr/ndvi")
ndvi_chunk = np.empty((len(masked_idx_in_window[is_masked]), T), dtype=np.float32)

# ----- gapfilling loop -----
# ----- GAPFILLING FUNCTION -----
def gapfill_timeseries(ndvi_series, lower, upper, param_iqr=1.5, bottom_q=0.3, top_q=0.7):
    """
    Gapfill NDVI time series:
    1. Normalize & mask clouds.
    2. Detect outliers vs logistic model bounds.
    3. Remove outliers.
    4. Linear interpolation towards model median.
    """
    # --- normalize ---
    ndvi_series = ndvi_series.astype(float) / 10000.0
    ndvi_series = np.where((ndvi_series > 1) | (ndvi_series < 0), np.nan, ndvi_series)

    # --- initialize output ---
    ndvi_gapfilled = ndvi_series.copy()
    n = len(ndvi_gapfilled)
    outlier_arr = np.zeros(n, dtype=bool)

    # --- valid points ---
    valid_idx = np.where(np.isfinite(ndvi_series))[0]
    if len(valid_idx) < 2:
        return ndvi_gapfilled, outlier_arr   # nothing to do

    valid_ndvi = ndvi_series[valid_idx]
    valid_upper = upper[valid_idx]
    valid_lower = lower[valid_idx]
    valid_iqr = valid_upper - valid_lower
    median_valid = valid_upper - valid_iqr / 2

    # --- outlier detection (threshold) ---
    th_hi = valid_upper + param_iqr * valid_iqr
    th_lo = valid_lower - param_iqr * valid_iqr
    is_outlier_threshold = (valid_ndvi > th_hi) | (valid_ndvi < th_lo)

    # --- slope/outlier detection ---
    deltas = valid_ndvi - median_valid
    delta_diff = np.diff(deltas)
    if len(delta_diff) > 2:
        q_low, q_hi = np.quantile(delta_diff, [bottom_q, top_q])
        slope_outlier = np.zeros_like(is_outlier_threshold)
        slope_outlier[1:-1] = (
            (delta_diff[1:] > q_hi) | (delta_diff[1:] < q_low)
        ) & (
            (delta_diff[:-1] > q_hi) | (delta_diff[:-1] < q_low)
        )
        is_outlier = is_outlier_threshold & slope_outlier
    else:
        is_outlier = is_outlier_threshold

    # --- remove outliers ---
    outlier_arr[valid_idx] = is_outlier
    ndvi_gapfilled[valid_idx[is_outlier]] = np.nan

    # --- linear interpolation between valid values ---
    valid_idx = np.where(np.isfinite(ndvi_gapfilled))[0]
    if len(valid_idx) < 2:
        return ndvi_gapfilled, outlier_arr

    for i in range(len(valid_idx) - 1):
        start, end = valid_idx[i], valid_idx[i+1]
        if end - start > 1:
            # interpolate relative to model median
            d1 = ndvi_gapfilled[start] - (upper[start] - (upper[start] - lower[start]) / 2)
            d2 = ndvi_gapfilled[end] - (upper[end] - (upper[end] - lower[end]) / 2)
            slope = (d2 - d1) / (end - start)
            for j, idx in enumerate(range(start+1, end)):
                ndvi_gapfilled[idx] = (upper[idx] - (upper[idx] - lower[idx]) / 2) + d1 + slope * (j+1)

    return ndvi_gapfilled, outlier_arr


# ----- GAPFILLING LOOP -----
ndvi_chunk = np.empty((n_masked_in_window, T), dtype=np.float32)
outlier_matrix = np.zeros((n_masked_in_window, T), dtype=bool)

sel = masked_idx_in_window[is_masked].tolist()

for i, pixel_sel in enumerate(sel):
    if i % 1000 == 0:
        print(f"Gapfilling pixel {i}/{len(sel)}")

    ndvi_series = z[pixel_sel, :]   # raw time series
    ndvi_gapfilled, outlier_arr = gapfill_timeseries(ndvi_series, lower, upper)

    ndvi_chunk[i, :] = ndvi_gapfilled
    outlier_matrix[i, :] = outlier_arr


def plot_results(pixel_idx, ndvi_series, ndvi_gapfilled, outlier_arr, lower, upper, dates, save_path=None):
    """
    Plot NDVI time series for a pixel:
      - raw NDVI with outliers highlighted
      - logistic model bounds
      - gapfilled NDVI
    """
    colors = np.where(outlier_arr, "red", "green")

    fig, ax = plt.subplots(figsize=(12, 6))

    # logistic bounds
    ax.plot(dates, lower, label="Lower Bound", color="blue", alpha=0.7)
    ax.plot(dates, upper, label="Upper Bound", color="blue", alpha=0.7)
    ax.fill_between(dates, lower, upper, alpha=0.2, color="blue")

    # raw NDVI with outliers marked
    ax.scatter(dates, ndvi_series, s=10, color=colors, label="Raw NDVI", zorder=3)

    # gapfilled NDVI
    ax.plot(dates, ndvi_gapfilled, color="black", label="Gapfilled NDVI")

    ax.set_title(f"Pixel {pixel_idx}")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")



ndvi_series = z[int(np.median(sel)) , :]

ndvi_gapfilled, outlier_arr = gapfill_timeseries(ndvi_series, lower, upper)

plot_results(
        pixel_idx=int(np.median(sel)),
        ndvi_series=ndvi_series / 10000.0,  # normalize for plotting
        ndvi_gapfilled=ndvi_gapfilled,
        outlier_arr=outlier_arr,
        lower=lower,
        upper=upper,
        dates=dates_sorted,   # your precomputed time axis
        save_path=f"pixel_{int(np.median(sel))}_gapfilled.png"
    )

print("done")


out_gif_combined = "ndvi_highland_evergreen_combined.gif"

frames = []
for t in range(d_frames):

    if t % 100 == 0:
        print(t)

    # --- Non gapfilled ---
    values_non_gapfilled = np.empty(n_masked_in_window, dtype=float)
    start = 0
    while start < n_masked_in_window:
        end = min(start + 100000, n_masked_in_window)
        sel = masked_idx_in_window[is_masked][start:end].tolist()
        values_batch = z[sel, t].astype(float) / 10000.0
        values_batch = np.where((values_batch > 1) | (values_batch < 0), np.nan, values_batch)
        values_non_gapfilled[start:end] = values_batch
        start = end

    window_non_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_non_gapfilled[is_masked] = values_non_gapfilled
    window_non_gapfilled = window_non_gapfilled.reshape((win_rows, win_cols))

    # --- Gapfilled ---
    values_gapfilled = ndvi_chunk[:, t].astype(float)
    window_gapfilled = np.full(win_rows * win_cols, np.nan, dtype=float)
    window_gapfilled[is_masked] = values_gapfilled
    window_gapfilled = window_gapfilled.reshape((win_rows, win_cols))

    # --- Plot side by side with shared colorbar ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6 * win_rows / win_cols), constrained_layout=True)
    
    im0 = axes[0].imshow(window_non_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_title("Non Gapfilled")
    axes[0].set_xlabel("EPSG:2056 (m)")
    axes[0].set_ylabel("EPSG:2056 (m)")
    
    im1 = axes[1].imshow(window_gapfilled, origin="upper", extent=extent, vmin=0, vmax=1, cmap="RdYlGn")
    axes[1].set_title("Gapfilled")
    axes[1].set_xlabel("EPSG:2056 (m)")
    axes[1].set_ylabel("EPSG:2056 (m)")
    
    # Shared colorbar on the right
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label("NDVI")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    frames.append(image)
    plt.close(fig)

imageio.mimsave(out_gif_combined, frames, fps=10)
print("Saved combined GIF:", out_gif_combined)
