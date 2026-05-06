# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tmp_area_visualization.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/logs/tmp_area_images.log 2>&1 &

import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import warnings

# ── CONFIGURATION ────────────────────────────────────────────────────────────
HISTO_PATH = "/mnt/data1/UniBe-swiss-ndvi/backup/historical_backup.zarr"
OBS_PATH   = "/mnt/data2/UniBe-swiss-ndvi/historic_data/tmp_2026-04-04_18h16_ndvi_01_downloaded_2017-01-01_2025-12-31.zarr"

DATE_A = "2021-06-10"
DATE_B = "2023-07-18"

warnings.filterwarnings("ignore", category=UserWarning, module="numcodecs.zarr3")

NO_COVERAGE = 32767  
INVALID     = -32768 

names = ["low_broad","high_broad","low_ever","high_ever","fire","non_fire","drought","storm"]
names_better = ["Lowland broadleaf area","Highland broadleaf area","Lowland evergreen area","Highland evergreen area","Biscth fire affected area","Biscth fire non affected area","Drought affected area","Vaia storm affected area"]

X = np.array([2694491, 2692020, 2761097, 2781537, 2644029, 2644328, 2690025, 2689564])
Y = np.array([1126023, 1121443, 1194613, 1182975, 1134128, 1134342, 1287413, 1154411])

# ── helpers ───────────────────────────────────────────────────────────────────

def _mask_and_scale(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(float)
    a[a == NO_COVERAGE] = np.nan
    a[a == INVALID]     = np.nan
    a /= 10000.0
    a[(a < -1) | (a > 1)] = np.nan # NDVI range is -1 to 1
    return a

def _pixel_subset(ds: xr.Dataset, cx: float, cy: float, half: int = 5000):
    x_vals = ds["x"].values.astype(float)
    y_vals = ds["y"].values.astype(float)
    return (np.abs(x_vals - cx) <= half) & (np.abs(y_vals - cy) <= half)

def _to_grid(vals: np.ndarray, xs: np.ndarray, ys: np.ndarray, res: int = 10):
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    ux = np.arange(x_min, x_max + res, res)
    uy = np.arange(y_min, y_max + res, res)
    col_map = {val: i for i, val in enumerate(ux)}
    row_map = {val: i for i, val in enumerate(uy)}
    grid = np.full((len(uy), len(ux)), np.nan)
    for v, x, y in zip(vals, xs, ys):
        if not np.isnan(v) and x in col_map and y in row_map:
            grid[row_map[y], col_map[x]] = v
    extent = [x_min - res/2, x_max + res/2, y_min - res/2, y_max + res/2]
    return grid, extent

def _panel(ax, grid, extent, title, cmap="RdYlGn", vmin=0, vmax=1):
    ax.set_facecolor('white')
    im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.4f}M"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.4f}M"))
    ax.set_xlabel("x [LV95]", fontsize=7)
    ax.set_ylabel("y [LV95]", fontsize=7)
    return im

# ── main ──────────────────────────────────────────────────────────────────────

def make_figure(cx, cy, out_path, name):
    print(f"\n--- Processing: {name} ---")
    
    # Open datasets once
    hist_ds = xr.open_zarr(HISTO_PATH, chunks={}, mask_and_scale=False)
    obs_ds = xr.open_zarr(OBS_PATH, chunks={}, mask_and_scale=False)

    # 1. Subset Historical
    h_mask = _pixel_subset(hist_ds, cx, cy, half=5000)
    if h_mask.sum() == 0:
        print(f"ERROR: No historical pixels found for {name}")
        return
    hist_sub = hist_ds.isel(pixel=h_mask)
    
    # 2. Subset Observations
    o_mask = _pixel_subset(obs_ds, cx, cy, half=5000)
    if o_mask.sum() == 0:
        print(f"ERROR: No observation pixels found for {name}")
        return
    obs_sub = obs_ds.isel(pixel=o_mask)

    # Get coordinate arrays for gridding
    xs = hist_sub["x"].values.astype(float)
    ys = hist_sub["y"].values.astype(float)

    # Load Processed Data (Historical Zarr uses 'date')
    proc_a = _mask_and_scale(hist_sub["ndvi_processed"].sel(date=DATE_A, method="nearest").values)
    proc_b = _mask_and_scale(hist_sub["ndvi_processed"].sel(date=DATE_B, method="nearest").values)

    # Load Raw Data (Observation Zarr uses 'datetime')
    raw_a = _mask_and_scale(obs_sub["ndvi"].sel(datetime=DATE_A, method="nearest").values)
    raw_b = _mask_and_scale(obs_sub["ndvi"].sel(datetime=DATE_B, method="nearest").values)

    # Create grids
    grid_proc_a, ext = _to_grid(proc_a, xs, ys)
    grid_proc_b, _   = _to_grid(proc_b, xs, ys)
    grid_raw_a,  _   = _to_grid(raw_a,  xs, ys)
    grid_raw_b,  _   = _to_grid(raw_b,  xs, ys)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True, facecolor='white')
    fig.suptitle(f"{name}\nCentre: ({cx:.0f}, {cy:.0f})", fontweight="bold")

    cmap = "RdYlGn"
    _panel(axes[0, 0], grid_proc_a, ext, f"Processed – {DATE_A}", cmap=cmap)
    _panel(axes[0, 1], grid_proc_b, ext, f"Processed – {DATE_B}", cmap=cmap)
    _panel(axes[1, 0], grid_raw_a,  ext, f"Obs Raw – {DATE_A}", cmap=cmap)
    im = _panel(axes[1, 1], grid_raw_b, ext, f"Obs Raw – {DATE_B}", cmap=cmap)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("NDVI (0 – 1)")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved figure → {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    for i in range(8):
        plotpath = "/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/report/fig/prova3/"
        OUT_PATH = f"{plotpath}{names[i]}_area.png"

        make_figure(
            cx = X[i],
            cy = Y[i],
            out_path = OUT_PATH,
            name = names_better[i]
        )