# nohup python -u /home/francesco/data_scratch/swiss-ndvi-processing/demo/notebook/update_parameter.py > /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/update.log 
# tail -f /home/francesco/data_scratch/swiss-ndvi-processing/demo/output/log/update.log



# library loading, nothing interesting
from IPython.display import IFrame, Image, display
import numpy as np
import math
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import xarray as xr
import torch
import torch.nn as nn
import pandas as pd
import statsmodels.api as sm

import gc
import imageio
from io import BytesIO
from affine import Affine
import matplotlib.dates as mdates
import os
from functions import *

save_path = "figure/"

os.makedirs(save_path, exist_ok=True)

ds = zarr.open_group("/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr",mode = "r")
params = ds["params"]
params_lower = params["params_lower"]
params_upper = params["params_upper"]
ndvi = ds["ndvi"]
ndsi = ds["ndsi"]
dates = pd.to_datetime([d.decode("utf-8") for d in ds["dates"][:]])

T_SCALE = 1.0 / 365.0
doy = dates.dayofyear
t = torch.tensor(doy * T_SCALE, dtype=torch.float32)

def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(torch.as_tensor(params, dtype=torch.float32), 1, dim=1)
    mat_minus_sos = torch.nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = torch.nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = torch.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t[:, None]) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = torch.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t[:, None]) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m

order = np.argsort(dates)
dates_sorted = np.array(dates)[order]
t_sorted = t[order]

# Raster info
height, width = 24542, 37728
left, bottom = 2474090.0, 1065110.0
px = 10.0
top = bottom + height * px

mask_path = "/data_2/scratch/sbiegel/processed/forest_mask.npy"


def extract_pixel(UL_x, UL_y,BR_x, BR_y ):

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

    sel = masked_idx_in_window[is_masked].tolist()

    mid = len(sel) // 2

    start = max(0, mid - 12)
    end   = min(len(sel), mid + 13)   

    sel_window = sel[start:end]
    return(sel_window)

def analysis(pixels,output):
    # --- loop over pixels 12 to 24 and create a 4x3 figure ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, pixel in enumerate(pixels):

        lower = double_logistic_function(t_sorted, params_lower[[pixel]]).squeeze().numpy()
        upper = double_logistic_function(t_sorted, params_upper[[pixel]]).squeeze().numpy()
        ndvi_vals = np.asarray(ndvi[pixel], dtype=np.float32)[order]
        ndsi_vals = np.asarray(ndsi[pixel])[order]
        is_snow = ndsi_vals >= 0.43

        is_valid = ((ndvi_vals > 0) & (ndvi_vals <= 10000.0))
        ndvi_vals[~is_valid] = np.nan
        ndvi_vals[is_snow] = np.nan
        ndvi_valid = ndvi_vals[is_valid] / 10000.0
        dates_sorted_valid = dates_sorted[is_valid]

        df = pd.DataFrame({
            'date': dates_sorted_valid,
            'ndvi': ndvi_valid
            })

        df_sorted = df.sort_values(by='date')

        ndvi_sorted = df_sorted['ndvi'].values
        y_delta_l , y_delta_h,r_delta_h, r_delta_l  = -0.25, 0.25,1,-1 #np.quantile(delta_diff, [0.1,0.9,0.95,0.05])
        y_iqr, r_iqr = 0.05,2

        # --- 1. Create initial df ---
        df_starting = df

        # --- 2. Handle duplicate dates ---
        df_starting = (
            df_starting.groupby("date", as_index=False)
            .mean(numeric_only=True)   # average if duplicates exist
            .set_index("date")
        )

        # --- 3. Reindex to full daily series ---
        full_index = pd.date_range(df_starting.index.min(), df_starting.index.max(), freq="D")
        df = df_starting.reindex(full_index)

        # add forecast column for later use
        df["forecast"] = np.nan
        df["upper"] = np.nan
        df["lower"] = np.nan
        df["gapfilled"] = np.nan
        df["outlier"] = True
        df["delta_smoothed"] = np.nan
        df["idx"] = np.arange(len(df))
        df["deltas_L1"] = np.nan
        df["ratio"] = np.nan
        df["delta_delta"] = np.nan
        df["use_delta"] = True
        df["check_std"] = np.nan

        df.index.name = "date"

        # remove data veofre marh 2018
        df = df.loc[df.index >= pd.Timestamp("2018-04-25")]

        # --- run your function row by row ---
        # need an inital condition
        last_date =  None #pd.Timestamp("2018-04-25")
        last_potential_date =None
        deltas_arr = []
        dates_delta_arr = []
        # this is extra just for the plotting
        latency = []
        current_date_latency = []

        for date in df.index[df.index >= pd.Timestamp("2018-04-26")]:
            last_date, last_potential_date_arr, deltas_arr, dates_delta_arr, latency,  current_date_latency = ndvi_continous_final_2(
                df=df,
                date=date,
                params_lower =  params_lower[[pixel]],
                params_upper =  params_upper[[pixel]],
                last_date=last_date,
                deltas_arr = deltas_arr,
                dates_delta_arr = dates_delta_arr,
                last_potential_date = last_potential_date,
                y_delta_l = y_delta_l, 
                y_delta_h = y_delta_h,
                r_delta_h = r_delta_h, 
                r_delta_l = r_delta_l,
                y_iqr= y_iqr, 
                r_iqr = r_iqr,
                tau = 45,
                smoothing_values = 9,
                latency = latency,
                current_date_latency = current_date_latency

            )

        print("ciao", i)

        df_plot = df[(df.index >= "2018-04-20") & (df.index <= df[df["outlier"] == False].index.max())]

        # --- pick the axis for this pixel ---
        ax = axes[i]

        # color code outliers
        color_map = df_plot["outlier"].map({False: "green", True: "red"}).fillna("gray")
        ax.scatter(df_plot.index, df_plot["ndvi"], color=color_map, s=8, label="NDVI")


        df_plot["smoothed_combined"] = np.where(
        df_plot["use_delta"],
        0.5 * (df_plot["upper"] + df_plot["lower"]) + df_plot["delta_smoothed"],
        df_plot["delta_smoothed"]
        )

        # some very wierd bug appears at the interface between the 2 option

        # hot fix, take the mean between the 2 nearest point
        left = df_plot["smoothed_combined"].shift(1)
        right = df_plot["smoothed_combined"].shift(-1)
        mask_neighbors_high = (left > 0.3) & (right > 0.3)

        # replace center point with the mean of the two neighbors
        df_plot.loc[mask_neighbors_high, "smoothed_combined"] = (left + right) / 2

        #print(min(df_plot["smoothed_combined"][np.isfinite(df_plot["smoothed_combined"])]))

        ax.plot(df_plot.index, df_plot["smoothed_combined"] ,color="black", lw=1, label="Smoothed")

        ax.plot(df_plot.index, df_plot["upper"], color="red", alpha=0.7, lw=1)
        ax.plot(df_plot.index, df_plot["lower"], color="red", alpha=0.7, lw=1)
        ax.fill_between(df_plot.index, df_plot["upper"], df_plot["lower"], alpha=0.2, color="red")

        ax.set_ylim(0, 1.0)
        ax.set_title(f"Pixel {pixel}")


    # global labels
    fig.supxlabel("Date")
    fig.supylabel("NDVI")


    fig.savefig(os.path.join(save_path, output), dpi=300, bbox_inches="tight")


# fire
center_x, center_y =  2643749.70, 1133693.64

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")
analysis(sel_1[:12],"fire_1.png")
analysis(sel_1[13:],"fire_2.png")
print("finish fire")


# lowland broadleaf
center_x, center_y =  2694491.82, 1126023.20

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"broad_low_1.png")
analysis(sel_1[13:],"broad_low_2.png")
print("finish broad low")

# highland broadleaf
center_x, center_y =  2692020.28, 1121443.47

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"broad_high_1.png")
analysis(sel_1[13:],"broad_high_2.png")
print("finish broad high")

# lowland evergreen
center_x, center_y =  2761097.61, 1194613.45

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"ever_low_1.png")
analysis(sel_1[13:],"ever_low_2.png")
print("finish ever low")

# highland evergreen
center_x, center_y =  2781537.00, 1182975.00

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"ever_high_1.png")
analysis(sel_1[13:],"ever_high_2.png")
print("finish ever high")

# non fire
center_x, center_y =  2644218.94, 1134325.81

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"non_fire_1.png")
analysis(sel_1[13:],"non_fire_2.png")
print("finish non fire")

# storm
center_x, center_y =  2689564.74, 1154411.88

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"storm_1.png")
analysis(sel_1[13:],"storm_2.png")
print("finish storm")

# drought
center_x, center_y =  2690025.48, 1287413.03

UL_x, UL_y = center_x - 30, center_y - 30 
BR_x, BR_y = center_x + 30, center_y + 30
sel_1 = extract_pixel( UL_x = UL_x, UL_y = UL_y, BR_x = BR_x, BR_y = BR_y) 
print("finish selection")

analysis(sel_1[:12],"drought_1.png")
analysis(sel_1[13:],"drought_2.png")
print("finish drought")
