#!/usr/bin/env python
# python /home/francesco/data_scratch/swiss-ndvi-processing/notebook/continous_integration.py


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import rioxarray
import zarr
import pandas as pd
import imageio
from io import BytesIO

ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/sample_seasonal_cycle_parameter_preds.zarr")
ndvi = ds['ndvi']
dates = ds['dates']

params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)

params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)

# convert dates to doy
dates_pd = pd.to_datetime(dates)

df = pd.DataFrame({
    'date': dates_pd
})

# Sort by date
df_sorted = df.sort_values(by='date')

# Extract the sorted arrays if needed
dates_sorted = df_sorted['date'].values
dates_pd_sorted = pd.to_datetime(dates_sorted)
doy = dates_pd_sorted.dayofyear.values

doy = torch.tensor(doy, dtype=torch.float32)
T_SCALE = 1.0 / 365.0
t = doy.unsqueeze(0).repeat(params_lower.shape[0], 1) * T_SCALE

# Define the double logistic function
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

param_iqr = 1.2

pixel = 32

# this whole part will be skipped in future

lower = double_logistic_function(t[[0]], params_lower[[pixel]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[pixel]]).squeeze().cpu().numpy()

iqr = upper - lower

# extract ndvi 
ndvi_timeseries = ndvi[pixel, :]

# normalization
ndvi_timeseries = ndvi_timeseries / 10000

    # assign cloud mask as nan
ndvi_timeseries = np.where((ndvi_timeseries > 1) | (ndvi_timeseries < 0), np.nan, ndvi_timeseries)
dates_pd = pd.to_datetime(dates)

# proper sorting

df = pd.DataFrame({
    'date': dates_pd,
    'ndvi': ndvi_timeseries
    })

df_sorted = df.sort_values(by='date')

dates_sorted = df_sorted['date'].values
ndvi_sorted = df_sorted['ndvi'].values

valid_idx = np.where(np.isfinite(ndvi_sorted))

previuos_data = ndvi_sorted[0:100]
previuos_data_valid = previuos_data[np.isfinite(ndvi_sorted)[0:100]]

valid_idx = np.where(np.isfinite(previuos_data))
last_three_points = valid_idx[0][-3:]

valid_ndvi = ndvi_sorted[valid_idx[0][-2]:]
valid_data = dates_sorted[valid_idx[0][-2]:]

ndvi_valid_plot = valid_ndvi[np.isfinite(ndvi_sorted[valid_idx[0][-2]:])]
data_valid_plot = valid_data[np.isfinite(ndvi_sorted[valid_idx[0][-2]:])]


last_two_points_mask = np.array([0,0])


if (ndvi_sorted[last_three_points[-1]] > upper[last_three_points[-1]] + iqr[last_three_points[-1]] * param_iqr) or ndvi_sorted[last_three_points[-1]] < lower[last_three_points[-1]] - iqr[last_three_points[-1]] * param_iqr:

    delta = ((ndvi_sorted[last_three_points[-1]]- ( upper[last_three_points[-1]] - iqr[last_three_points[-1]])) - ((ndvi_sorted[last_three_points[-2]] - ( upper[last_three_points[-2]] - iqr[last_three_points[-2]])))) / (last_three_points[-2] - last_three_points[-1])

    if delta > 0.2:

        last_two_points_mask[1] = 2


if (ndvi_sorted[last_three_points[-2]] > upper[last_three_points[-2]] + iqr[last_three_points[-2]] * param_iqr) or ndvi_sorted[last_three_points[-2]] < lower[last_three_points[-2]] - iqr[last_three_points[-2]] * param_iqr:


    delta_1 = ((ndvi_sorted[last_three_points[-2]]- ( upper[last_three_points[-2]] - iqr[last_three_points[-2]])) - ((ndvi_sorted[last_three_points[-3]] - ( upper[last_three_points[-3]] - iqr[last_three_points[-3]])))) / (last_three_points[-3] - last_three_points[-2])

    delta_2 = ((ndvi_sorted[last_three_points[-1]]- ( upper[last_three_points[-1]] - iqr[last_three_points[-1]])) - ((ndvi_sorted[last_three_points[-2]] - ( upper[last_three_points[-2]] - iqr[last_three_points[-2]])))) / (last_three_points[-2] - last_three_points[-1])


    if (delta_1 > 0.2) and (delta_2 > 0.2):

        last_two_points_mask[0] = 1


# to mimick the contnous integration wee need an initial set of data (for example the first 100)

gapfilled_data = np.copy(previuos_data)
outlier_mask = np.copy(last_two_points_mask)

last_known_position = 100 - last_three_points[-1]

days_difference = dates_sorted[last_known_position] - dates_sorted[100 - last_three_points[-2]]


def mimick_continous_integration(pixel, layer):

    global last_known_position
    global gapfilled_data
    global outlier_mask
    global days_difference

    last_known_position += 1

    # Calculate bounds
    lower = double_logistic_function(t[[0]], params_lower[[pixel]]).squeeze().cpu().numpy()
    upper = double_logistic_function(t[[0]], params_upper[[pixel]]).squeeze().cpu().numpy()
    iqr = upper - lower

    # Extract NDVI and normalize
    ndvi_timeseries = ndvi[pixel, :] / 10000
    ndvi_timeseries = np.where((ndvi_timeseries > 1) | (ndvi_timeseries < 0), np.nan, ndvi_timeseries)

    # Sort by date
    df_sorted = pd.DataFrame({'date': pd.to_datetime(dates), 'ndvi': ndvi_timeseries}).sort_values(by='date')
    ndvi_sorted = df_sorted['ndvi'].values

    new_data = ndvi_sorted[100 + layer]

    outlier_new_data = 0  # default assume valid

    if np.isfinite(new_data):

        # Check outlier
        is_outlier = (new_data > upper[100 + layer] + iqr[100 + layer] * param_iqr) or \
                     (new_data < lower[100 + layer] - iqr[100 + layer] * param_iqr)

        if is_outlier:
            # Compare with last known valid value
            previous_data_1 = ndvi_sorted[100 + layer - last_known_position]
            delta = abs(
                (new_data - (upper[100 + layer] - iqr[100 + layer])) -
                (previous_data_1 - (upper[100 + layer - last_known_position] - iqr[100 + layer - last_known_position]))
            ) / last_known_position

            if delta > 0.05:
                outlier_new_data = 2  # mark as potential outlier, skip gapfill

        if outlier_new_data == 0:
            # gapfill from last valid value
            last_true_value = ndvi_sorted[100 + layer - last_known_position]

            if last_true_value > upper[100 + layer - last_known_position] and new_data > upper[100 + layer]:

                delta_1 = last_true_value - upper[100 + layer - last_known_position]
                delta_2 = new_data - upper[100 + layer]

            elif last_true_value > upper[100 + layer - last_known_position] and new_data < upper[100 + layer]:

                delta_1 = last_true_value - upper[100 + layer - last_known_position]
                delta_2 = new_data - lower[100 + layer]
            
            elif last_true_value < upper[100 + layer - last_known_position] and new_data > upper[100 + layer]:

                delta_1 = last_true_value - lower[100 + layer - last_known_position]
                delta_2 = new_data - upper[100 + layer]

            else:
                delta_1 = last_true_value - lower[100 + layer - last_known_position]
                delta_2 = new_data - lower[100 + layer]

            slope = (delta_2 - delta_1) / last_known_position

            multiplier = np.arange(1, last_known_position + 1)  # include last position
            new_gapfilled_data = (upper[101 + layer - last_known_position:101 + layer] - 
                                  iqr[101 + layer - last_known_position:101 + layer] + slope * multiplier)

            gapfilled_data = np.concatenate((gapfilled_data, new_gapfilled_data))

            # If previous last_known_position was marked as outlier, correct it
            if len(outlier_mask) > 0 and outlier_mask[-1] == 2:

                if slope > 0.05:
                    outlier_mask[-1] = 1
                else:
                    outlier_mask[-1] = 0

            last_known_position = 0
            
        outlier_mask = np.append(outlier_mask, outlier_new_data)

    days_difference = dates_sorted[100 +layer] - dates_sorted[100 + layer -last_known_position]

frames = []

# Run the function for the first N layers and capture each frame
for i in range(0, 500):
    mimick_continous_integration(pixel, i)

    fig, ax = plt.subplots(figsize=(12, 6))

    # point obs for plot

    
    # Only plot data up to current layer
    current_len = len(gapfilled_data)
    ax.plot(dates_sorted[100:current_len], lower[100:current_len], label="Lower Bound")
    ax.plot(dates_sorted[100:current_len], upper[100:current_len], label="Upper Bound")
    ax.fill_between(dates_sorted[100:current_len], lower[100:current_len], upper[100:current_len], alpha=0.2, color="red")
    ax.plot(dates_sorted[100:current_len], gapfilled_data[100:current_len], color="black", label="NDVI gapfilled")
    ax.set_title(f"Pixel {pixel} - Step {i} - waiting time {days_difference.astype('timedelta64[D]').astype(int)} - step wating time {last_known_position}")
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("DOY")
    ax.set_ylabel("NDVI")
    ax.legend()
    plt.tight_layout()

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    frames.append(imageio.v2.imread(buf))
    plt.close()

# Save all frames as GIF
imageio.mimsave(f'pixel_{pixel}_ndvi_animation2.gif', frames, duration=0.5)
print(f"GIF saved to pixel_{pixel}_ndvi_animation.gif")