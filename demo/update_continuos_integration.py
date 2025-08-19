# python /home/francesco/data_scratch/swiss-ndvi-processing/notebook/update_continuos_integration.py


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
from scipy.signal import savgol_filter

ds = xr.open_zarr("/home/francesco/data_scratch/swiss-ndvi-processing/sample_seasonal_cycle_parameter_preds.zarr")
outlier_bands = zarr.open("/home/francesco/data_scratch/swiss-ndvi-processing/notebook/outliers_continous_integration.zarr", mode="r")
ndvi = ds['ndvi']
dates = ds['dates']



lower_band = outlier_bands[:, 0, :]   # all pixels, lower band
upper_band = outlier_bands[:, 1, :]  

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

pixel = 32
param_iqr = 1.5
upper_iqr = 0.6
bottom_iqr = 0.4
window_length = 15
polyorder = 2

    # caclulate lower and upper bonds
lower = double_logistic_function(t[[0]], params_lower[[pixel]]).squeeze().cpu().numpy()
upper = double_logistic_function(t[[0]], params_upper[[pixel]]).squeeze().cpu().numpy()
# initialize outlier flag
outlier_arr = np.repeat(False,len(dates))

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

    # initialize ndvi gapfilled
ndvi_gapfilled = np.copy(ndvi_sorted)

valid_idx = np.where(np.isfinite(ndvi_sorted))
valid_ndvi = ndvi_sorted[valid_idx]
valid_idx = np.array(valid_idx[0])
valid_upper = upper[valid_idx]
valid_lower = lower[valid_idx]

iqr = upper - lower
valid_iqr = valid_upper - valid_lower

median = upper - iqr / 2
median_valid = valid_upper - valid_iqr / 2


#################################
## outlier detection first 100 ##
#################################

last_three = valid_idx[valid_idx< 100]
last_three = last_three[-3:]

# calculate threshold
delta_threshold_upper = valid_upper[last_three] + param_iqr * valid_iqr[last_three]
delta_threshold_lower = valid_lower[last_three] - param_iqr * valid_iqr[last_three]

# outlier detection by threshold

deltas = []
is_outlier_threshold = []

for i in range(0,len(last_three)):

    if valid_ndvi[i] > valid_upper[i]:

            delta = valid_ndvi[i] - valid_upper[i]

            if valid_ndvi[i] > delta_threshold_upper[i]:
                
                outlier = True
            else:
                outlier = False
    
    elif (valid_ndvi[i] < valid_upper[i]) and (valid_ndvi[i] > valid_lower[i]):
          
        delta = valid_ndvi[i] - (valid_upper[i] - valid_iqr[i])
        outlier = False

    else:
            delta = valid_ndvi[i] - valid_lower[i]

            if valid_ndvi[i] < delta_threshold_lower[i]:
                 
                 outlier = True
            else:
                 outlier = False

    deltas.append(delta)
    is_outlier_threshold.append(outlier)
    
is_outlier_threshold = np.array(is_outlier_threshold)
deltas = np.array(deltas)

# outlier detection by deltas neighbour
# the deltas are calculated based on the neaster bound

delta_delta_left = (deltas[1:] - deltas[:-1]) 
delta_delta_right = (deltas[:-1] - deltas[1:]) 

delta_delta_left = np.array(delta_delta_left)
delta_delta_right = np.array(delta_delta_right)

slope_is_outlier = np.logical_and(
        
        np.logical_or(delta_delta_left > np.quantile(delta_delta_left,upper_iqr), 
                      delta_delta_left  <  np.quantile(delta_delta_left,bottom_iqr)),

        np.logical_or(delta_delta_right  > np.quantile(delta_delta_right,upper_iqr), 
                      delta_delta_right  < np.quantile(delta_delta_right,bottom_iqr))
    )

# to be an outlier, a point must met both conditions
is_outlier = np.logical_and(is_outlier_threshold[1:],slope_is_outlier)

is_not_outlier = ~is_outlier

last_two = last_three[1:]


last_known_position =  last_two[is_not_outlier == True][-1]


gapfilled_data = np.copy(ndvi_sorted)


last_delta = ndvi_sorted[last_known_position] - median[last_known_position]

for i in np.arange(101,1080):
     
    new_data = ndvi_sorted[i]

    new_outlier = 0

    if np.isfinite(new_data):

        current_delta = new_data - median[i]

        slope = current_delta - last_delta / (i - last_known_position)
         
        if (new_data > upper[i] + iqr[i] * param_iqr or  new_data < lower[i] - iqr[i] * param_iqr) and (delta_delta > 0.5 or delta_delta < 0.5):
                 
            new_outlier = 2

            potential_outlier = i
            potential_delta = current_delta


        if new_outlier == 0:
            
            # gapfilling

            """if i - last_known_position == 1:
                # gapfilled_data = np.append(gapfilled_data,new_data)
            
            else:"""
            if  i - last_known_position > 1:

                valid_idx = np.arange(last_known_position + 1, i)

                multiplier = np.arange(1,len(valid_idx) +1)

                new_gapfilled_data = median[valid_idx] + last_delta + slope * multiplier

                gapfilled_data[valid_idx] = new_gapfilled_data


            last_known_position = i
            last_delta = current_delta
    
    is_not_outlier = np.append(is_not_outlier, new_outlier)



fig, ax = plt.subplots(figsize=(12, 6)) 


ax.plot(dates_sorted, lower, label="Lower Bound")
ax.plot(dates_sorted, upper, label="Upper Bound")
ax.fill_between(dates_sorted, lower, upper, alpha=0.2, color="red")
ax.scatter(dates_sorted, ndvi_sorted, s=10, color="black", label="NDVI", zorder=3)
ax.plot(dates_sorted, gapfilled_data, color="black", label="NDVI gapfilled")
ax.set_title(f"Pixel {pixel}")
ax.set_ylim(-0.1, 1)
ax.set_xlabel("DOY")
ax.set_ylabel("NDVI")
ax.legend()

plt.tight_layout()

plt.savefig("prova_integration.png", dpi=300, bbox_inches="tight")



             


