import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import rioxarray
import zarr
import pandas as pd

outlier = zarr.open("outliers.zarr", mode='r')
store = zarr.open("sample.zarr", mode='r')
quantile_up = zarr.open("quantiles_up.zarr", mode='r')
quantile_down = zarr.open("quantiles_down.zarr", mode='r')
ds = xr.open_zarr("/data_2/scratch/sbiegel/processed/seasonal_cycle_sample_full.zarr")

params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)

dates = store["dates"][:]

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



# select a random layer to mimic incoming data
# param_iqr and qualitle are the same as the previous notebook

param_iqr = 1.5
layer = 789
ndvi = store['ndvi']

layer_ndvi = ndvi[:, layer]
# normalize and filtering
layer_ndvi = layer_ndvi / 100000

valid_entry =  np.where((layer_ndvi >= 0) & (layer_ndvi <= 1))

def gapfill(delta_1, delta_2, iqr_1, iqr_2, distance):

    iqr_1 + ((delta_1- iqr_1) - (delta_2 - iqr_2)) / distance

if len(valid_entry) > 0: # check if any data incoming is not na

    """logic behind continous integration

    a non-outlier can be flagged as potential outlier or "true value"
    if the last non outlier is "true value" 

        if the new data is in the iqr or the delta compared the previous value is small = "true value" -> gapfilling
        else = potential outlier

    if the previous value IS a potential outlier, this means that the previous point is outside the IQR and the delta is high

        do as above

            if the new data is true value, so the previous one it is -> gapfilling
            if the new data is potential outlier, the previous is outlier -> no gapfilling

    the gapfilling will be done as the previous notebook
    artificially select potential outlier (the for cycle will be removed)"""
    
    potential_arr = []
    
    for entry in valid_entry[0]:

        true_indices = np.flatnonzero(outlier[entry,0:layer])[-2:]

        # calculate bounds and iqr

        lower = double_logistic_function(t[[0]], params_lower[[entry]]).squeeze().cpu().numpy()
        upper = double_logistic_function(t[[0]], params_upper[[entry]]).squeeze().cpu().numpy()
        iqr = upper - lower

        ndvi_timeseries = ndvi[entry, :]

        if len(true_indices) == 2:

            if (ndvi_timeseries[ true_indices[1]]> upper[layer] + param_iqr * iqr[layer]) or (ndvi_timeseries[ true_indices[1]] < lower[layer] - param_iqr * iqr[layer]):

                # calculate deltas
                if ndvi_timeseries[true_indices[0]] > upper[layer] + param_iqr * iqr[layer]:
                    delta_1 = ndvi_timeseries[true_indices[0]] -  upper[layer]
                else: 
                    delta_1 = ndvi_timeseries[ true_indices[0]] -  lower[layer]

                if ndvi_timeseries[true_indices[1]] > upper[layer] + param_iqr * iqr[layer]:
                    delta_2 = ndvi_timeseries[true_indices[1]]-  upper[layer]
                else: 
                    delta_2 = ndvi_timeseries[true_indices[1]] -  lower[layer]

                delta = delta_2 - delta_1

                if delta > quantile_up[entry, true_indices[1]] or delta < quantile_down[entry, true_indices[1]]:
                    potential_arr.append(True)
            
            else:
                potential_arr.append(False)
        else:
            potential_arr.append(False)

    potential_arr = np.array(potential_arr, dtype=bool)

print(len(potential_arr))
        
# for now on the true gapfilling process

for entry in valid_entry[0]:

    lower = double_logistic_function(t[[0]], params_lower[[entry]]).squeeze().cpu().numpy()
    upper = double_logistic_function(t[[0]], params_upper[[entry]]).squeeze().cpu().numpy()
    iqr = upper - lower

    true_indices = np.flatnonzero(outlier[entry,0:layer])[-2:]
    # first: check if lies in the iqr or not

    if (ndvi_timeseries[layer] > upper[layer] + param_iqr * iqr[layer]) or (ndvi_timeseries[layer] < lower[layer] - param_iqr * iqr[layer]):
     
        true_indices = np.flatnonzero(outlier[entry,0:layer])[-2:]
            
        if len(true_indices) == 2:

            if potential_arr[entry] == True:

                # calculate deltas
                if ndvi_timeseries[true_indices[0]] > upper[layer]:
                    delta_1 = ndvi_timeseries[true_indices[0]] -  upper[layer]
                    iqr_1 = upper[true_indices[0]]
                else: 
                    delta_1 = ndvi_timeseries[true_indices[0]] -  lower[layer]
                    iqr_1 = lower[true_indices[0]]

                if ndvi_timeseries[layer] > upper[layer]:
                    delta_2 = ndvi_timeseries[layer] -  upper[layer]
                    iqr_2 = upper[entry]
                else: 
                    delta_2 = ndvi_timeseries[layer] -  lower[layer]
                    iqr_2 = lower[entry]

                delta = delta2 - delta_1

                if delta > quantile_up[entry,layer] or delta < quantile_down[entry,layer]:
                    outlier[true_indices[1],layer] = True
                    # new data = "potential outlier"
                    potential_arr.append(False)
                else:
                    outlier[true_indices[1],layer] = False
                    # new data = "true value"
                    potential_arr.append(True)

                    distance = np.range(1,(true_indices[1] - true_indices[0]) +1)
                    distance = np.array(distance)
                    ndvi_timeseries[(true_indices[0]+1):true_indices[1]] = gapfill(delta_1, delta_2, iqr_1, iqr_2, distance)

            else:

                # calculate deltas
                if ndvi_timeseries[true_indices[1]] > upper[layer]:
                    delta_1 = ndvi_timeseries[true_indices[1]] -  upper[layer]
                else: 
                    delta_1 = ndvi_timeseries[true_indices[1]] -  lower[layer]

                if ndvi_timeseries[layer] > upper[layer]:
                    delta_2 = ndvi_timeseries[layer] -  upper[layer]
                else: 
                    delta_2 = ndvi_timeseries[layer] -  lower[layer]

                delta = delta2 - delta_1

                if delta > quantile_up[entry,layer] or delta < quantile_down[entry,layer]:
                    pass
                    # new data = "potential outlier"
                    potential_arr.append(True)
                else:
                    # new data = "true value"
                    potential_arr.append(False)

                    ndvi_timeseries[(true_indices[0]+1):true_indices[1]] = gapfill(delta_1, delta_2, iqr_1, iqr_2, distance)

    else:

        if potential_arr[entry] == False:
            outlier[true_indices[1],layer] = False
            ndvi_timeseries[(true_indices[1]+1):layer] = gapfill(delta_1, delta_2, iqr_1, iqr_2, distance)

        else:

            if ndvi_timeseries[true_indices[1]] > upper[layer]:
                delta_1 = ndvi_timeseries[true_indices[1]] -  upper[layer]
            else: 
                delta_1 = ndvi[true_indices[1],layer] -  lower[layer]

            if ndvi_timeseries[layer] > upper[layer]:
                delta2 = ndvi_timeseries[layer] - upper[layer]
            else: 
                delta2 = ndvi_timeseries[layer] - lower[layer]

            if delta > quantile_up[entry,layer] or delta < quantile_down[entry,layer]:
                outlier[true_indices[1],layer] = True
                # new data = "true value"
                potential_arr.append(False)
                ndvi_timeseries[(true_indices[1]+1):layer] =gapfill(delta_1, delta_2, iqr_1, iqr_2, distance)
            else:
                outlier[true_indices[1],layer] = False
                # new data = "true value"
                potential_arr.append(False)
                ndvi_timeseries[(true_indices[1]+1):layer] = gapfill(delta_1, delta_2, iqr_1, iqr_2, distance)
