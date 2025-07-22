# once the wrokstation is back online, check if is working
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ds = xr.open_zarr("C:/Users/Lenovo/Downloads/dimpeo-main/seasonal_cycle_sample.zarr")

# Get data
params_lower = torch.tensor(ds["params_lower"].values)
params_upper = torch.tensor(ds["params_upper"].values)
ndvi = torch.tensor(ds["ndvi"].values)
doy = torch.tensor(ds["doy"].values, dtype=torch.float32)
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

# select valid data (last 2 are non na)
valid_pixel = np.array([])

for pixel in range(0,99): # to modify
   
    ndvi_vals = ndvi[pixel].cpu().numpy()
    ndvi_vals = ndvi_vals[len(ndvi_vals)-3:len(ndvi_vals)]
    
    # interpolate if last is na (assuming that the previous one are not)
    if any(np.isnan(ndvi_vals[0:2])) == True:
        pass
    else: # data are ok
        valid_pixel = np.append(valid_pixel,int(pixel))


pixel_with_na = np.array([])
ndvi_gapfilled_arr = np.array([])
pixel_ok = np.array([])
ndvi_ok = np.array([])
is_outlier = np.array([])

threshold = 0.1

for pixel in valid_pixel:
    ndvi_vals = ndvi[int(pixel)].cpu().numpy()

    if np.isnan(ndvi_vals[-1]) == True: # gapfill
        lower = double_logistic_function(t[[pixel]], params_lower[[pixel]]).squeeze().cpu().numpy()
        upper = double_logistic_function(t[[pixel]], params_upper[[pixel]]).squeeze().cpu().numpy()
        ndvi_vals = ndvi_vals[-3:]
        upper = upper[-3:]
        lower = lower[-3:]
        # check which deltas are closer
        delta_up = sum(abs(ndvi_vals - upper))
        delta_down = sum(abs(ndvi_vals - lower))
        if delta_up > delta_down:
            new_delta = 2 * (ndvi_vals[1] - upper[1]) - (ndvi_vals[0] - upper[0])
            ndvi_gapfilled = upper[2] + new_delta
            outlier = abs(new_delta) > threshold


        else:
            new_delta = 2 * (ndvi_vals[1] - lower[1]) - (ndvi_vals[0] - lower[0])
            ndvi_gapfilled = lower[2] + new_delta
        pixel_with_na = np.append(pixel_with_na,pixel)
        ndvi_gapfilled_arr = np.append(ndvi_gapfilled_arr,ndvi_gapfilled)
        outlier = abs(new_delta) > threshold

    else:
        pixel_ok = np.append(pixel_ok,pixel)
        ndvi_ok = np.append(ndvi_ok,ndvi_vals[-1])
        if ndvi_vals[-1] > upper:
            outlier = ndvi_vals[-1] > upper + threshold
        else:
            outlier = ndvi_vals[-1] < lower - threshold
    
    is_outlier = np.append(is_outlier,outlier)


# in this case all are na so no pixel ok

all_pixel = np.concatenate([pixel_ok, pixel_with_na])
all_ndvi = np.concatenate([ndvi_ok, ndvi_gapfilled_arr])

valid_pixel= valid_pixel.astype(int)
subset_ds = ds.isel(pixel=valid_pixel)
print(subset_ds)

for i in range(1,len(all_pixel)):
    subset_ds["ndvi"][i,-1] = all_ndvi[i] 

# gapfilling

for pixel in range(0,99): 

    ndvi_vals = ndvi[pixel].cpu().numpy()
    lower = double_logistic_function(t[[pixel]], params_lower[[pixel]]).squeeze().cpu().numpy()
    upper = double_logistic_function(t[[pixel]], params_upper[[pixel]]).squeeze().cpu().numpy()

    delta_up = (ndvi_vals - upper)
    delta_down = (ndvi_vals - lower)

    len_arr = len(ndvi_vals)

    valid_idx = np.where(np.isfinite(ndvi_vals))
    distances = valid_idx[0][1:] - valid_idx[0][:-1]
    # first and last need to gapfill in other way
    valid_deltas =  delta_up[np.where(np.isfinite(delta_up))]


    slopes = (valid_deltas[1:] - valid_deltas[:-1]) /distances

    for i in range(0,len(valid_idx[0])-1):
        idx_to_gapfill = range(valid_idx[0][i]+1,valid_idx[0][i+1])
        idx_to_gapfill = np.array(idx_to_gapfill)

        if len(idx_to_gapfill) != 0: # if len == 0 means 2 contigous obs data
            multiplier = range(1,len(idx_to_gapfill)+1)
            multiplier = np.array(multiplier)
            ds["ndvi"][pixel,idx_to_gapfill] = slopes[i] * multiplier + float(ds["ndvi"][pixel,valid_idx[0][i]])
