# This script is jsut to ensure that the data is collected as the step 4 and the mask_array is present in all the dates
# nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo.log 2>&1 &


import numpy as np
import xarray as xr
from dask.distributed import Client
import statsmodels.api as sm
import os
import shutil
import requests
import pandas as pd
import pystac_client
import argparse
import warnings
# import zarr
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr"
SOURCE_ZARR_MERGED = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/04_merged_ndvi/"  # This is the OUT_ZARR from script 4

OUTPUT_BASE        = "/mnt/data1/UniBe-swiss-ndvi/tmp_ndvi_05_processed.zarr"
#TODO: try to remove this: OUTPUT_BASE_tmp = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/05_processed/tmp.zarr" # here the computation will be written and then stacked to the OUTPUT_BSAE folder
DASK_TEMP_DIR = "/mnt/data1/UniBe-swiss-ndvi/tmp_data/"
os.makedirs(DASK_TEMP_DIR, exist_ok=True)


# TODO: this script 5 has been written with script 4 merging a copy of the historic NDVI together with newly downloaded data.
#       This is not needed and we should be able to use the freshly downloaded data directly: TODO use below SOURCE_ZARR
# SOURCE_ZARR = "/mnt/data1/UniBe-swiss-ndvi/data/demo_all_pixel/02-03_ndvi_dataset_temporal.zarr" # the zarr from script 3
# new_ds.to_zarr("/mnt/data1/UniBe-swiss-ndvi/TODO_FB.zarr", mode="w", consolidated=True)
    # TODO: or alternatively use this TODO_FB.zarr

# return the dates to analyse

def get_swisstopo_sentinel_dates(start='2025-12-01', end='2026-02-16'): # TODO: we have a similar (but different) code in 3_add_dates. TODO: consolidate.

    # Connect to Swisstopo STAC API
    service = pystac_client.Client.open('https://data.geo.admin.ch/api/stac/v0.9/')
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    bbox_swiss_4326 = [5.70, 45.8, 10.6, 47.95]

    item_search = service.search(
        bbox=bbox_swiss_4326,
        datetime=f'{start}/{end}',
        collections=['ch.swisstopo.swisseo_s2-sr_v100']
    )
    s2_files = list(item_search.items())

    dates = []
    for item in s2_files:
        assets = item.assets
        asset_key_metadata = next((key for key in assets.keys() if key.endswith('metadata.json')), None)
        metadata_asset = assets[asset_key_metadata]
        json_link_metadata = metadata_asset.href
        response = requests.get(json_link_metadata)
        metadata_json = response.json()
        dates.append(metadata_json['BANDS-10M']['SOURCE_COLLECTION_PROPERTIES']['date'])

    # Convert to datetime64[D] array
    pd_dates = pd.to_datetime(dates)
    dates_array = pd_dates.values.astype('datetime64[D]')

    return dates_array

def smoothing_and_gapfilling(ndvi_arr, median_ndvi_arr, last_array_dates_idx,
                             last_delta, current_delta, deltas_arr,current_date_idx, pot_outlier_present):

    if pot_outlier_present:

        pot_date_idx = int(last_array_dates_idx[7])

        pot_ndvi = ndvi_arr[pot_date_idx] / 10000.0
        pot_median_ndvi = median_ndvi_arr[pot_date_idx] / 10000.0
        pot_delta = pot_ndvi - pot_median_ndvi

        # L1
        idx_to_interpolate = np.arange(int(last_array_dates_idx[6]),current_date_idx +1)

        deltas_L1 = np.array([last_delta, pot_delta, current_delta])
        deltas_L1_idx = np.array([int(last_array_dates_idx[6]), int(last_array_dates_idx[7]), current_date_idx +1])

        deltas_interpolated = np.interp(idx_to_interpolate, deltas_L1_idx, deltas_L1)
        L1_ndvi = deltas_interpolated + median_ndvi_arr[int(last_array_dates_idx[6]) :current_date_idx +1] / 10000.0

        ndvi_arr[int(last_array_dates_idx[6]) :current_date_idx +1] = L1_ndvi

        # L2 smoothing

        deltas_L2 = np.concatenate([deltas_arr, np.array([current_delta])])

        smoothed_deltas = sm.nonparametric.lowess(deltas_L2, np.arange(1, len(deltas_L2) + 1),
                                                  frac=1, it=3, return_sorted=False)
        idx_to_interpolate = np.arange(int(last_array_dates_idx[2]), int(last_array_dates_idx[4]) + 1)
        deltas_L2_idx = last_array_dates_idx[2:5]
        deltas_interpolated = np.interp(idx_to_interpolate,
                                       deltas_L2_idx, smoothed_deltas[2:5])
        
        L2_ndvi = deltas_interpolated + median_ndvi_arr[int(last_array_dates_idx[2]) : int(last_array_dates_idx[4]) + 1] / 10000.0

        ndvi_arr[int(last_array_dates_idx[2]) : int(last_array_dates_idx[4]) + 1] = L2_ndvi

    else:

        # L1 linear
        idx_to_interpolate = np.arange(int(last_array_dates_idx[6]), current_date_idx +1)
        deltas_interpolated = np.linspace(last_delta, current_delta, num=len(idx_to_interpolate))
        L1_ndvi = deltas_interpolated + median_ndvi_arr[int(last_array_dates_idx[6]) :current_date_idx +1] / 10000.0
        ndvi_arr[int(last_array_dates_idx[6]) :current_date_idx +1] = L1_ndvi

        # L2 smoothing
        smoothed_deltas = sm.nonparametric.lowess(deltas_arr, np.arange(1, len(deltas_arr) + 1),
                                                  frac=1, it=3, return_sorted=False)

        idx_to_interpolate = np.arange(int(last_array_dates_idx[2]), int(last_array_dates_idx[3]) + 1)
        deltas_L2_idx = last_array_dates_idx[2:4]
        deltas_interpolated = np.interp(idx_to_interpolate,
                                       deltas_L2_idx, smoothed_deltas[2:4])

        L2_ndvi = deltas_interpolated + median_ndvi_arr[int(last_array_dates_idx[2]) : int(last_array_dates_idx[3]) + 1] / 10000.0

        ndvi_arr[int(last_array_dates_idx[2]) : int(last_array_dates_idx[3]) + 1] = L2_ndvi

    return ndvi_arr


def continous_analysis(ndvi_arr_2, median_arr,bool_dates,first_date, full_dates, current_date):
        
    # placeholder for dates to generate the tiff
    mask_ndvi_arr_2  = np.empty(len(bool_dates), dtype=np.int8)
    mask_ndvi_arr_2.fill(0)

    current_date_idx = int((current_date - first_date) / np.timedelta64(1, "D"))

    # spinup here
    mask_ndvi_arr = mask_ndvi_arr_2[:int(current_date_idx)]
    ndvi_subset = ndvi_arr_2[:int(current_date_idx)]
    ndvi_subset_mask = ndvi_arr_2[:int(current_date_idx)]
    median_subset = median_arr[:int(current_date_idx)]
    date_subset = full_dates[:int(current_date_idx)]
    bool_subset = bool_dates[:int(current_date_idx)]
    bool_arr_mask = bool_dates[:int(current_date_idx)]

    ndvi_subset = ndvi_subset[bool_subset]
    date_subset = date_subset[bool_subset]
    median_subset = median_subset[bool_subset]

    obs_mask = (ndvi_subset_mask > 0) & (ndvi_subset_mask < 10000.0) & bool_arr_mask # this wil filter all the observation

    # valid mask
    valid_mask = (ndvi_subset > 0) & (ndvi_subset < 10000.0)

    #outlier detection
    
    if np.sum(valid_mask) > 6:

        # check if the obs. are outlier, pot. out. or true obs.
        last_ndvi_arr = ndvi_subset[valid_mask] / 10000.0
        last_median_arr = median_subset[valid_mask] / 10000.0

        delta = last_ndvi_arr - last_median_arr

        # check last position alone beacuse does not have 2 neighbour

        delta_last = delta[-1]
        delta_delta_last = delta[-1] - delta[-2]

        pot = False

        if (abs(delta_last) > 0.1 and abs(delta_delta_last)> 0.1):
            pot = True

        delta_delta_left = delta[2:]
        delta_delta_rigth = delta[:-2]
        outlier_mask = ((abs(delta[1:-1]) > 0.1) & (abs(delta_delta_left) > 0.1) & (abs(delta_delta_rigth) > 0.1))

        # last 7 valid dates
        last_dates = date_subset[valid_mask][1:-1]

        if pot == False:

            last_valid_dates = last_dates[~outlier_mask][-6:]
            

            # append last dates
            last_valid_dates = np.append(last_valid_dates,date_subset[valid_mask][-1])
            last_valid_dates = last_valid_dates[-7:]
            # always output 8 slots
            last_dates_array = np.full(8, np.datetime64("1900-01-01", "D"), dtype="datetime64[D]")
            last_dates_array[:len(last_valid_dates)] = last_valid_dates

            obs_mask = (ndvi_subset_mask > 0) & (ndvi_subset_mask < 10000.0) & bool_arr_mask # this wil filter all the observation
            filter_obs_to_smooth = obs_mask
            filter_obs_smooted = obs_mask

            idx = int((last_dates_array[3] - first_date) / np.timedelta64(1, "D")) + 1

            filter_obs_to_smooth[idx:] = False
            filter_obs_smooted[:idx] = False

            obs_mask = (ndvi_subset_mask > 0) & (ndvi_subset_mask < 10000.0) & bool_arr_mask

            mask_ndvi_arr = np.zeros(len(bool_dates[:current_date_idx]), dtype=np.int8) 

            idx = int((last_dates_array[3] - first_date) / np.timedelta64(1, "D")) + 1

            idx = max(0, min(int(idx), len(mask_ndvi_arr)))

            mask_ndvi_arr[obs_mask] = 2

            before = np.arange(len(mask_ndvi_arr)) < idx

            mask_ndvi_arr[ before & obs_mask ] = 3
            mask_ndvi_arr[ before & (~obs_mask) ] = 1
            mask_ndvi_arr_2[:len(mask_ndvi_arr)] = mask_ndvi_arr


        else:

            last_valid_dates = last_dates[~outlier_mask][-7:]
            # always output 8 slots
            last_dates_array = np.full(8, np.datetime64("1900-01-01", "D"), dtype="datetime64[D]")
            last_dates_array[:len(last_valid_dates)] = last_valid_dates
            last_dates_array[-1] = date_subset[valid_mask][-1]

            filter_obs_to_smooth = obs_mask
            filter_obs_smooted = obs_mask

            idx = int((last_dates_array[3] - first_date) / np.timedelta64(1, "D")) + 1
            filter_obs_to_smooth[idx:] = False
            filter_obs_smooted[:idx] = False

            obs_mask = (ndvi_subset_mask > 0) & (ndvi_subset_mask < 10000.0) & bool_arr_mask
            mask_ndvi_arr = np.zeros(len(bool_dates[:current_date_idx]), dtype=np.int8)  # length = window length you're working with

            idx = int((last_dates_array[3] - first_date) / np.timedelta64(1, "D")) + 1
            idx = max(0, min(int(idx), len(mask_ndvi_arr)))
            mask_ndvi_arr[obs_mask] = 2

            before = np.arange(len(mask_ndvi_arr)) < idx
            mask_ndvi_arr[ before & obs_mask ] = 3
            mask_ndvi_arr[ before & (~obs_mask) ] = 1
            mask_ndvi_arr_2[:len(mask_ndvi_arr)] = mask_ndvi_arr

        # finished last dates array generation

        last_dates_array = last_dates_array.astype("datetime64[D]")

        last_array_dates_idx =  ((last_dates_array - first_date) / np.timedelta64(1, "D")).astype(int)

    else:
        # no enough date
        return ndvi_arr_2, mask_ndvi_arr_2

    

    # if not enough valid dates (any of first 7 < 0) -> skip
    if np.any(last_array_dates_idx[:7] < 0):
        
        last_dates_array = last_dates_array.astype("datetime64[D]")

        return ndvi_arr_2, mask_ndvi_arr_2

    # compute values
    current_ndvi = ndvi_arr_2[current_date_idx] / 10000.0
    median_current = median_arr[current_date_idx] / 10000.0

    last_idx = int(last_array_dates_idx[6])
    last_ndvi = ndvi_arr_2[last_idx] / 10000.0
    last_median = median_arr[last_idx] / 10000.0

    last_delta = last_ndvi - last_median
    current_delta = current_ndvi - median_current
    delta_delta = current_delta - last_delta

    if (current_ndvi > 0) and (current_ndvi < 1):

        if (abs(delta_delta) > 0.1) and (abs(current_delta) > 0.1):

            last_dates_array[7] = current_date  
            last_dates_array = last_dates_array.astype("datetime64[D]")

            return ndvi_arr_2, mask_ndvi_arr_2

        deltas_arr = (ndvi_arr_2[last_array_dates_idx[:7].astype(int)] - median_arr[last_array_dates_idx[:7].astype(int)]) / 10000.0

        if last_array_dates_idx[7] > 0:

            pot_idx = int(last_array_dates_idx[7])
            pot_delta = (ndvi_arr_2[pot_idx] - median_arr[pot_idx]) / 10000.0

            if abs(pot_delta) < 0.1:

                pot_deltas_arr = (ndvi_arr_2[last_array_dates_idx.astype(int)] - median_arr[last_array_dates_idx.astype(int)]) / 10000.0

                ndvi_arr_2 = smoothing_and_gapfilling(ndvi_arr_2, median_arr, last_array_dates_idx, last_delta, current_delta, pot_deltas_arr,current_date_idx, pot_outlier_present=True)

                return ndvi_arr_2, mask_ndvi_arr_2
            
            else:

                ndvi_arr_2 = smoothing_and_gapfilling(ndvi_arr_2, median_arr, last_array_dates_idx, last_delta, current_delta, deltas_arr,current_date_idx, pot_outlier_present=False)

                return ndvi_arr_2, mask_ndvi_arr_2
        else:

            ndvi_arr_2 = smoothing_and_gapfilling(ndvi_arr_2, median_arr, last_array_dates_idx, last_delta, current_delta, deltas_arr,current_date_idx, pot_outlier_present=False)

            return ndvi_arr_2, mask_ndvi_arr_2
    else:

        # no observation -> estimate
        tau = 2#len(ndvi_subset) - last_idx
        estimated_delta = last_delta * np.exp(- tau / 45.0)
        ndvi_arr_2[current_date_idx] = estimated_delta + median_current
        last_dates_array = last_dates_array.astype("datetime64[D]")

        return ndvi_arr_2, mask_ndvi_arr_2


def continuous_ndvi(ndvi_arr, median_arr,bool_dates,*, full_dates, dates_array):

    # coerce shapes: ensure 1D
    ndvi_arr_2 = np.asarray(ndvi_arr).copy().ravel()
    median_arr = np.asarray(median_arr).ravel()
    full_dates = np.asarray(full_dates).astype("datetime64[D]").ravel()
    bool_dates = np.asarray(bool_dates).ravel()
    first_date = full_dates[0].astype("datetime64[D]")
    mask_ndvi_arr = np.zeros_like(ndvi_arr_2, dtype=np.int8)

    for current_date in np.sort(dates_array):

        ndvi_arr_2,mask_for_date = continous_analysis(ndvi_arr_2, median_arr,bool_dates,first_date, full_dates, current_date)

        current_date_idx = int((current_date - first_date) / np.timedelta64(1, "D"))

        # in this way it updates only the current value
        mask_ndvi_arr[current_date_idx] = mask_for_date[current_date_idx]

    return ndvi_arr_2, mask_ndvi_arr

# -----------------------------
# 1) Setup Dask client
# -----------------------------

if __name__ == '__main__':

    #N_WORKERS = 150 # TODO reactivate
    # N_WORKERS = 50 # NOTE: going above 50 seems to lead to communication issues
    N_WORKERS = 100
    MEMORY_PER_WORKER = "36GB"

    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=MEMORY_PER_WORKER,
        dashboard_address=":8345",
        local_directory= DASK_TEMP_DIR
    )
    print(client.dashboard_link)

    # -----------------------------
    # 2) Open Zarr dataset
    # -----------------------------
    #if os.path.exists(OUTPUT_ZARR):
    #    shutil.rmtree(OUTPUT_ZARR)
    
    SOURCE_ZARR_MERGED_LIST = [entry.path for entry in os.scandir(SOURCE_ZARR_MERGED) if entry.is_dir()]
    ds = xr.open_mfdataset(SOURCE_ZARR_MERGED_LIST)
    #ds = ds.isel(pixel = slice(0, 10**6)) # TODO: generate here a small example on the fly. 
    #ds = ds.isel(pixel=slice(0,3000)) # TODO: generate here a small example on the fly
    #ds = ds.isel(pixel=slice(0,100)) # TODO: generate here a small example on the fly

    #ds_to_use_in_future = xr.open_zarr(SOURCE_ZARR) # TODO: directly load this instead of SOURCE_ZARR_MERGED
    #ds_to_use_in_future = zarr.open_group(SOURCE_ZARR, mode="r")
    # ds_hist = xr.open_dataset("/mnt/data1/UniBe-swiss-ndvi/data/ndvi_historic_v3_compr.zarr").isel(pixel = slice(0, 10**6)) # TODO: remove again
    #  # TODO: directly load SOURCE_ZARR and ds_hist instead of SOURCE_ZARR_MERGED (avoiding script 4 entirely)
    
    # append median NDVI to the observed NDVI
    lookuptable  = xr.open_zarr(INPUT_LOOKUPTABLE, consolidated=False)
    doy = ds["date"].dt.dayofyear
    doy_noLeap = xr.where(doy == 366, 365, doy) # remove leap year if encountered
    median_ndvi = lookuptable["median_ndvi"].sel(doy=doy_noLeap)
    ds["median_ndvi"] = median_ndvi # add as variable to ds
    # NOTE: now we have in ds: 'obs_date', 'ndvi' and 'median_ndvi'
    #       TODO: will we now add mask_array (from 0 to 4) ???

    # Apply continuous ndvi function:

    # Prepare arguments
    ndvi_array   = ds["ndvi"].chunk({"date": -1})
    median_array = ds["median_ndvi"].chunk({"date": -1})
    bool_dates   = ds["obs_date"].chunk({"date": -1})
    dates        = ds["date"] 

    # For dates_array, use full range:
    dates_array = get_swisstopo_sentinel_dates(start="2017-01-01", end="2025-12-31")    # TODO: what happens with this in the future? Will this break? Will this slow down?
    # dates_array_orig = dates_array
    # append the last dates if not present
    if dates_array[-1].astype('datetime64[D]') != np.datetime64('2025-12-31', 'D'):     # TODO: what happens with this in the future? Will this break? Will this slow down?
        dates_array = np.append(dates_array, np.datetime64('2025-12-31', 'D'))
    dates_array = np.sort(np.unique(dates_array))
    #np.save("/mnt/data1/UniBe-swiss-ndvi/data/temp_dates_array.npy", dates_array)
    #dates_array = np.load("/mnt/data1/UniBe-swiss-ndvi/data/temp_dates_array.npy")


    # Run continuous_ndvi
    ndvi_arr, mask_ndvi_arr = xr.apply_ufunc(
        continuous_ndvi,
        ndvi_array,
        median_array,
        bool_dates,
        input_core_dims=[["date"],["date"], ["date"]],
        output_core_dims=[["date"],["date"]],
        vectorize=True,
        dask="parallelized",
        #dask = "forbidden", # TODO: this is for development
        kwargs={
            "full_dates" : dates,
            "dates_array": dates_array
        },
        output_dtypes=[ndvi_array.dtype,ndvi_array.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    
    out_ds = xr.Dataset(
        {
            "ndvi_processed": ndvi_arr,
            "mask_array": mask_ndvi_arr
        },
        coords={
            "date": ds["date"],
            "pixel": ds["pixel"],
            "x": ds["x"],
            "y": ds["y"]
        }
    )

    # Chunk explicitly to avoid Dask graph explosion
    out_ds = out_ds.chunk({"pixel": 100000, "date": 365})

    # Remove leftover compressor info if copying from another dataset
    for var in out_ds.variables: # TODO: here should we go over .data_vars only or over .variables???
        out_ds[var].encoding.pop("compressor", None)
        out_ds[var].encoding.pop("chunks", None)

    # Write to Zarr
    out_ds.to_zarr(OUTPUT_BASE, mode="w", consolidated=True, compute=True) # TODO: ideally here we would just append

    print("Done")

    client.close()
