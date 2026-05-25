import datetime as dt
import numpy as np
import statsmodels.api as sm
from dask.distributed import Client
from dask import visualize
import dask.array as da
import xarray as xr
import argparse
import os, shutil, sys
import time
from numcodecs import blosc, Blosc, zarr3
from zarr.codecs import BloscCodec

INPUT_LOOKUPTABLE  = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr" # TODO: move to data2

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Numcodecs codecs are not in the Zarr version 3 specification",
    module="numcodecs.zarr3"
)

NO_COVERAGE = 32767
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step
INVALID     = -32768
INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows

# HOW TO RUN FROM BASH:
# source /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/.venv/bin/activate
# SCRIPT_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient.py"
# LOG_FILE="/home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/5_analyse_demo_efficient_FB_$(date "+%Y-%m-%d_%Hh%Mm%S").log"
# NEW_NDVI="/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_10kmX10km_4th.zarr"
# HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr"
# HISTO_OUTPUT="/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km_extended2.zarr"
# python -u $SCRIPT_FILE $NEW_NDVI $HISTO_INPUT --histo-output=$HISTO_OUTPUT > $LOG_FILE  2>&1 &

# NOTE: for development: ndvi_arr_original, median_arr_original, mask_array_original, dates_arr_original = ndvi_array, median_array, mask_array, dates_array
def continuous_ndvi(ndvi_arr_original, median_arr_original, mask_array_original, dates_arr_original):
        ### Illustration of indexing:  - (day without observation), x (observation), o (outlier observation), I (invalid observation, e.g. -0.7)
        ###                            x̂ (smoothed observation), . unspecified (used for different things)
        ###                            0: L0 value (obs. or gapfilled) from historical processing (or last CI processing)
        ###                            1: L1 value (obs. or gapfilled) from historical processing (or last CI processing)
        ###                            2: L2 value (obs. or gapfilled) from historical processing (or last CI processing)
        ###
        ### [...]2222222222222222111111110-x--xI-o--x-x--     len() = 3000  (ndvi_arr_original, median_arr_original, mask_array_original, is_observation_date, dates_arr_original)
        ### [...]. ..  ..  .  .                               len() = 800   (obs_L2) Indices of observation dates
        ###                   .                               len() = 1     (obs_L2[-1]) Index last L2 obs available
        ###            .                                      len() = 1     (crop_start) Index (-4)
        ###                        . .                        len() = 2     (obs_L1) Indices of observation dates
        ### [...]222222                                       len() = 2977  (ndvi_not_processed, mask_not_processed)
        ###            ..................................     len() = 34    (dates_crop)
        ###            01234567dddddddddddddddddddddddddd     len() = 34    (days_diff)
        ###            2222222222111111110-x--xI-o--x-x--     len() = 34    (ndvi_crop, median_crop, mask_crop)
        ###            x̂x̂--x̂--x̂----o-x---x-x--x--o--x-x--     len() = 34    (ndvi_crop)
        ###            TTffTffTffffTfTfffTfTffTffTffTfTff     len() = 34    (valid_obs_mask) Invalid gets dropped because outside 0 and 1.
        ###            x̂x̂  x̂  x̂    o x   x x  xI o  x x       len() = 13    ()
        ###            x̂x̂  x̂  x̂    o x   x x  x  o  x x       len() = 12    (delta_ndvi_1, ndvi_1, median_1, days_diff_1, obs_original_idx)
        ###            01  4  7    d d   d d  d  d  d d       len() = 12    (days_diff_1)
        ###             x̂  x̂  x̂    o x   x x  x  o  x         len() = 10    (outlier_mask....  and delta_delta_left, ...)
        ###             x̂  x̂  x̂      x   x x  x     x         len() = 8     (delta_ndvi_2, days_diff_2) 
        ###
        ### OBJECTIVE: kkkkkkkkuuuuuuuuuuu111111111111100     meaning: k=keep unchanged, 
        ###                                                            u=update value to L2 level, 
        ###                                                            1=update value to L1, 
        ###                                                            0=update value to L0
        ###
        ### NOTE: apply linear interpolation window (and within that a 7-observation window for smoothing):
        ### Define newly smoothed L2 values:
        ###                          x̂   x̂                    len() = 5     (delta_ndvi_to_interpolate_inner)
        ### With these 7-obs windows (looping one after the other):
        ###            [x̂  x̂  x̂      x   x x  x]              len() = 7     (for i = 0: delta_ndvi_2[i:i+7])
        ###               [x̂  x̂      x̂   x x  x     x]        len() = 7     (for i = 1: delta_ndvi_2[i:i+7]) # NOTE that this uses the smoothed value from the run from i=0 (different from the historic application)
        ### concatenate these components:
        ###             x̂  x̂  x̂      x   x x  x     x         len() = 8     (delta_ndvi_2, days_diff_2, ndvi_valid_2, nonOutlier_idx_2) 
        ###            x̂                                      len() = 1     left-most obs (days_diff_1[0:1], delta_ndvi_1[0:1])
        ###                                           x       len() = 1     right-most obs (days_diff_1[-1:], delta_ndvi_1[-1:])
        ###
        ###             x̂  x̂  x̂                               len() = 3     (delta_ndvi_2[:3]) # 3x latest, previous L2 values
        ###                          x̂   x̂                    len() = 5     (delta_ndvi_to_interpolate_inner)
        ###                                x  x     x         len() = 3     (delta_ndvi_2[-3:], # L1 outlier-filtered
        ### to:
        ###            x̂x̂  x̂  x̂      x̂   x̂ x  x     x x       len() = 9     (dates_to_interpolate) = [days_diff_1[0:1], days_diff_2, days_diff_1[-1:]]
        ###            01  4  7      d   d d  d     d d d     len() = 9     (dates_to_interpolate)    # last value of today is only appended if today there is no observation
        ###                                                                 last value (=zero-NDVI-delta) is only appended if today there is no observation
        ###            x̂x̂  x̂  x̂      x̂   x̂ x  x     x x       len() = 9     (delta_ndvi_to_interpolate) = [delta_ndvi_1[0:1], delta_ndvi_2[:3],delta_ndvi_to_interpolate_inner,delta_ndvi_2[-3:],delta_ndvi_1[-1:]]
        ###            x̂x̂  x̂  x̂      x̂   x̂ x  x     x x 0     len() = 10    (delta_ndvi_to_interpolate)   # 0 for today is only appended if today there is no observation
        ###                                                                 last value (=zero-NDVI-delta) is only appended if today there is no observation
        ### and use them to do linear interpolation, to these targets:
        ###            01234567dddddddddddddddddddddddddd     len() = 34    (days_diff)
        ###            x̂x̂..x̂..x̂..........................     len() = 34    (interpolated_values, ndvi_processed, mask_crop)
        ### Then do final concatenation for update:
        ### [...]......                                       len() = 2977  (ndvi_not_processed,mask_not_processed)(historical cropping of L2 not changed anymore)
        ### [...]......x̂x̂..x̂..x̂..........................     len() = 3000  (final_ndvi_value, mask_array_final)(final merging and returning the timeseries)
        ###
        ### For the masking we use following helper variables:
        ###                          x                        len() = 1     (nonOutlier_idx_2[-4])
        ###      TTTTTTTTTTTTTTTTTTTTffffffffffffffffffff     len() = 40    ('before' defined as '< nonOutlier_idx_2[-4]' )
        ###                       o              o            len() = 2     (outlier_idx_2 encoding the obs_original_idx)
        ###
        ### NOTE: what about these two?:               --     # REPLY: they are linearly interpolated towards 0. Giving us L0 estimation.
        ### NOTE: what about?: ......x                        # REPLY: these are just linearly interpolated between already smoothed and not yet smoothed values. 
        ###                                                            These are now called L2, since they will not change anymore.
        ### NOTE: what about?:       x̂   x̂ x  x     x         # REPLY: these are 2 smoothed and 3 non-smoothed values used for linear interpolation.
        ### NOTE: There is no backpropagation of outliers. 
        ###       E.g. even when we have a new observation:
        ###                                         x-x--x:   let's call these [t-1], [t0], and [t+1]
        ###       Prior to having [t+1]-observation: [t0] could not be determined outlier.
        ###       With this new   [t+1]-observation: [t0] might now be determined outlier, if all three conditions (left, right, median) are met simultaneously.
        ###       Even in that case:                 [t-1]'s outlier status will not change. Since it only depends on [t-2,t0,median-1]. Thus t-1 is fixed and there is no backpropagation.

        obs_L2 =  np.nonzero(mask_array_original == 3)[0] # NOTE: this also drops outliers (mask_array==4).
        # obs_L1 =  np.nonzero(mask_array_original == 2)[0] # (unused)

        # Ensure mask_array is writable
        mask_array_original = np.array(mask_array_original, copy=True)

        if len(obs_L2) < 3:
            return ndvi_arr_original, mask_array_original

        # A) Crop (subset) whole time series to last part to be processed,
        # defined so that it contains enough L2 to do the smoothing => variable suffix '_crop':
        crop_start = obs_L2[-4]  # Index (-4), so that even when 
                                 # dropping left-most (and right-most), we end up 
                                 # with 3 L2 values (i.e. the left half of 7 obs window) to smooth
        ndvi_crop = ndvi_arr_original[crop_start:]
        median_crop = median_arr_original[crop_start:]
        dates_crop = dates_arr_original[crop_start:]
        mask_crop = mask_array_original[crop_start:] 

        ndvi_not_processed = ndvi_arr_original[:crop_start] 
        mask_not_processed = mask_array_original[:crop_start] 

        days_diff = (dates_crop- dates_crop[0])  / np.timedelta64(1, 'D')
        ndvi_crop = ndvi_crop / 10000
        median_crop = median_crop  / 10000
        

        # B) Filter for validity and if it is an observation (as opposed to an interpolated value) and ensure it was not identified as outlier previously => variable suffix '_1':
        valid_obs_mask = (ndvi_crop > 0) & (ndvi_crop < 1) & ((mask_crop == 2) | (mask_crop == 3)) # Note that this keeps L1 and L2 observations, and drops L2 outliers (mask_crop == 4)

        ndvi_1      = ndvi_crop[valid_obs_mask]
        median_1    = median_crop[valid_obs_mask]
        days_diff_1 = days_diff[valid_obs_mask]
        obs_original_idx = np.arange(len(ndvi_crop))[valid_obs_mask] # used to keep track of delta ndvi position and the outlier position
        
        
        # C) Outlier detection
        delta_threshold = 0.05
        delta_delta_threshold = 0.1

        delta_ndvi_1 = ndvi_1 - median_1
        delta_delta_left = delta_ndvi_1[:-2] - delta_ndvi_1[1:-1]
        delta_delta_right = delta_ndvi_1[2:] - delta_ndvi_1[1:-1]
        outlier_mask = ((abs(delta_ndvi_1[1:-1])  > delta_threshold) & 
                        (abs(delta_delta_left)  > delta_delta_threshold) & 
                        (abs(delta_delta_right) > delta_delta_threshold))
        
        # outlier removal (and left-, right-most obs removal) => variable suffix '_2'
        ndvi_valid_2 = ndvi_1[1:-1][~outlier_mask]
        delta_ndvi_2 = delta_ndvi_1[1:-1][~outlier_mask]
        days_diff_2  = days_diff_1[1:-1][~outlier_mask]
        
        outlier_idx_2 = obs_original_idx[1:-1][outlier_mask]
        nonOutlier_idx_2 = obs_original_idx[1:-1][~outlier_mask]
        

        # some sites do not have any observation or very few
        if len(delta_ndvi_2) > 6:
        
            # D) LOESS smoothing of new L2 values: x => x̂
            # loop over the 7 rolling deltas. If the deltas are too negative for 5 or more observations (e.g. extreme events as fire) 
            # or if the original values too close to the boundaries (1.0 and 0.0) we keep the non-smoothed value to do the linear interpolation
            # otherwise we smooth the value (wit LOESS) to then do the linear interpolation

            # D1: Fill up the inner part of the vector for linear interpolation (this can be only 1 observation or it can be multiple observations 
            #     if the code is run less frequently)
            delta_ndvi_to_interpolate_inner = np.full(len(delta_ndvi_2)-6, np.nan)

            for i in np.arange(len(delta_ndvi_to_interpolate_inner)):
                # ndvi_valid_to_check: not needed for consistency with historical processing
                delta_window_to_smooth = delta_ndvi_2[i:i+7] # window to smooth, the center value will be smoothed
                ndvi_valid_to_check    = ndvi_valid_2[i:i+7] # this will be used to check if the absolute value is close to the boundaries condition

                if (np.any((ndvi_valid_to_check < 0.05) | (ndvi_valid_to_check > 0.95))or (np.sum(delta_window_to_smooth < -0.2) >= 5)): 
                    # exceptional case:
                    # here, check for the NDVI close to the boundaries or extreme negative NDVI values (fire but not drought)                       
                    # in case this conditions are met, skip the smoothing and keep the non-smoothed delta
                    delta_ndvi_to_interpolate_inner[i] = delta_window_to_smooth[3]

                else:
                    # normal case:
                    # smooth the 7 rolling window
                    loess = sm.nonparametric.lowess(delta_window_to_smooth, np.arange(0,7), frac= 1, it=3, return_sorted=False)
                    delta_ndvi_to_interpolate_inner[i] = loess[3] # store x̂ in `delta_ndvi_to_interpolate_inner` for linear interpolation below
                    delta_ndvi_2[i+3] = loess[3]                  # also use loess-smoothed value x̂ for next iteration


            # E) Linear interpolation of L2, L1 and L0 values:
            # Combine inner (new L2 values) with previous L2 values, and new L1 values for linear interpolation.

            delta_ndvi_to_interpolate = np.concatenate([
                delta_ndvi_1[0:1],               # (left-most observation)     # making sure this does not change values
                delta_ndvi_2[:3],                # last three L2 available     # making sure these do not change values
                delta_ndvi_to_interpolate_inner, # new L2, newly smoothed
                delta_ndvi_2[-3:],               # L1 observations outlier-filtered
                delta_ndvi_1[-1:]                # last observation (right-most observation, not outlier-filtered because there is no right-hand neighbor (yet))
            ]) 
            dates_to_interpolate = np.concatenate([
                days_diff_1[0:1],
                days_diff_2,     # NOTE: By including ALL delta_ndvi_2 except the new smoothed value, len delta_ndvi_2 == len(days_diff_2)
                days_diff_1[-1:] # date of the last observation without 2 right-hand neighbor
            ])

            # Special case for L0 extra- or interpolation:
            #   if the current day is an observation above is sufficient.
            #   if the current day is not an observation, perform L0 linear decay:
            if (abs(days_diff_1[-1] - days_diff[-1])>0.1): # equivalent to if days_diff_1[-1] != days_diff[-1]
                delta_ndvi_to_interpolate = np.concatenate([
                    delta_ndvi_to_interpolate,
                    np.array([0]) # L0 linear decay
                ])
                dates_to_interpolate = np.concatenate([
                    dates_to_interpolate,
                    np.array([days_diff[-1]]) # today
                ])

            # using linear interpolation (ensures `delta_ndvi_to_interpolate` at previously existing L2 positions remain unchanged)
            interpolated_values = np.interp(
                days_diff,                    # evaluate here f()
                dates_to_interpolate,         # observed x
                delta_ndvi_to_interpolate     # observed f(x)
            )

            # prepare return value for NDVI:
            ndvi_processed = 10000 * (interpolated_values + median_crop)

            # Store processing status (L0, L1, L2) encoded as following 5 values:
            # indexing of array mask
                # mask_array == 0: the date is not an observation and is yet to be smoothed (L1 or L0)
                # mask_array == 1: the date is not an observation and is already smoothed (L2)
                # mask_array == 2: the date is     an observation and is yet to be smoothed (L1)
                # mask_array == 3: the date is     an observation and is already smoothed (L2)
                # mask_array == 4: the date is     an observation and is an outlier

            # Code starts out with default values of 0 or 2 (defined when new_ds['mask_array'] was appended)

            # Define all observation dates:
            mask_crop[valid_obs_mask] = 2 # NOTE: this overwrites the mask value of already processed L2 obs (mask_array == 3). (But not of L2-outliers (mask_array == 4), since they were correctly not considered for valid_obs_mask).
            # TODO: BUG all of this processing status should not overwrite previously processed L2 values. Therefore start at earliest at 

            # Mark the L2-finalized values (smoothed), i.e. all those to the left of the last center point
            before = np.arange(len(mask_crop)) < nonOutlier_idx_2[-4] # marks all L2-finalized

            mask_crop[ before & valid_obs_mask ] = 3
            mask_crop[ before & (~valid_obs_mask) ] = 1

            # Mark the outlier dates (4): 
            mask_crop[outlier_idx_2] = 4


            # Concatenate return values:
            mask_array_final =  np.concatenate([mask_not_processed, mask_crop])
            final_ndvi_value =  np.concatenate([ndvi_not_processed, ndvi_processed])

            return final_ndvi_value, mask_array_final
        
        else:

            return ndvi_arr_original, mask_array_original

if __name__ == "__main__":

    # PARSE ARGUMENTS:
    parser = argparse.ArgumentParser()

    parser.add_argument("INPUT_ZARR",        help="Full path to Zarr folder with newly downloaded NDVI data")
    parser.add_argument("HISTO_ZARR_INPUT",  help="Full path to Zarr folder with historic NDVI data")
    parser.add_argument("--histo-output", dest = "HISTO_ZARR_OUTPUT", default=None,
                        help="Full path for updated historic Zarr (if omitted, defaults to HISTO_ZARR_INPUT)"+
                             "Path must either be a non-existing folder or then HISTO_ZARR_INPUT. In latter case data is appended.")
    args = parser.parse_args()

    INPUT_ZARR        = args.INPUT_ZARR
    HISTO_ZARR_INPUT  = args.HISTO_ZARR_INPUT
    HISTO_ZARR_OUTPUT = args.HISTO_ZARR_OUTPUT or HISTO_ZARR_INPUT # if None defaults to HISTO_ZARR_INPUT

    # if running interactively use e.g.:
    #   # HISTO_ZARR_INPUT  = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr"
    #   # HISTO_ZARR_OUTPUT = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr"
    #   # INPUT_ZARR        = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12_processed.zarr"
    
    #   # HISTO_ZARR_INPUT  = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_10kmX10km.zarr"
    #   # HISTO_ZARR_OUTPUT = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v5_chk_40000_365_10kmX10km.zarr"
    #   # INPUT_ZARR        = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-18_17h39_ndvi_01_downloaded_2025-11-30_2025-12-12_processed.zarr"
    #   # INPUT_ZARR        = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-03-23_12h50_ndvi_01_downloaded_2025-11-30_2026-03-22_processed.zarr/"

    #   # HISTO_ZARR_INPUT     = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_1000mX1000m.zarr"
    #   # HISTO_ZARR_OUTPUT    = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_1000mX1000m_extended.zarr" # TODO: remove this and instea do it circular
    #   # INPUT_ZARR           = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_1000mX1000m_4th.zarr"
    #   # HISTO_ZARR_INPUT     = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km.zarr"
    #   # HISTO_ZARR_OUTPUT    = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_10kmX10km_extended.zarr" # TODO: remove this and instea do it circular
    #   # INPUT_ZARR           = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_10kmX10km_4th.zarr"
    #   HISTO_ZARR_INPUT     = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_100kmX100km.zarr"
    #   HISTO_ZARR_OUTPUT    = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_100kmX100km_extended.zarr" # TODO: remove this and instea do it circular
    #   INPUT_ZARR           = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_100kmX100km_4th.zarr"
    #   # HISTO_ZARR_INPUT     = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr.zarr"
    #   # HISTO_ZARR_OUTPUT    = "/mnt/data2/UniBe-swiss-ndvi/input_data/ndvi_historic_v4_compr_extended.zarr" # TODO: remove this and instea do it circular
    #   # INPUT_ZARR           = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_ndvi_04_merged-v4_4th.zarr"
    #   # INPUT_LOOKUPTABLE = "/mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr"
    #   # INPUT_ZARR = "/mnt/data2/UniBe-swiss-ndvi/data/tmp_2026-04-29_07h16_ndvi_01_downloaded_2026-01-03_2026-01-03_processed.zarr"
    #   # HISTO_ZARR_INPUT = "/mnt/data1/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7.zarr/"
    #   # HISTO_ZARR_INPUT = "/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c.zarr/"
    
    # START PROCESSING:
    t0 = time.perf_counter()

    # Definition of output format of new
    # TODO: when going circular this is probably not needed anymore.
    COMPRESSOR = zarr3.Blosc(cname="zstd", clevel=3, shuffle=2)
    
    # N_WORKERS = 10           # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # N_THREADS_PER_WORKER = 1 # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1         # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000     # e) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 120s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '240GB'

    # N_WORKERS = 20        # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 90s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1      # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 90s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000  # b) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 90s (incl Zarr); 586503 pixels => XXs; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '190GB'
    # N_THREADS_PER_WORKER = 1

    N_WORKERS = 30        # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 97s (incl Zarr rewrite); 586503 pixels => (434s rewrite, 90s append); 16041205 pixels => 3300s; 105715396 pixels => XXs
    DATE_CHUNKS = -1      # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 97s (incl Zarr rewrite); 586503 pixels => (434s rewrite, 90s append); 16041205 pixels => 3300s; 105715396 pixels => XXs
    PIXEL_CHUNKS = 10000  # c) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 97s (incl Zarr rewrite); 586503 pixels => (434s rewrite, 90s append); 16041205 pixels => 3300s; 105715396 pixels => XXs
    MEMORY_PER_WORKER = '120GB'
    N_THREADS_PER_WORKER = 1

    PIXEL_CHUNKS    = 40000 # 10000  # TODO: with v5 move back from 500k,30 to 10k,365 or 40k,365
    DATE_CHUNKS_OUT = 365            # TODO: with v5 move back from 500k,30 to 10k,365 or 40k,365

    # TODO: check: 16041205 pixels in 640s in pipeline_FB_2026-03-19_09h09m26.log
    #              16041205 pixels in 3300s in pipeline_FB_2026-03-19_11h38m18.log
    #              Why so much longer? 
    #                 Is it due to the compression when writing? 
    #                 If so, then this would be smaller in case of appending.
    #                 The dashboard showed some computation to be indeed over after 10mins. Then "PerformanceWarning: Increasing number of chunks by factor of 245". And then dashboard didn't show any activity anymore.

    # N_WORKERS = 60        # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # DATE_CHUNKS = -1      # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # PIXEL_CHUNKS = 10000  # d) 13 dates (2026-11-30 to 2026-12-12): 4216 pixels => 33s; 586503 pixels => 57s; 16041205 pixels => XXs; 105715396 pixels => XXs
    # MEMORY_PER_WORKER = '66GB'
    # N_THREADS_PER_WORKER = 1
    
    t0=time.perf_counter()
    DASK_TEMP_DIR = "/mnt/data2/UniBe-swiss-ndvi/tmp_data6/"
    client = Client(
        n_workers=N_WORKERS,
        threads_per_worker=N_THREADS_PER_WORKER,
        memory_limit=MEMORY_PER_WORKER,
        local_directory= DASK_TEMP_DIR,
        processes=True,  # Use separate processes (not threads, but this appears to create non-shared memory)
        dashboard_address=':8343')
    print(client, flush = True)
    print(client.dashboard_link, flush = True) # use this dashboard to follow progress

    # DATE_CHUNKS  = historic_ds.chunks['date'][0]  # should be 30 days # TODO: why not this?
    # PIXEL_CHUNKS = historic_ds.chunks['pixel'][0]                     # TODO: why not this?

    historic_ds  = xr.open_zarr(HISTO_ZARR_INPUT, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    new_ds       = xr.open_zarr(INPUT_ZARR, chunks={}).chunk({"pixel": PIXEL_CHUNKS, "date": -1})
    lookuptable  = xr.open_zarr(INPUT_LOOKUPTABLE).chunk({"pixel": PIXEL_CHUNKS})

        # NOTE: minor fix historic_ds in v4 does not have mask_array as int8 but as bool. 
        # TODO: this is an error. Check with Francesco what values are needed.
        # historic_ds["mask_array"] = historic_ds["mask_array"].astype(np.int8)
        # historic_ds.to_zarr(HISTO_ZARR_INPUT.replace(".zarr",".zarr_bkp_fixedMaskArray"))
        # xr.open_zarr(HISTO_ZARR_INPUT.replace(".zarr",".zarr_bkp_fixedMaskArray"))
        # This has now been worked-around with .zarr_bkp_fixedMaskArray
    
    
    # NOTE: minor fix lookuptable does not have pixel as int32 but as int64. This prevents appending to zarr.
    lookuptable = lookuptable.assign_coords(
        # change number types of dimensions (pixel is that way 420MB instead of 840MB)
        pixel = ('pixel', lookuptable.pixel.values.astype(np.int32)),
        doy   = ('date', lookuptable.doy.values.astype(np.int32)))

    def show_ds_structure(ds):
        for c in list(ds.coords) + list(ds.data_vars):
            print(str(c).ljust(15) + ":   " + str(ds[c].encoding))
    
    #show_ds_structure(historic_ds)
    #show_ds_structure(new_ds)
    #show_ds_structure(lookuptable)

    print("Last dates in historic_ds:\n  "+"\n  ".join(np.datetime_as_string(historic_ds.date.isel(date = slice(-10,None)), unit='D')), flush=True)
    print("First dates in newly downloaded:\n  "+"\n  ".join(np.datetime_as_string(new_ds.date.isel(date = slice(0,10)), unit='D')), flush=True)
    # TODO: there is an overlap, do we need to remove this for application of continuous_ndvi()

    print("Current historic dataset:", flush = True)
    print(historic_ds, flush = True)
    print("Newly downloaded dataset:", flush = True)
    print(new_ds, flush = True)
    
    # --- concatenate full datasets along time ----------------------------------
    # Add median NDVI from model    
    # to new_ds:
    doy_noLeap = xr.where(new_ds.doy == 366, 365, new_ds.doy) # remove leap year if encountered
    new_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_noLeap,
            pixel=new_ds.pixel) # this is to join by pixels and doy
    # to historic_ds: # TODO: note that each time we are adding the medians to the historic data again and again. Maybe just add it once and store it?
    doy_noLeap = xr.where(historic_ds.doy == 366, 365, historic_ds.doy) # remove leap year if encountered
    historic_ds["median_ndvi"] = lookuptable["median_ndvi"].sel(
            doy=doy_noLeap,
            pixel=historic_ds.pixel) # this is to join by pixels and doy


    # Add mask_array to new_ds (filled with 0 or 2 at this point):
        # mask_array == 0: the data is not an observation and is yet to be smoothed
        # mask_array == 1: the data is not an observation and is smoothed
        # mask_array == 2: the data is an observation and is yet to be smoothed
        # mask_array == 3: the data is an observation and is smoothed
        # mask_array == 4: the data is an observation and is an outlier
    # THIS WAS TOO SIMPLE SINCE AN OBS_DATE DOES NOT COVER ALL OF CH: mask_0or2_1D = xr.where(new_ds["obs_date"], 2, 0).astype(np.int8)   # dims: date
    # THIS WAS TOO SIMPLE SINCE AN OBS_DATE DOES NOT COVER ALL OF CH: mask_0or2_2D = mask_0or2_1D.expand_dims({"pixel": new_ds.pixel})
    # THIS WAS TOO SIMPLE SINCE AN OBS_DATE DOES NOT COVER ALL OF CH: new_ds = new_ds.assign(mask_array=mask_0or2_2D)
    mask_2or0 = (
        (new_ds["obs_date"]) & 
        (new_ds["ndvi_obs"] < NO_COVERAGE) & 
        (new_ds["ndvi_obs"] > INVALID))
    new_ds['mask_array'] = xr.where(mask_2or0, np.int8(2), np.int8(0))

    # --- concatenate full datasets along time ----------------------------------
    new_ds = new_ds.rename(
        {'ndvi_obs':'ndvi_processed',
            'ndsi_obs':'ndsi_processed'}
    ).drop_vars('ndsi_processed')
    # Bind together with historic:
    merged_ds = (
        xr.concat(
            [historic_ds, new_ds], 
            dim="date")
        .sortby("date")
        .chunk({"pixel": PIXEL_CHUNKS, "date": DATE_CHUNKS})
    )

    # --- apply gapfilling and outlier detection function: continuous_ndvi() ----------------------------------

    # prepare arguments spanning historic and new data: all lazy
    ndvi_array   = merged_ds["ndvi_processed"].persist()
    median_array = merged_ds["median_ndvi"].persist()
    dates_array  = merged_ds["date"].persist()
    mask_array   = merged_ds["mask_array"].persist()
    # using persist() reduces graph size

    # reduce graph size by using futures
    # dates_future  = client.scatter(dates_array)
    # ndvi_future   = client.scatter(ndvi_array)
    # median_future = client.scatter(median_array)
    # dates_future  = client.scatter(dates_array)
    # mask_future   = client.scatter(mask_array)
    # then reference *_future inside tasks/closures instead of passing *_array
    # visualize(dates_future)

    # reduce graph size by handing NumPy arrays to dask:
    # dates_daskarray = da.from_array(dates_array)   # Hand NumPy array to Dask

    # call gufunc where core dim is "time" (1D arrays per pixel)
    output_dtypes = [ndvi_array.dtype, mask_array.dtype] # prespecify types
    ndvi_processed, mask_processed = xr.apply_ufunc(
        continuous_ndvi,
        ndvi_array,        # this is the observed/gapfilled/processed NDVI value
        median_array,      # this is the modelled median NDVI for the corresponding DOY
        mask_array,        # this is the integer processing status
        input_core_dims=[["date"], ["date"], ["date"]],    # each call gets 1D time arrays
        output_core_dims=[["date"],["date"]],
        vectorize=True, 
        dask="parallelized",
        kwargs={"dates_arr_original": dates_array},           # this contains all daily dates
        output_dtypes=output_dtypes, 
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    # ndvi_processed.isel(pixel=1, date=slice(3160,3170)).compute() # TODO: check why this is [ 4845,  4835,  4826,  4819, 32767, 32767, 32767, 32767, 32767, 32767]
    
    # g = mask_processed.__dask_graph__()
    g = ndvi_processed.__dask_graph__()
    print(f"Constructed graph with {len(g.layers)} layers, and {len(g)} tasks.", flush=True)
    #                    586_503 pixels:                 | 16_041_205 pixels:              | 105_715_396 pixels:
    # without persist(): 49    layers, and 196760 tasks  | 49    layers, and 1289428 tasks | xxx layers, and xxx tasks
    # with persist():    16-17 layers, and  31953 tasks  | 16-17 layers, and  872665 tasks | xxx layers, and xxx tasks
    # without persist(): .............. and 10.58 MiB
    # with persist():    size 23.08 MiB and 10.58 MiB
    
    # visualize(ndvi_processed)

    # --- append the new processed data to the historic_ds ----------------------------------

    # specifying where new data starts ()
    start_date = historic_ds['date'].max().values # TODO: actually this should start earlier to be able to overwrite L1 values to L2 status.

    historic_ds_to_extend = (
        historic_ds
        .drop_vars('median_ndvi')        # TODO: note that each time we are adding the medians to the historic data again and again. Maybe just add it once and store it?
        # .isel(date = slice(-10, None)) # NOTE just for development
    )

    ndvi_processed_to_append = ndvi_processed.sel(date = slice(start_date + 1, None)) # Note the shift +1
    mask_processed_to_append = mask_processed.sel(date = slice(start_date + 1, None)) # Note the shift +1
    ds_to_append = (
        xr.Dataset({"ndvi_processed": ndvi_processed_to_append, 
                     "mask_array":    mask_processed_to_append})
        .chunk({"pixel": PIXEL_CHUNKS, 
                 "date": DATE_CHUNKS_OUT})
    )

    ds_to_append.attrs["pixel_definition"] = historic_ds.attrs["pixel_definition"]


    #ndvi_processed_to_append.compute() 
    #mask_processed_to_append.compute()
    #ds_to_append.compute()               # starts on 2025-12-01 # Note the shift +1
    #historic_ds_to_extend.compute()      # ends   on 2025-11-30

    # For development
    # show_ds_structure(ds_to_append)
    # show_ds_structure(extended_historic_ds)
     


    def fallback_action_overwrite_zarr(outfile):
        # concatenate to complete dataset
        extended_historic_ds = (
            xr.concat([historic_ds_to_extend, ds_to_append], dim="date")
            .sortby("date")
            .chunk({"pixel": PIXEL_CHUNKS, 
                    "date": DATE_CHUNKS_OUT})
        )
        
        # Explicit encoding: simple compressor for each data var
        # encoding = {v: {"compressors": None      } for v in extended_historic_ds.data_vars} # TODO: why not? this should be following what was done to create v4 of historic
        encoding = {v: {"compressors": COMPRESSOR} for v in extended_historic_ds.data_vars}

        # drop any coord/data var chunk encodings that conflict   # TODO: is this needed?
        for name in list(extended_historic_ds.coords) + list(extended_historic_ds.data_vars): # TODO: remove this again if possilbe
            extended_historic_ds[name].encoding.pop("chunks", None)                           # TODO: remove this again if possilbe
            extended_historic_ds[name].encoding.pop("compressor", None)                       # TODO: remove this again if possilbe
            extended_historic_ds[name].encoding.pop("compressors", None)                      # TODO: remove this again if possilbe

        # overwrite (mode="w")
        extended_historic_ds.to_zarr(
            outfile, 
            mode="w", 
            compute=True,
            encoding=encoding, 
            zarr_format=3
        )

    if len(ds_to_append['date'].values) == 0: # this might be 0 if these dates have already been appended to the historic NDVI
        warnings.warn("Did not modify historic NDVI since no new dates were found.")
        raise ValueError("Did not modify historic NDVI since no new dates were found.")
    else:
        if HISTO_ZARR_OUTPUT == HISTO_ZARR_INPUT:
            print(f"appending to file\n  {HISTO_ZARR_OUTPUT}", flush=True)
            try:
                print("Appending new dates to existing zarr store...", flush=True)
                # NOTE: Ensure that we can append and the zarr structure remains intact and unchanged:
                # We had the issue that secondary coords got demoted to data variables.
                #   If there are mismatches (in coordinates or in dtypes ^*) it is 
                #   possible that secondary coordiantes get demoted from coordinates 
                #   to data variables. This breaks the workflow of continuous 
                #   appending from the next run on.
                #      ^* During development we encountered the issue that 
                #         HISTO_ZARR_OUTPUT had mask_array encoded as bool while 
                #         in ds_to_append it was correctly encoded as int8. This
                #         mismatch lead to ["x_idx", "y", "y_idx", "x"] being demoted 
                #         to data variables
                # BOTTOM LINE: ensure the data set to append uses exactly same 
                #              structure as the one to be appended:            
                # Opening HISTO_ZARR_OUTPUT is not needed, but just for checking the structure
                existing = xr.open_zarr(HISTO_ZARR_OUTPUT)
                assert sorted(list(ds_to_append.dims)) == sorted(list(existing.dims)), "Aborted append: dimensions are not equal"
                assert sorted(list(ds_to_append.coords)) == sorted(list(existing.coords)), "Aborted append: coordinates are not equal" # this is not strictly needed
                assert sorted(list(ds_to_append.data_vars)) == sorted(list(existing.data_vars)), "Aborted append: list of data variables are not equal" # this is not strictly needed
                for c in [c for c in list(ds_to_append.coords) if c not in ['date','doy']]:  # e.g. ["pixel", "x_idx", "y", "y_idx", "x"]:
                    assert ds_to_append[c].dtype == existing[c].dtype
                    assert ds_to_append[c].shape == existing[c].shape
                    # optional but safest:
                    assert (ds_to_append[c].values == existing[c].values).all()
                for c in list(ds_to_append.data_vars): # e.g ndvi_processed, mask_array
                    assert ds_to_append[c].dtype == existing[c].dtype
                # show_ds_structure(ds_to_append)

                # NOTE: dropping secondary coordinates (non-dimension coordinates) 
                #       seems safest to append
                # Only keep main coords (and doy) for correct appending:
                ds_to_append.drop_vars(["x_idx", "y", "y_idx", "x"]).to_zarr(
                    HISTO_ZARR_OUTPUT,
                    mode="a-",
                    append_dim="date",
                    compute=True,
                    encoding={},  # NOTE: since we append encoding must not be provided
                )
                # dds = xr.open_dataset(HISTO_ZARR_OUTPUT)
                
                # Post-writing check of resulting file content, if this fails do 
                # the fallback procedure and resolve to a full rewrite. 
                # NOTE: Unsure: Is a full rewrite still possible given that we 
                #       attempted to overwrite values with the appending above? 
                #       I do believe so, since we used (mode = "a-"), but not 100% sure.
                n_appended = ds_to_append.sizes['date']
                old_and_new_dates = (xr.open_dataset(HISTO_ZARR_OUTPUT)
                    .isel(date = slice(-n_appended-1,-n_appended+1))
                    .date.values) 
                if (old_and_new_dates[1] - old_and_new_dates[0]) != np.timedelta64(1, 'D'):
                    raise ValueError(f"Dates of resulting data set are not exactly 1 day apart at interface: {old_and_new_dates}")
                else:
                    print("Append successfully completed.", flush=True)

            except Exception as e:
                fallback_output = HISTO_ZARR_INPUT + ".failedAppending_" + dt.datetime.now().strftime("%Y%m%d%H%M%S")
                print(f"Appending failed: {e}. Writing whole file to {fallback_output}", flush=True)
                fallback_action_overwrite_zarr(fallback_output)

                # print(f"Appending failed: {e}. Falling back to rewrite.", flush=True)
                # # Backup original store (move directory) and write full dataset
                # backup = HISTO_ZARR_INPUT + ".backup_" + dt.datetime.now().strftime("%Y%m%d%H%M%S")
                # try:
                #     shutil.move(HISTO_ZARR_INPUT, backup)
                #     print(f"Backed up original store to {backup}", flush=True)
                # except Exception as e2:
                #     print(f"Backup failed: {e2} -- continuing to overwrite.", flush=True)
                # 
                # # duplicate of else
                # fallback_action_overwrite_zarr()
        else:
            print(f"writing to new file\n  {HISTO_ZARR_INPUT}\n=> {HISTO_ZARR_OUTPUT}", flush=True)
            fallback_action_overwrite_zarr(HISTO_ZARR_OUTPUT)

    client.close()

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds")

    print("Modified/Created file: ", flush = True)
    print(HISTO_ZARR_OUTPUT, flush = True)
    sys.exit(0)
