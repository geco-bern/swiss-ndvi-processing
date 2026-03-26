# Workflow report

In this report, we will explain in detail each script used to perform the historical.

## Table of content

- [Aim of the project](#aim)

0. [Summary of method](#method)

    1. [Create lookup table](#lookptable)
    2. [Download satellite images](#donwload)
    5. [Create zarr folder for historical analysis](#create_zarr)
    6. [Run historical NDVI processing](#historic)

<a name = "aim"></a>

# Aim of the project

The goal of the project is to create a workflow that automatically process newly acquired NDVI data from Senitnel-2 at 10m of spatial resolution at daily scale. The workflow must be able to differentiate the new data as true observation or outlier. To do so, we have created an outlier detection method based on 2 global parameters. Each data is validated by calculating the difference with the expected value and the actual one, called delta, and the difference between the delta of the observed value to the neighbouring delta, called delta-delta. 

The validation is made with 2 global parameters: the delta threshold and delta-delta threshold, but set at 0.1.


<a name = "method"></a>

# Summary of method

Here, I'll described the method proposed to perform the NDVI processing on the full timeserie from april 2017 to november 2025.



<a name = "lookptable"></a>

## 0_create_lookuptable.py

The analysis includes the donwload of satellite images, the pixel-wise outlier detection, smoothing the observed NDVI and linearly interpolate the missing data between observation at dailiy scale.

To do so, we used the model developed by Samantha, generating the 6 parameters required to calculate the lower and upper double logistic functions. We average the values for each DOY and pixel to create a lookuptable used to perform the analysis.

The lookup table was computed on our machine and the data are transferred at /mnt/data1/UniBe-swiss-ndvi/data/lookup_table_median_ndvi.zarr.

<a name = "donwload"></a>

## 1_extract_swisstopo_dataset.py

To download the satellite images, we use the pystac_client library. We cover the entire Switzerland from 2017-04-01 to 2025-11-30. The data were selected based on the forest mask avaible on Swisstopo VHI dataset.

We apply a filter based on 4 bands, which at least one conditions is met  (green == 9999) | (swir_10m == 9999) (terrain_mask == 255) | (cloud_mask == 255) 

After the filtering, the NDVI and NDSI are computed, the NDVI is filtered out when NDVI >= 0.43. The missing data are flagged with a placeholder of -2^15. The values with no data at given timestep (due to the different orbit) are flagged with a value of 2^16 -1. 

TODO: add forset_mask explanation

Along with the NDVI and NDSI, we retrieve the date of each image, the pixel ID and spatial idx and coordinates, the final output will look like this TODO: add it


## 2_historic_NDVI.py

The second script will perform the NDVI processing oh historical data. The analysis can be split in three parts:

- outlier detection
- anomalies detection and smoothing
- interpolation at dailiy scale

The outlier detection is the first step to filter out the non-observation values and outliers.

The scaled-down observation must be within 0-1 so it is easy to filter out the missing data with -2^15 or no data with 2^16 -1.

After the first filter is applied, we remove the outliers, we defined an outlier according to this defintion:

- The difference between the absolute NDVI value and the corresponded expected value (hereinafter called median) is above a a threshold (0.1), so called delta.
- The difference between the current analysed delta and the two neighbourh delta is above a threshold (0.1), so called delta-delta.

When both conditions are met, the value is flagged as outlier and is removed from the timeserie.

The following passage is to create the delta timeserie used to linearly interpolate the missing data. This array is created by evaluating each delta on a rolling window of 7 values. 

Within the window, we check if the data inside the window are close to the boundaries conditions or have extreme negative NDVI values, if one (or both) conditions are met the delta is added as it is, otherwise the smoothing is performed. 

the smoothing of the deltas is performed on the non-outlier observed NDVI. We use the LOESS alogorithm to perform the smoothing on a rolling window of 7 observation and 3 iterations of the algoritm. 

After the rolling window reach the last-fourth observation (so that is centered) we cannot proceed using this method, hence we append the remaining deltas that will be flagged as "observation yet to smooth" (L1 linearly interpolated product).

After the delta timeserie is created, we linearly interpolate the results by taking into account their position on the timeserie, the interpolated delta are summed to the medians NDVI to obtain the processed NDVI values.

The final timeserie will have from the third to the last fourth observation the smoothing values (L2) and from the day after the alst fourth observation onwards the linarly delta interpolation (L1).

After the processing, we create the mask array for the TIFF generation. The mask will have integer values from 0 to 4 according to this list

- **0**: the data is not an observation and is yet to be smoothed
- **1**: the data is not an observation and is smoothed
- **2**: the data is an observation and is yet to be smoothed
- **3**: the data is an observation and is smoothed
- **4**: the data is an observation and is an outlier

## 3 TIFF generation

The TIFF are generated ...