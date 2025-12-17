# Workflow report

In this report, we will explain in detail each script used to perform the historical and continous NDVI processing.

The model used can be found in detail in the method report. Here we explaiined how each of the passages are developed.

## Table of content

0. [Summary of method](#method)

1. [Milestone 1: historical NDVI processing](#MS1)
    0. [Create lookup table](#lookptable)
    1. [Download satellite images](#donwload)
    2. [Tranpose zarr folder](#transponse)
    3. [Add dates](#add_dates)
    4. [Create zarr folder for historical analysis](#create_zarr)
    5. [Run historical NDVI processing](#historic)
2. [Milestone 2: continous NDVI processing](#MS2)
    4. [Merge historical and aquired NDVI](#merge_zarr)
    5. [Run continous NDVI processing](#continous)
    6. [Genreate TIFF files](#tiff)

<a name = "method"></a>

# Summary of method

Here, I'll briefly described the method proposed to filter out the outlier, smooth the observation and linearly interpolate the daily data between observation.

An outlier is encountered when both these conditions are met

- The difference between the absolute NDVI value and the corresponded expected value (hereinafter called median) is above a certaion threshold (0.1), so called delta.
- The difference between the current analysed delta and the two neighbourh delta is above certain threshold (0.1), so called delta-delta.

The outlier data are removed from the analysis.

After the outlier removal, we perform the smoothing of the deltas. We select to perfrom the smoothing using the LOESS algorithm with a moving window of 7 observation and a 3 cycle of iteration. For each window, we take the 4th value, which is exatcly in the middle of the window.

We avoid to perform the smoothing if one of the following two conditions are met

- Extreme negative NDVI values: if 5 out of 7 deltas are below - 0.2 we do not smooth the 4th but we take it as it is
    - The idea is that under extreme conditions the vegetation may not follow the expected seasonality
- If a absolute NDVI value is close to the boundaries conditions (0.95, 0.05)
    - We do so to prevent the smoothed deltas to exceed the boundaries conditions

Once we have the completed smoothed deltas timeserie (from the first to the last fourth observed deltas are smoothed), we linearly interpolate the remaining missing data. We lastly sum the interpolated deltas with the medians to obtain the smoothed and gapfilled NDVI.

<a name = "MS1"></a>

# Milestone 1: Historical NDVI analysis

This folder contains all the detailed information of the script used to generate the historical NDVI anlaysis.
The first 5 script (from 0 to 4) are also used to perform the continous NDVI processing.

<a name = "lookptable"></a>

## 0_create_lookuptable.py

The first part of the processing is to create the mean of the bands generate it by Samantha.
To do so, we start by the already generated set of parameters for each pixels. If someone wants to geneate the parameters it can follows the processing folder (or simply we can send it).

We calculate the lower and upper band for each doy (from 1 to 365), take the mean between the two and save it in a zarr folder.

<a name = "donwload"></a>

## 1_extract_swisstopo_dataset.py

To download the satellite images, we use the pystac_client library. We cover the entire Switzerland from 2017-04-01 to 2025-11-30. In case a NDVI value is missing, we set a placeholder of -2^15 for filtered out pixels, as cloud shadows, and 2^15 - 1 for the pixels with no data for the given date.

<a name = "transponse"></a>

## 2_transpose_swisstopo_dataset.py

This script will simply transpose the downloaded dataset using zarr, since the downloaded values are chunked time-wise.

<a name = "add_dates"></a>

## 3_add_dates.py

This script will add the dates in which an observation is found. We use pystac_client as before to retrive the date array for the whole period and we added to the transposed zarr folder.

We noticed that the array of dates contains 1181 values, wherease the transposed NDVI array has 1180 timesteps. We noticed that there are missing data on date 2021-02-05, this is the log message

```bash
'/vsicurl/https://data.geo.admin.ch/ch.swisstopo.swisseo_s2-sr_v100/2021-02-05t102221/ch.swisstopo.swisseo_s2-sr_v100_mosaic_2021-02-05t102221_masks-10m.tif' 
does not exist in the file system, and is not recognized as a supported dataset name.
```

For this reason, we filter out that specific date.

Up to this script, the historic and continous NDVI processing are identical

<a name = "create_zarr"></a>

# 4_create_zarr_for_analysis.py

This script will preapre the final zarr folder to peform the historical NDVI analysis.

The first things that is done is to remove the missing date, so to have the date and NDVI arrays of the same lenghts.

After that, we generate an array of dates evenly spaced at daily resoultion.
Starting from the original date and NDVI arrays, we map the evenly spaced date array with the NDVI, when no NDVI is avaible for a specific date we use a placeholder (2^15 -1).

We then generate a boolean array starting mapping the original date array into the evenly spaced array. This array will be used in the historical analysis to create the mask for the final product.

We then added the median NDVI values based on the lookup table created in the script 0. To do so, we generated an additional array based on the evenly spaced date, we calculate the create the median array by mapping the doy and pixel values.

We lastly combined altogheter and save the zarr folder in the V2 format. We didn't save it in V3 format because Xarray doesn't support the new format yet.

<a name = "historic"></a>

## 5_historic_NDVI.py

In this scirpt, we run the historical NDVI analysis.

The output is composed by two arrays for each pixels, the first is the processed NDVI array and the second is the mask indicating the nature of each NDVI value. The mask will have integer values from 0 to 4 according to this list

- **0**: the data is not an observation and is yet to be smoothed
- **1**: the data is not an observation and is smoothed
- **2**: the data is an observation and is yet to be smoothed
- **3**: the data is an observation and is smoothed
- **4**: the data is an observation and is an outlier

The processing starts with the outlier detection, we filter the outlier based on the same method as in the method report.
The two parameters, distance from the median (delta) and distance with the neighbouring deltas (delta_delta) are set to 0.1 but it can be changed in line 35 and 36.

After the outlier detection is performed, if a timeserie have at least 7 observation, we perform the smoothing of the deltas using the LOESS function as described in the method report. We smooth the delta values from the first deltas to the last-fourth deltas when do not meet the conditions described in the method section. We then combined the smoothed deltas with the remaning deltas (the last three) and we linearly interpolate the deltas in the dates with no observation.

Aftetr the NdVI processing is performed, we generate the mask array according to the logic explained before.

<a name = "MS2"></a>

# Milestone 2: continous NDVI setup

The continous NDVI processing follows the same step of historical NDVI processing from script 0 to 3, there is no need to create tow separate lookup table. For this reason, we documented the last three script. The main differences here are the starting and ending date in which the analysis is performed. Both can be find and the top of the script right after the library loading except for **5_continous_NDVI.py** where they are located at the bottom of the script inside the **__main__** module

<a name = "merge_zarr"></a>

## 4_merge_zarr.py

The first part of this script is identical to the generation of the zarr folder for the historical NDVI processing.
The differences between the two scripts are found in the second half on this script. Here we stack the new obtained data to the previous "historic" zarr folder.

<a name = "continous"></a>

## 5_continous_NDVI.py

The processing of NDVI on continous setup is identical to the historical processing. IN this case however there are some notable differences.

- If the newly acquired data (for each pixels) is not an observation, we estimate the NDVI based on the exponential deacy as described in the method report.

- If the data is a potential outlier, we skip the computation and will be hcecked when a new observation is found for that timeserie.

- If the data is an observation, we trigger the analysis as in the historical NDVI processing. Here we evaluate only the last 7 observation, we smooth and interpolate between the last fifth and fourth observation and we interpolate between the last and current observation, as described in the method report.

The same two parameters (delta_treshold and delta_delta treshold) are use here and can be tuned in line X and Y.

<a name = "tiff"></a>

## 6_create_cogtiff.py

The last part of the processing is the COGTIFF generation.

With the last scirpt, we generate the TIFF file for each date, covering the whole Switzerland.

To keep track of which date have already been processed, we decided to save the file TIFF file with the following name "YYYY-MM-DD.tiff". In this way we are able to know immidiately which is the last date to be processed.

From there, we can evaluate each of the remaining date (up to the current date). If the amount of smoothed data (indicated by hte mask array generated in the previous script) reach a certain percentage the processing is triggered.

The percentage of smoothed data per date can be tune in line X, be wary that if the percentage is set to 1, no tiff file will be created because there are some pixels with no observation.