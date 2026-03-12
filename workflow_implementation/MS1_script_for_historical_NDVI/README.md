# Milestone 1: Historical NDVI analysis

This folder contains all the detailed information of the script used to generate the historical NDVI anlaysis.
The first 5 script (from 0 to 4) are also used to perform the continous NDVI processing.

## 0_create_lookuptable.py

The first part of the processing is to create the mean of the bands generate it by Samantha.
To do so, we start by the already generated set of parameters for each pixels. If someone wants to geneate the parameters it can follows the processing folder (or simply we can send it).

We calculate the lower and upper band for each doy (from 1 to 365), take the mean between the two and save it in a zarr folder.

## 1_extract_swisstopo_dataset.py

To download the satellite images, we use the pystac_client library. We cover the entire Switzerland from 2017-04-01 to 2025-11-30. In case a NDVI value is missing, we set a placeholder of -2^15 for filtered out pixels, as cloud shadows, and 2^15 - 1 for the pixels with no data for the given date.

## 2_transpose_swisstopo_dataset.py

This script will simply transpose the downloaded dataset using zarr, since the downloaded values are chunked time-wise.

## 3_add_dates.py

This script will add the dates in which an observation is found. We use pystac_client as before to retrive the date array for the whole period and we added to the transposed zarr folder.

We noticed that the array of dates contains 1181 values, wherease the transposed NDVI array has 1180 timesteps. We noticed that there are missing data on date 2021-02-05, this is the log message

```bash
'/vsicurl/https://data.geo.admin.ch/ch.swisstopo.swisseo_s2-sr_v100/2021-02-05t102221/ch.swisstopo.swisseo_s2-sr_v100_mosaic_2021-02-05t102221_masks-10m.tif' 
does not exist in the file system, and is not recognized as a supported dataset name.
```

For this reason, we filter out that specific date.

Up to this script, the historic and continous NDVI processing are identical

# 4_create_zarr_for_analysis.py

This script will preapre the final zarr folder to peform the historical NDVI analysis.

The first things that is done is to remove the missing date, so to have the date and NDVI arrays of the same lenghts.

After that, we generate an array of dates evenly spaced at daily resoultion.
Starting from the original date and NDVI arrays, we map the evenly spaced date array with the NDVI, when no NDVI is avaible for a specific date we use a placeholder (2^15 -1).

We then generate a boolean array starting mapping the original date array into the evenly spaced array. This array will be used in the historical analysis to create the mask for the final product.

We then added the median NDVI values based on the lookup table created in the script 0. To do so, we generated an additional array based on the evenly spaced date, we calculate the create the median array by mapping the doy and pixel values.

We lastly combined altogheter and save the zarr folder in the V2 format. We didn't save it in V3 format because Xarray doesn't support the new format yet.

## 5_historic_NDVI.py

In this script, we run the historical NDVI analysis.

The output is composed by two arrays for each pixels, the first is the processed NDVI array and the second is the mask indicating the nature of each NDVI value. The mask will have integer values from 0 to 4 according to this list

- **0**: the data is not an observation and is yet to be smoothed
- **1**: the data is not an observation and is smoothed
- **2**: the data is an observation and is yet to be smoothed
- **3**: the data is an observation and is smoothed
- **4**: the data is an observation and is an outlier

The processing starts with the outlier detection, we filter the outlier based on the same method as in the method report.
The two parameters, distance from the median (delta) and distance with the neighbouring deltas (delta_delta) are set to 0.1 but it can be changed in line 35 and 36.

After the outlier detection is performed, if a timeserie have at least 7 observation, we perform the smoothing of the deltas using the LOESS function as described in the method report. We avoid to perform the smoothing if one of the 2 following cases are encountered

- extreme negative NDVI deltas: at least 5 out 7 deltas have values lower than -0.2
    - this will include only the extreme negative events (fire, non drought)
- at least one value is close to the boundaries conditions (0.95, 0.05)
    - this will prevent the processed NDVI to be outside the boundaries condition

If the window data to smooth doesn't fall in these tow categories, we smooth the delta values from the first deltas to the last-fourth deltas, then we combined the smoothed deltas with the remaning deltas (the last three) and we linearly interpolate the deltas in the dates with no observation.

After the NdVI processing is performed, we generate the mask array according to the logic explained before.

## 6_append_coords_to_historic_ndvi.py
This simplifies use of the historic NDVI data set by
- appending x and y coordinates
- reduces file size by using compression
- and uses a better chunking structure to facilitate appending new NDVI data
