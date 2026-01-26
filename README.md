# Gapfill NDVI

## How to Use the Package

To use the package, you have to install the required libraries using pip

### Setup with pip
Follow the steps outlined in `workflow_implementation/demo/00_setup.sh`.
They create a new virtual environment `ndvi` and install the necessary packages
with `pip`.

# Requirements

Big storage capacity are needed to store the intermediate results (see **prerequisites**) section. Using the workstation, (256GB RAM; 80 CPU) each passages expect for the first and last takes a couple of minutes maximum using 10 CPUs.


# How to process the NDVI data

Here I'll show how to process the NDVI data for a small subset of pixel. The analysis can be already run with this [script](workflow_implementation/demo/5_analyse_demo.py) without changing anything. 

If someone wants to try the entire workflow from downloading the satellite images to the NDVI processing, it must follows the script from 1 to 5 in [this folder](workflow_implementation/demo).

The script 0 contains the code to generate the means of upper and lower bands using the Samantha's model. In order to reproduce the dataset, is necessary to follows her instruction in [processing folder](processing). We already have generated the lookup table for all doy and pixels, in [lookup_table folder](data_for_demo/lookup_table.zarr) are stored the values for the subset of pixels used in the demo.

## Prerequisites

To process the data, 2 dataset are needed. 

- The historical NDVI processing with all the past observation.
- The lookup table containg the means upper and lower precomputed per doy for each pixels

Both dataset are already generated and are stored inside the workstation. We upload the dataset for the demo in [data folder](data_for_demo).

Below I'll explain how to use the demo. The intermediate data generated from step 1 to step 3 contains all the pixels (105M) and cannot be uploaded on Github. For this reason the only lines of code to change are the ones used to store the intermediate files. The storage size needed is in order of hundreds of GB.

## Simulate the continous NDVI processing

To simulate the continous NDVI processing, the first step is to download the data.

### Donwload the data

The script [1_extract_swisstopo_dataset.py](workflow_implementation/demo/1_extract_swisstopo_dataset.py)
 will download the data, in [line 124](workflow_implementation/demo/1_extract_swisstopo_dataset.py#L124)
 is it possible to select the time window to simulate the continous ingestion. I select to ingest data from 2018-06-01 to 2018-06-05.

#### Required parameter to modify inside the script

- the output path in [line 30](workflow_implementation/demo/1_extract_swisstopo_dataset.py#30)
- the starting and ending date in [line 124](workflow_implementation/demo/1_extract_swisstopo_dataset.py#L124), if needed.

### Transpose the data from time-wise to space-wise chunking

The following step are to transpose the dataset from time-wise chunking to space-wise chunking, the script [2_transpose_swisstopo_dataset.py](workflow_implementation/demo/2_transpose_swisstopo_dataset.py) will do that.

#### Required parameter to modify inside the script

- the input path in [line 11](workflow_implementation/demo/2_transpose_swisstopo_dataset.py#L11). !!! IMPORTANT Must be equal to the output path of the previous step.
- the output path in [line 13](workflow_implementation/demo/2_transpose_swisstopo_dataset.py#L13).

### Add the new date

The script (3_add_dates.py)[workflow_implementation/demo/3_add_dates.py] will download the new date where an observation in present, extented to be evenly spacing at daily resoultion and create the mask of where an observation is found (this mask in used in continous ndvi setup). Here there is nothing to change and can be run immidiately.

#### Required parameter to modify inside the script

There is nothing to modify here

### Merge the historical dataset with the new data set

To run the analysis, it is encessary to have the historical analysis and the newly acquired data. The script [4_merge_zarr.py](workflow_implementation/demo/4_merge_zarr.py) will load both dataset and merged togheter. The new median ndvi data will be added using the lookup table.

#### Required parameter to modify inside the script

- The path of input new data in [line 15](workflow_implementation/demo/4_merge_zarr.py#15). This must be the same path as the output file in the previous script. 
- The path of temporary output in the following line before the merging.

If the starting end ending dates in [1_extract_swisstopo_dataset.py](workflow_implementation/demo/1_extract_swisstopo_dataset.py) have been changed, it is also necessary to change them in line 91, 92, 177, 191


### Run the analysis

After the merging, it is possible to run the analysis with the script [5_analyse_demo.py](workflow_implementation/demo/5_analyse_demo.py). The analysis can be already run as it is.

The output data will also have a mask map for each pixel and date with the following values:

- **0**: the data is not an observation and is yet to be smoothed
- **1**: the data is not an observation and is smoothed
- **2**: the data is an observation and is yet to be smoothed
- **3**: the data is an observation and is smoothed
- **4**: the data is an observation and is an outlier

#### Required parameter to modify inside the script

There is nothing to modify here.

### Create COG tiff

The script [6_create_cogtiff.py](workflow_implementation/demo/6_create_cogtiff.py) will generate the cogtiff file based on the analysis. To work as it is now, it will read all the file stored [here](data_for_demo/output_cogtiff) which contains the tiff files for each date. 

From them it will select the newest date and it will check all the remaining dates. If for a given date the number masked data have values 1 or 3, as specified above, exceed the threshold specified in [line 21](workflow_implementation/demo/6_create_cogtiff.py#L21), the script will generate the tiff of the selected date. Please note that putting the threshold at 1 (100%) will not generate anything because there are some pixels with no observation.

The tiff generation can be run it as it is.

#### Required parameter to modify inside the script

In [line 37](workflow_implementation/demo/6_create_cogtiff.py#L37)is it possible to remove the -100 create more tiff files
