# Gapfill NDVI

## How to Use the Package

To use the package, you have to download the required libraries using **Conda** or **Miniconda**.

### Setup with Conda
```bash
conda env create -f environment.yml --name ndvi
```

# Content

The demo is divided into two notebooks:  

1. **Smoothing options**  
2. **NDVI gapfilling**  

Both can be found here: `demo/notebook/`

---

## Pixel Selection

The notebooks can be used without any changes.  
We select **25 pixels** for each of the following biomes in this specific order:

| Biome                                | Coordinates (x, y)         | Pixel Range |
|--------------------------------------|----------------------------|-------------|
| Lowland broadleaf                     | 2694027.49, 1123239.84    | 0–24        |
| Highland broadleaf                    | 2694027.49, 1123239.84    | 25–49       |
| Lowland evergreen                     | 2613028.38, 1127777.24    | 50–74       |
| Highland evergreen                    | 2782037.00, 1183475.00    | 75–99       |
| Biscth fire affected area             | 2644008.20, 1133794.76    | 100–124     |
| Biscth fire nearby non-affected area  | 2644328.07, 1134342.81    | 125–149     |
| Drought-affected area                 | 2690025.48, 1287413.03    | 150–174     |
| Burlgim storm affected area           | —                         | 175–199     |

---

## Area of Interest

The areas of interest can be visualized directly in the notebook by inserting the coordinates.  

The data is stored here: `demo/pixel_biomes.zarr`  

To select any biome, choose the pixels accordingly in the given order.

The areas affected by Burglind storm area are generously sended by Marius Rüetschi, which collects the area with  almost 100% of damage. You can find the area affected [here](https://www.sturmarchiv.ch/index.php?title=20180103_01_Storm_Alpennordseite
)

---

## Smoothing Notebook

The **smoothing option notebook** presents three different smoothing methods:  

- **Savitzky–Golay filter**  
- **LOESS**  
- **Low-pass filter**  

We believe that the **Savitzky–Golay filter yields the best results**.

---

## NDVI Gapfilling Notebook

The **NDVI gapfilling notebook** provides a demo and explanation of the gapfilling process.  

It provides the full gapfilling using:  

- **L1**: linear interpolation  
- **L2**: smoothing with Savitzky–Golay  
- **Continuous integration setup**: L1, then smoothed with L2  


# How to process the NDVI data

Here I'll show how to process the NDVI data for a small subset of pixel. The analysis can be already run with this [script](workflow_implementation/demo/5_analyse_demo.py) without changing anything. 

If someone wants to try the entire workflow from downloading the satellite images to the NDVI processing, it must follows the script from 1 to 5 in [this folder](workflow_implementation/demo).

The script 0 contains the code to generate the means of upper and lower bands using the Samantha's model. In order to reproduce the dataset, is necessary to follows her instruction in [this folder](processing). We already have generated the lookup table for all doy and pixels, in [this folder](data_for_demo/lookup_table.zarr) are stored the values for the subset of pixels used in the demo.

## Prerequisites

To process the data, 2 dataset are needed. 

- The first is the historical NDVI processing with all the past observation.
- The lookup table containg the means upper and lower precomputed per doy for each pixels

Both dataset are already generated and are stored inside the workstation. We upload the dataset for the demo in [this folder](data_for_demo).

Below I'll explain how to use the demo. The intermediate data generated from step 1 to step 3 contains all the pixels (105M) and cannot be uploaded on Github. For this reason the only lines of code to change are the ones used to store the intermediate files.

## Simulate the continous NDVI processing

To simulate the continous NDVI processing, the first step is to download the data.

### Donwload the data

The script [1_extract_swisstopo_dataset.py](workflow_implementation/demo/1_extract_swisstopo_dataset.py)
 will download the data, in [line 124](workflow_implementation/demo/1_extract_swisstopo_dataset.py#L124)
 is it possible to select the time window to simulate the continous ingestion. I select to ingest data from 2018-06-01 to 2018-06-05 (TODO: understand how the starting and ending dates can be passed automatically).

#### Required parameter to modify inside the script
  - the starting and ending date in [line 124](workflow_implementation/demo/1_extract_swisstopo_dataset.py#L124)
  - the outputpath in [line 142 and 153](workflow_implementation/demo/1_extract_swisstopo_dataset.py#L142)

### Transpose the data from time-wise to space-wise chunking

The following step are to transpose the dataset from time-wise chunking to space-wise chunking, the script [2_transpose_swisstopo_dataset.py](workflow_implementation/demo/2_transpose_swisstopo_dataset.py) will do that.

#### Required parameter to modify inside the script

- the input path in [line 11](workflow_implementation/demo/2_transpose_swisstopo_dataset.py#L11). !!! IMPORTANT Must be equal to the output path of the previous step.
- the output path in [line 13](workflow_implementation/demo/2_transpose_swisstopo_dataset.py#L13).

### Add the new date

The script (3_add_dates.py)[workflow_implementation/demo/3_add_dates.py] will download the new date where an observation in present, extented to be evenly spacing at daily resoultion and create the mask of where an observation is found (this mask in used in continous ndvi setup). Here there is nothing to change and can be run immidiately.

### Merge the historical dataset with the new data set

To run the analysis, it is encessary to have the historical analysis and the newly acquired data. The script [4_merge_zarr.py](workflow_implementation/demo/4_merge_zarr.py) will load both dataset and merged togheter.

#### Required parameter to modify inside the script

- The path of input new data in [line 15](workflow_implementation/demo/4_merge_zarr.py#15). This must be the same path as the output file in the previous script. 
- The path of temporary output in the following line before the merging.

### Run the analysis

After the merging, it is possible to run the analysis with the script [5_analyse_demo.py](workflow_implementation/demo/5_analyse_demo.py). 

#### Required parameter to modify inside the script

There is nothing to modify here.

### Create COG tiff

The 
#### Required parameter to modify inside the script

Once the forset pixel mask is avaible nothing, since it will use the data from [here](data_for_demo/processed_ndvi.zarr)