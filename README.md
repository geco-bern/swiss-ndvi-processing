# Smooth and gap-free NDVI of Swiss forests

## Aim
Vegetation health and activity can be monitored based on remotely sensed data of the Normalized Difference Vegetation Index (NDVI). Sentinel-2 provides NDVI data at high spatial resolution (10 m) at frequent revisit times (up to 5 days) and an additionally enhanced data product (swissEO S2-SR), covering the area of Switzerland, is made available through swisstopo. However, challenges for data interpretability and usability for operational vegetation health (https://www.swisstopo.admin.ch/de/satellitenbilder-swisseo-vhi) and national drought monitoring (https://www.trockenheit.admin.ch/en) remain. Challenges arise from data gaps, the large scatter, and apparent outliers in the data. The data availability largely depends on the overpass date of the Sentinel-2 satellites, cloud, and snow cover.

<!-- <figure style="text-align: center;">
    <img src="illustrations/00_NDVI_availability.png" style="width:60%;">
    <figcaption>Figure 1: Illustration of data availability of NDVI values during October 2025 in the swissEO S2-SR product (screenshot from https://www.swisstopo.admin.ch/en/satelliteimage-swisseo-s2-sr).</figcaption>
</figure> -->

In this project, we create a workflow to process the NDVI data from swissEO S2-SR, creating a smooth and gap-free data product at the daily scale and 10 m resolution, covering forest areas Switzerland. The processing includes:

- Filtering of observations based on cloud and snow cover data
- Identification of outlier observations
- Smoothing and interpolation of the observations

The workflow is applied to available data from April 2017 to November 2025 (historical processing) and a setup to work with newly incoming data is developed and presented. Accordingly, the project yields two products:

1. **Historical processing (HP)**: A smooth and gap-free daily NDVI product, based on S2-SR, covering past dates (April 2017 to November 2025). These data are made available as Cloud Optimized GeoTIFF files.
2. **Continuous integration (CI)**: An algorithm to continuously ingest incoming NDVI data from S2-SR and update cleaning, gapfilling, and smoothing in consistency with the _Historical processing_. 


Further details are found in the report: `report/method_report_v2.ipynb`

## How to Run HP
Details how to re-run the historical processing are found in section 3.1 of the report: `report/method_report_v2.ipynb`.

The full workflow is implemented in by the bash script: `0_1_run_historic_analysis.sh`. Log files are created into folder `workflow_implementation/MS2_continuous/logs`. It can all be run e.g. with `ssh tunder; tmux; cd /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_historical/; bash 0_1_run_historic_analysis.sh`

In a nutshell it can be done with:
```bash
ssh tunder
tmux 
cd /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS1_historical/

bash 0_1_run_historic_analysis.sh
# and check progress on the dashboard (for the dashboard link see log files in workflow_implementation/MS2_continuous/test_all_pixels/pipeline_logs)
```


## How to Run CI
Details how to run the continuous integration with example datasets (not contained in this repo, but already provided on tunder) are found in section 3.2.2 and Appendix B of the report: `report/method_report_v2.ipynb`

To run `0_1_run_pipeline.sh`: 
- first update following parameters in the script:
    - `HISTO_INPUT="/mnt/data2/UniBe-swiss-ndvi/historic_data/historical_2026-04-04_18h16_historical_v7c_SUBSET-focus-sites.zarr"`
    - `END_DATE="2026-01-06"`
    - `START_DATE`, (optional) uses by default the last date of `HISTO_INPUT`
    - `HISTO_OUTPUT` if you do not want to extend to the existing `HISTO_INPUT` (ensure the parent directory exists)
- second run the shell script, e.g. by doing : `ssh tunder; tmux; cd /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels; bash 0_1_run_pipeline.sh`
- check the log file in folder `workflow_implementation/MS2_continuous/test_all_pixels/pipeline_logs`
- (optional): for repeated testing, it can be good to outcomment certain steps (e.g. slow downloading)

In a nutshell it can be done with:
```bash
ssh tunder
tmux 
cd /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/MS2_continuous/test_all_pixels

bash 0_1_run_pipeline.sh

# and check progress on the dashboard (for the dashboard link see log files in workflow_implementation/MS2_continuous/test_all_pixels/pipeline_logs)
```