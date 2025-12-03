#!/bin/bash
#------------------------
#SBATCH --account=invest
#SBATCH --qos=job_icpu-stocker
#------------------------
#SBATCH --job-name="NDVI data download"
#SBATCH --time=169:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=12G

# Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate ndvi

python3 processing/1_extract_swisstopo_dataset.py 