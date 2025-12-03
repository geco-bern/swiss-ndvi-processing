# Setting up download on UBELIX

## Initial setup 

Go to UBELIX OnDemand and start a session.
https://ondemand.hpc.unibe.ch/pun/sys/dashboard/batch_connect/sessions

Define the needed modules BEFORE starting your interactive job:
```
module load Anaconda3
```

In VSCode on UBELIX, select "Clone Git Repository" and establish GitHub authentication.
Then clone the needed repository.

Open README.md to read about Python environment initialization.
Open a Terminal in VSCode and do:
```
conda env create -f environment.yml --name ndvi
```

Wait until everything is installed.
Then you should be setup to go.


Not entirely it appeared that I still needed to do:
```
conda install tqdm
conda install conda-forge::pystac-client
## conda install pystac-client # (not available)
```




## Download data
Open the file: `processing/1_extract_swisstopo_dataset.py`
Ensure the right conda environment is used by clicking at the bottom and selecting
the `ndvi` environment (`~/.conda/envs/ndvi/bin/python`).

( Maybe you need to do `conda init` a first time. )

Then we can do it interactively through ondemand webpage (but limited to 12h running jobs)
or with a batch script:

```
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
```

```
ssh ubelix
cd ~/GitHub/geco-bern/swiss-ndvi-processing
conda activate ndvi
sbatch processing/1_extract_UBELIX.sh
```