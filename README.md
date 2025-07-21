## Setup with conda

```
conda create -n ndvi python=3.12
conda activate ndvi
conda install -c conda-forge gdal sqlite numpy zarr rasterio tqdm dask pandas taudem
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install pystac_client whitebox
pip install -e .
```
