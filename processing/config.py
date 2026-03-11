from rasterio.coords import BoundingBox
from rasterio.transform import Affine
from rasterio.crs import CRS

REF_BBOX = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
REF_BBOX_4326 = BoundingBox(left=5.70, bottom=45.8, right=10.6, top=47.95)

REF_WIDTH = int((REF_BBOX.right - REF_BBOX.left) / 10.0)
REF_HEIGHT = int((REF_BBOX.top - REF_BBOX.bottom) / 10.0)

REF_TRANSFORM = Affine(10.0, 0.0, REF_BBOX.left,
                       0.0, -10.0, REF_BBOX.top)

REF_CRS = CRS.from_epsg(2056)

CHUNK_SIZE = 4000

DATA_DIR = "/data_2/scratch/sbiegel/processed"
TEMPORAL_DATASET_ZARR = f"{DATA_DIR}/ndvi_dataset_temporal.zarr"
SPATIAL_DATASET_ZARR = f"{DATA_DIR}/ndvi_dataset_spatial.zarr"
FOREST_MASK = f"{DATA_DIR}/forest_mask.npy"
DASK_LOCAL_DIRECTORY = f"{DATA_DIR}/dask_worker_space"

SERVICE_URL = 'https://data.geo.admin.ch/api/stac/v0.9/'

INVALID = -2**15 # Filtered out pixels, e.g. cloud shadows
NO_COVERAGE = 2**15 - 1 # Pixels with no data for the given time step