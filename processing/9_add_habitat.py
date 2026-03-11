"""
Add habitat type frequencies from habitat map to the dataset.
"""
import rasterio
import numpy as np
import subprocess
from numba.typed import Dict
from numba import types
from numba import njit
import zarr
from rasterio.windows import Window
from tqdm import tqdm

from processing.config import REF_BBOX, REF_WIDTH, REF_HEIGHT, CHUNK_SIZE, TEMPORAL_DATASET_ZARR, FOREST_MASK, DATA_DIR

NODATA_VAL = 65535

def align_habitat_map(habitat_path, aligned_path):
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:2056",
        "-te", str(REF_BBOX.left), str(REF_BBOX.bottom), str(REF_BBOX.right), str(REF_BBOX.top),
        "-tr", str(1), str(1),
        "-tap",
        "-srcnodata", str(NODATA_VAL),
        "-dstnodata", str(NODATA_VAL),
        "-co", "BLOCKXSIZE=1024",
        "-co", "BLOCKYSIZE=64",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "BIGTIFF=YES",
        "-overwrite",
        habitat_path,
        aligned_path
    ]

    subprocess.run(cmd, check=True)


def convert_habitat_code(code):
    if code == NODATA_VAL:
        return code
    elif str(code).startswith('6'):
        return code
    else:
        return int(str(code)[0])
    

def collect_all_habitat_codes(path, blocksize=1024):
    unique_codes = set()
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        for row in range(0, height, blocksize):
            for col in range(0, width, blocksize):
                window = Window(col, row, min(blocksize, width - col), min(blocksize, height - row))
                data = src.read(1, window=window)
                flat = np.unique(data)
                mapped = map(convert_habitat_code, flat)
                unique_codes.update(mapped)
    return sorted(unique_codes)


def build_code_to_index(codes):
    code_to_index = Dict.empty(
        key_type=types.uint16,
        value_type=types.uint8
    )
    for i, code in enumerate(codes):
        code_to_index[code] = i
    return code_to_index


@njit
def fast_frequency_count(block, code_dict):
    n_codes = len(code_dict)
    n_forest_pixels = block.shape[0]
    counts = np.zeros((n_forest_pixels, n_codes), dtype=np.uint8)
    for i in range(n_forest_pixels):
        for j in range(block.shape[1]):
            code = block[i, j]
            idx = code_dict[code]
            counts[i, idx] += 1
    return counts
    

def collect_habitat_frequencies(array_group, habitat1m_path, forest_mask, code_dict):
    forest_pixel_insert_pos = 0
    filter_function = np.vectorize(convert_habitat_code)
    with rasterio.open(habitat1m_path) as src:
        for row in tqdm(range(0, REF_HEIGHT), desc="Processing rows"):
            window = Window(0, row * 10, REF_WIDTH * 10, 10)
            block = src.read(1, window=window)
            block = block.reshape(10, REF_WIDTH, 10).transpose(1, 0, 2).reshape(REF_WIDTH, 100)

            forest_row = forest_mask[row].astype(bool)
            if not np.any(forest_row):
                continue
            forest_block = block[forest_row, :]
            nr_pixels_added = len(forest_block)

            filtered_block = filter_function(forest_block).astype(np.uint16)

            frequency_count_row = fast_frequency_count(filtered_block, code_dict)
            array_group['habitat'][forest_pixel_insert_pos:forest_pixel_insert_pos + nr_pixels_added, :] = frequency_count_row
            forest_pixel_insert_pos += nr_pixels_added

if __name__ == "__main__":
    habitat_path = f"{DATA_DIR}/habitatmap_v1_1_20241025.tif"
    aligned_path = f"{DATA_DIR}/tmp/habitat_aligned_1m_to_refgrid.tif"

    align_habitat_map(habitat_path, aligned_path)

    habitat_codes = collect_all_habitat_codes(habitat_path)
    # habitat_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 60, 61, 62, 63, 64, 65, 66, 612, 614, 621, 622, 623, 624, 625, 631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 651, 652, 653, 661, 662, 663, 664, 665, 60000, 60001, 65535]
    print("Habitat codes in map:", habitat_codes)
    code_to_index = build_code_to_index(habitat_codes)

    forest_mask = np.load(FOREST_MASK)
    N_forest_pixels = int(np.sum(forest_mask))
    N_habitat_codes = len(habitat_codes)

    group = zarr.open_group(TEMPORAL_DATASET_ZARR, mode='a')
    feat_grp = group.require_group('features')

    feat_grp.create_array(
        name='habitat',
        shape=(N_forest_pixels, N_habitat_codes),
        dtype='uint8',
        fill_value=0,
        chunks=(CHUNK_SIZE, N_habitat_codes),
        overwrite=True
    )

    collect_habitat_frequencies(feat_grp, aligned_path, forest_mask, code_to_index)

    feat_grp['habitat'].attrs['habitat_codes'] = habitat_codes
    feat_grp['habitat'].attrs['nodata'] = 255
