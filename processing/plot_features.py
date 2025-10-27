import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import zarr
import csv

forest_mask = np.load("/data_2/scratch/sbiegel/processed/forest_mask.npy")
height, width = forest_mask.shape
flat_indices = np.flatnonzero(forest_mask)

root = zarr.open("/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr", mode='r', zarr_format=3)
habitat_codes = root['features']['habitat'].attrs['habitat_codes']

stats = []
feature_list = ['habitat']

for feature_name in feature_list:
    arr = root['features'][feature_name][:]
    if feature_name in ['northing','profile_curv','median_forest_height','plan_curv',
                         'roughness','mean_curv','easting','twi','tri','slope','accum','dem']:
        nodata = -9999
        flat_map = np.full(height*width, nodata, dtype=arr.dtype)
        flat_map[flat_indices] = arr
        map2d = flat_map.reshape(height, width)
        masked = np.ma.masked_equal(map2d, nodata)
        cmap = plt.get_cmap('seismic') if feature_name in ['profile_curv','plan_curv','mean_curv'] else plt.get_cmap('viridis')
        if feature_name == 'median_forest_height':
            cmap = plt.get_cmap('YlGn')
        if feature_name in ['profile_curv','plan_curv','mean_curv']:
            vmax = np.nanmax(np.abs(arr))
            vmin = -vmax
            im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = plt.imshow(masked, cmap=cmap)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        stats.append((feature_name, '', mean_val, std_val))
        cmap.set_bad('black')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(feature_name)
        plt.axis('off')
        plt.gcf().set_size_inches(12,10)
        plt.savefig(f"{feature_name}_map.png", bbox_inches='tight')
        plt.close()

    elif feature_name == 'tree_species':
        flat_map = np.full(height*width, 255, dtype=int)
        flat_map[flat_indices] = arr.astype(int)
        index_map = flat_map.reshape(height, width)
        masked = np.ma.masked_equal(index_map, 255)
        species_codes = np.unique(index_map)
        species_codes = species_codes[species_codes != 255]
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        stats.append((feature_name, '', mean_val, std_val))
        cmap = plt.get_cmap('tab20', len(species_codes))
        cmap.set_bad('black')
        im = plt.imshow(masked, cmap=cmap, vmin=0, vmax=len(species_codes)-1)
        plt.gcf().set_size_inches(12,12)
        plt.legend(handles=[mpatches.Patch(color=cmap(i), label=str(species_codes[i])) 
                            for i in range(len(species_codes))],
                   bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
        plt.title(feature_name)
        plt.axis('off')
        plt.savefig(f"{feature_name}_map.png", bbox_inches='tight')
        plt.close()

    elif feature_name == 'habitat':
        nodata = 255
        counts = arr
        for idx, code in enumerate(habitat_codes):
            flat_map = np.full(height*width, nodata, dtype=np.uint16)
            flat_map[flat_indices] = counts[:, idx]
            map2d = flat_map.reshape(height, width)
            masked = np.ma.masked_equal(map2d, nodata)
            cmap = plt.get_cmap('plasma')
            cmap.set_bad('black')
            im = plt.imshow(masked, cmap=cmap)
            plt.gcf().set_size_inches(12,12)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"habitat_{code}")
            plt.axis('off')
            plt.savefig(f"habitat_{code}_map.png", bbox_inches='tight')
            plt.close()
            mean_val = np.mean(counts[:, idx])
            std_val = np.std(counts[:, idx])
            stats.append((feature_name, code, mean_val, std_val))
            print(f"Habitat code {code}: {np.sum(counts[:, idx])}")

# Export statistics to CSV
with open('feature_stats.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['feature', 'code', 'mean', 'std'])
    for feature, code, mean_val, std_val in stats:
        writer.writerow([feature, code, f"{mean_val:.6f}", f"{std_val:.6f}"])
