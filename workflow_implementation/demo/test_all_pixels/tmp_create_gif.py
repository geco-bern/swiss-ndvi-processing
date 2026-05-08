
#  nohup python -u /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/workflow_implementation/demo/test_all_pixels/tmp_create_gif.py > /home/Shared/UniBe-swiss-ndvi/GitHub/swiss-ndvi-processing/logs/tmp_create_gif.log 2>&1 &


import os
import glob
from PIL import Image

OUT_DIR = "./img_tmp"
GIF_DIR = "./gifs"
os.makedirs(GIF_DIR, exist_ok=True)

names =  ["Lowland broadleaf","Highland broadleaf","Lowland evergreen","Highland evergreen","Biscth fire affected","Biscth fire non affected","Drought affected","Vaia storm affected"]

for name in names:
    pixel_dir = os.path.join(OUT_DIR, name)
    
    # Get all images for this pixel, sorted by date
    all_images = sorted(glob.glob(os.path.join(pixel_dir, "*.png")))
    
    # Filter to March–October only
    filtered_images = [
        img for img in all_images
        if any(f"-{month:02d}-" in os.path.basename(img) 
               for month in range(3, 11))  # months 3 to 10
    ]
    
    if not filtered_images:
        print(f"No images found for {name}, skipping.")
        continue
    
    # Load images
    frames = [Image.open(img) for img in filtered_images]
    
    # Save as GIF
    gif_path = os.path.join(GIF_DIR, f"{name}.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,   # milliseconds per frame
        loop=0          # 0 = loop forever
    )
    print(f"Created: {gif_path}")

print("All GIFs done.")