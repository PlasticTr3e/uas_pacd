import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
from pystac_client import Client
import planetary_computer 
from rasterio.warp import transform_bounds


img_dir = "img"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

print("searching for cloud-free imagery of Pulau Pari")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

bbox_pulau_pari = [106.605, -5.868, 106.635, -5.852] 

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox_pulau_pari,
    datetime="2023-01-01/2023-12-31",
    query={"eo:cloud_cover": {"lt": 5}}
)
items = list(search.items())
print(f"Found {len(items)} matching images. Using the clearest one.")
item = items[0]
asset_href = item.assets["B08"].href

with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
    with rasterio.open(asset_href) as src:
        transformed_bbox = transform_bounds("EPSG:4326", src.crs, *bbox_pulau_pari)
        
        window = rasterio.windows.from_bounds(*transformed_bbox, src.transform)
        
        raster_subset = src.read(1, window=window).astype('float32')
        
        print(f"Successfully cropped image with shape: {raster_subset.shape} pixels")
        plt.imsave(os.path.join(img_dir, 'Raw_Pulau_Pari.png'), raster_subset, cmap='gray')
        print("saved visual raw image to 'Raw_Pulau_Pari.png'")

data = rescale_intensity(raster_subset, out_range=(-1, 1))

markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.6] = 1  #water
markers[data > 0.2] = 2   #land

print("running random walker segmentation")
labels = random_walker(data, markers, beta=80, mode='bf')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

ax1.imshow(data, cmap='gray')
ax1.axis('off')
ax1.set_title('Raw Sentinel-2 (NIR Band)')

ax2.imshow(markers, cmap='magma', interpolation='nearest')
ax2.axis('off')
ax2.set_title('Algorithm Seeds')

ax3.imshow(labels, cmap='ocean')
ax3.axis('off')
ax3.set_title('Extracted Island Boundary')

plt.tight_layout()

output_file = os.path.join(img_dir, "Pulau_Pari_Extraction.png")
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"saved as {output_file}")

plt.show()