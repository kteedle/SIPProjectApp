import rasterio
from rasterio.merge import merge
import numpy as np

# List of input band files (ensure they’re ordered correctly — e.g., B1, B2, B3, etc.)
band_files = [
    "D:\CSRE MTech IITB\SIP\GrpProject\LC08_L1TP_148047_20180423_20180502_01_T1_B2.TIF",
    "D:\CSRE MTech IITB\SIP\GrpProject\LC08_L1TP_148047_20180423_20180502_01_T1_B4.TIF",
    "D:\CSRE MTech IITB\SIP\GrpProject\LC08_L1TP_148047_20180423_20180502_01_T1_B3.TIF",
    "D:\CSRE MTech IITB\SIP\GrpProject\LC08_L1TP_148047_20180423_20180502_01_T1_B4.TIF",
    "D:\CSRE MTech IITB\SIP\GrpProject\LC08_L1TP_148047_20180423_20180502_01_T1_B5.TIF"
]

# Read metadata from the first band to use for output
with rasterio.open(band_files[0]) as src0:
    meta = src0.meta.copy()

# Update metadata to reflect the number of layers
meta.update(count=len(band_files))

# Read each band and stack into a 3D NumPy array (bands, height, width)
bands_data = []
for band_path in band_files:
    with rasterio.open(band_path) as src:
        bands_data.append(src.read(1))  # Read the first (and only) band

# Convert list to NumPy array
stacked = np.stack(bands_data)

# Save as multiband GeoTIFF
output_path = "./output_multiband.tif"
with rasterio.open(output_path, 'w', **meta) as dst:
    for i in range(stacked.shape[0]):
        dst.write(stacked[i], i + 1)

print(f"✅ Multichannel image saved at: {output_path}")

