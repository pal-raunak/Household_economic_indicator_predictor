import os
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform

from utils import io as custom_io
from utils.constants import OUTPUTS_DIR

def crop_and_locate(selected_id, df, tiff_file):
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        
        # Get lat/lon in WGS84
        row = df[df['id'] == selected_id].iloc[0]
        lon_wgs84, lat_wgs84 = row['longitude'], row['latitude']
        print(f"Input coordinates (WGS84): {lon_wgs84}, {lat_wgs84}")
        
        # Read image depending on whether uploaded or pre-stored
        if hasattr(tiff_file, "read"):  
            tiff_path = os.path.join(OUTPUTS_DIR, "uploaded_temp.tif")
            with open(tiff_path, "wb") as f:
                f.write(tiff_file.read())
        else:  # Pre-stored TIFF
            tiff_path = custom_io.get_local_map_path(tiff_file)
            
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"TIFF file not found at {tiff_path}")

        # Crop box size - using same size as script1_crop.py
        box_size = 2000

        with rasterio.open(tiff_path) as src:
            print(f"TIFF CRS: {src.crs}")
            print(f"TIFF bounds: {src.bounds}")
            print(f"TIFF dimensions: {src.width}x{src.height}")
            
            # Transform coordinates from WGS84 to the TIFF's CRS
            lon_proj, lat_proj = transform(
                'EPSG:4326', src.crs, [lon_wgs84], [lat_wgs84]
            )
            print(f"Projected coordinates: {lon_proj[0]}, {lat_proj[0]}")
            
            # Get pixel coordinates
            row, col = src.index(lon_proj[0], lat_proj[0])
            print(f"Pixel coordinates: {col}, {row}")
            
            # Get transform matrix for pixel dimensions
            transform_matrix = src.transform
            pixel_width = transform_matrix[0]
            pixel_height = -transform_matrix[4]
            
            # Read and transpose image to match script1_crop.py
            image = src.read()
            image = np.transpose(image, (1, 2, 0))

            if image.shape[2] < 3:
                raise ValueError("Image has less than 3 bands. Cannot process RGB.")

            # Calculate crop boundaries using same logic as script1_crop.py
            half = box_size // 2
            top_left = (max(0, col - half), max(0, row - half))
            bottom_right = (min(image.shape[1] - 1, col + half),
                          min(image.shape[0] - 1, row + half))

            # Create image with bounding box
            image_rgb = image[:, :, :3].copy()
            image_with_box = image_rgb.copy()
            cv2.rectangle(image_with_box, top_left, bottom_right, color=(0, 255, 255), thickness=50)
            
            # Resize the image with box to a more manageable size
            # Calculate resize ratio to maintain aspect ratio
            max_dimension = 4000  # Maximum dimension for the resized image
            height, width = image_with_box.shape[:2]
            ratio = min(max_dimension / width, max_dimension / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize the image
            resized_box_image = cv2.resize(image_with_box, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA)
            
            # Save resized image with bounding box
            box_path = os.path.join(OUTPUTS_DIR, "output_with_box.png")
            cv2.imwrite(box_path, resized_box_image)

            # Create cropped image
            cropped = image_rgb[top_left[1]:bottom_right[1], 
                              top_left[0]:bottom_right[0]]

            if cropped is None or cropped.size == 0:
                raise ValueError("Failed to create cropped image")

            # Save cropped image
            cropped_path = os.path.join(OUTPUTS_DIR, "cropped_house_area.png")
            cv2.imwrite(cropped_path, cropped)

            if not os.path.exists(cropped_path):
                raise ValueError("Failed to save cropped image")

            # Calculate relative pixel location
            relative_row = row - top_left[1]
            relative_col = col - top_left[0]

            return cropped_path, box_path, (relative_col, relative_row)

    except Exception as e:
        print(f"Error in crop_and_locate: {str(e)}")
        raise
