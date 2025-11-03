"""
Image I/O utilities for reading, writing, and handling different bit depths.
Supports both regular images and geospatial raster formats.
"""
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple, List, Dict, Any
import os
import rasterio
from rasterio.windows import Window
import logging

logger = logging.getLogger(__name__)

def load_image(file_path: str, max_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load image and convert to normalized float array.
    Supports both regular images and geospatial formats.
    
    Args:
        file_path: Path to image file
        max_size: Optional maximum (width, height) for downsampling
        
    Returns:
        Image as float array in range [0, 1]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Use rasterio for geospatial formats
    if file_ext in ('.tif', '.tiff', '.jp2', '.img'):
        return load_geospatial_image(file_path, max_size)
    else:
        return load_regular_image(file_path, max_size)

def load_regular_image(file_path: str, max_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load regular image formats using PIL."""
    pil_image = Image.open(file_path)
    
    # Downsample if max_size specified
    if max_size and (pil_image.width > max_size[0] or pil_image.height > max_size[1]):
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Downsampled image to {pil_image.size}")
    
    # Convert to RGB if necessary (remove alpha channel)
    if pil_image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[-1])
        pil_image = background
    elif pil_image.mode == 'P':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB' and pil_image.mode != 'L':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    return normalize_image_array(image_array)

def load_geospatial_image(file_path: str, max_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load geospatial image formats using rasterio."""
    try:
        with rasterio.open(file_path) as src:
            # Get image properties
            count = src.count
            height, width = src.height, src.width
            dtype = src.dtypes[0]
            
            logger.info(f"Loading geospatial image: {width}x{height}, {count} bands, dtype: {dtype}")
            
            # Calculate downsampling factor if max_size specified
            scale = 1.0
            if max_size:
                scale_x = max_size[0] / width
                scale_y = max_size[1] / height
                scale = min(scale_x, scale_y, 1.0)  # Don't upsample
                
                if scale < 1.0:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    logger.info(f"Downsampling geospatial image to {new_width}x{new_height}")
                    
                    # Read resampled data
                    data = src.read(
                        out_shape=(count, new_height, new_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                else:
                    data = src.read()
            else:
                data = src.read()
            
            # Handle NoData values
            if src.nodata is not None:
                data = data.astype(np.float64)
                for i in range(count):
                    band_data = data[i]
                    nodata_mask = band_data == src.nodata
                    if np.any(nodata_mask):
                        # Replace NoData with band mean
                        band_mean = np.mean(band_data[~nodata_mask])
                        band_data[nodata_mask] = band_mean
            
            # Transpose to (height, width, bands) format
            if count > 1:
                image_array = np.transpose(data, (1, 2, 0))
            else:
                image_array = data[0]  # Single band
                
            return normalize_image_array(image_array)
            
    except Exception as e:
        logger.error(f"Failed to load geospatial image {file_path}: {str(e)}")
        # Fall back to PIL
        logger.info("Falling back to PIL loading")
        return load_regular_image(file_path, max_size)

def normalize_image_array(image_array: np.ndarray) -> np.ndarray:
    """Normalize image array to [0, 1] float range."""
    # Handle different data types and normalize to [0, 1]
    if image_array.dtype == np.uint8:
        normalized = image_array.astype(np.float64) / 255.0
    elif image_array.dtype == np.uint16:
        normalized = image_array.astype(np.float64) / 65535.0
    elif image_array.dtype == np.float32:
        normalized = image_array.astype(np.float64)
    elif image_array.dtype == np.int32 or image_array.dtype == np.uint32:
        # Handle 32-bit images
        if np.max(image_array) > 1.0:
            normalized = image_array.astype(np.float64) / np.max(image_array)
        else:
            normalized = image_array.astype(np.float64)
    else:
        # Auto-normalize unknown types
        normalized = image_array.astype(np.float64)
        if np.max(normalized) > 1.0:
            normalized = normalized / np.max(normalized)
    
    # Clip to [0, 1] for safety
    normalized = np.clip(normalized, 0, 1)
    
    return normalized

def save_image(image: np.ndarray, file_path: str, bit_depth: int = 8, 
               compress: bool = True, **kwargs):
    """
    Save image array to file. Supports both regular and geospatial formats.
    
    Args:
        image: Image array (float [0,1] or integer or binary)
        file_path: Output file path
        bit_depth: Output bit depth (8 or 16)
        compress: Use compression for geospatial formats
        **kwargs: Additional parameters for geospatial saving
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Use rasterio for geospatial formats
    if file_ext in ('.tif', '.tiff', '.jp2'):
        save_geospatial_image(image, file_path, bit_depth, compress, **kwargs)
    else:
        save_regular_image(image, file_path, bit_depth)

def save_regular_image(image: np.ndarray, file_path: str, bit_depth: int = 8):
    """Save regular image formats using PIL with robust type handling."""
    try:
        # Make a copy to avoid modifying original
        image_to_save = image.copy()
        
        # Determine the target data type based on bit_depth
        if bit_depth == 8:
            target_dtype = np.uint8
            scale_factor = 255.0
        elif bit_depth == 16:
            target_dtype = np.uint16  
            scale_factor = 65535.0
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Convert to appropriate data type and range
        if image_to_save.dtype == np.float32 or image_to_save.dtype == np.float64:
            # Float image in [0,1] range
            image_normalized = np.clip(image_to_save, 0, 1)
            image_converted = (image_normalized * scale_factor).astype(target_dtype)
        elif image_to_save.dtype != target_dtype:
            # Different data type - normalize and convert
            image_min, image_max = np.min(image_to_save), np.max(image_to_save)
            if image_max > image_min:
                image_normalized = (image_to_save - image_min) / (image_max - image_min)
                image_converted = (image_normalized * scale_factor).astype(target_dtype)
            else:
                image_converted = np.zeros_like(image_to_save, dtype=target_dtype)
        else:
            # Already correct data type
            image_converted = image_to_save
        
        # Handle multi-band vs single-band
        if len(image_converted.shape) == 3 and image_converted.shape[2] in [3, 4]:
            # RGB or RGBA image
            if image_converted.shape[2] == 3:
                mode = 'RGB'
            else:
                mode = 'RGBA'
            
            # PIL limitation: uint16 RGB/RGBA not well supported, convert to uint8 for color
            if image_converted.dtype == np.uint16 and mode in ['RGB', 'RGBA']:
                # Convert 16-bit color to 8-bit for PIL compatibility
                image_converted = (image_converted / 256).astype(np.uint8)
                logger.debug("Converted 16-bit color image to 8-bit for PIL compatibility")
            
            pil_image = Image.fromarray(image_converted, mode)
            
        elif len(image_converted.shape) == 3 and image_converted.shape[2] > 4:
            # More than 4 bands - use first 3 as RGB
            image_rgb = image_converted[:, :, :3]
            if image_rgb.dtype == np.uint16:
                image_rgb = (image_rgb / 256).astype(np.uint8)
            pil_image = Image.fromarray(image_rgb, 'RGB')
            
        else:
            # Single channel image
            if len(image_converted.shape) == 3:
                image_converted = image_converted.squeeze()
            
            if image_converted.dtype == np.uint16:
                # 16-bit grayscale
                mode = 'I;16'
            else:
                # 8-bit grayscale
                mode = 'L'
                
            pil_image = Image.fromarray(image_converted, mode)
        
        # Save the image
        pil_image.save(file_path)
        logger.debug(f"Successfully saved: {file_path} (shape: {image_converted.shape}, dtype: {image_converted.dtype})")
        
    except Exception as e:
        logger.error(f"Error saving {file_path}: {str(e)}")
        logger.error(f"Input - shape: {image.shape}, dtype: {image.dtype}")
        logger.error(f"Converted - shape: {image_converted.shape if 'image_converted' in locals() else 'N/A'}, dtype: {image_converted.dtype if 'image_converted' in locals() else 'N/A'}")
        raise e

def save_geospatial_image(image: np.ndarray, file_path: str, bit_depth: int = 8,
                         compress: bool = False, **kwargs):
    """Save geospatial image formats using rasterio."""
    # Determine output data type
    if bit_depth == 8:
        dtype = 'uint8'
        image_save = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        dtype = 'uint16'
        image_save = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    # Prepare rasterio profile
    profile = {
        'driver': 'GTiff' if file_ext in ('.tif', '.tiff') else 'JP2OpenJPEG',
        'dtype': dtype,
        'count': 1 if len(image_save.shape) == 2 else image_save.shape[2],
        'width': image_save.shape[1],
        'height': image_save.shape[0],
    }
    
    # Add compression if requested
    if compress:
        profile.update({
            'compress': 'LZW',
            'predictor': 2 if bit_depth == 16 else 1,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        })
    
    # Update with any provided kwargs
    profile.update(kwargs)
    
    # Write file
    with rasterio.open(file_path, 'w', **profile) as dst:
        if len(image_save.shape) == 2:
            dst.write(image_save, 1)
        else:
            for i in range(image_save.shape[2]):
                dst.write(image_save[:, :, i], i + 1)

def get_image_bands(file_path: str) -> List[Dict[str, Any]]:
    """
    Get information about available bands in an image.
    
    Args:
        file_path: Path to image file
        
    Returns:
        List of band information dictionaries
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ('.tif', '.tiff', '.jp2', '.img'):
        # Geospatial image - get band descriptions if available
        try:
            with rasterio.open(file_path) as src:
                bands = []
                for i in range(src.count):
                    band_info = {
                        'index': i,
                        'description': src.descriptions[i] if src.descriptions and i < len(src.descriptions) else f'Band {i+1}',
                        'dtype': str(src.dtypes[i]),
                        'nodata': src.nodata
                    }
                    bands.append(band_info)
                return bands
        except:
            # Fallback for non-geospatial TIFFs
            pass
    
    # Regular image - infer from shape
    try:
        image = load_image(file_path)
        bands = []
        if len(image.shape) == 2:
            bands.append({'index': 0, 'description': 'Grayscale', 'dtype': str(image.dtype)})
        else:
            for i in range(image.shape[2]):
                band_names = ['Red', 'Green', 'Blue', 'Alpha', 'Band 5', 'Band 6', 'Band 7', 'Band 8']
                desc = band_names[i] if i < len(band_names) else f'Band {i+1}'
                bands.append({'index': i, 'description': desc, 'dtype': str(image.dtype)})
        return bands
    except:
        return []

def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about image array.
    
    Args:
        image: Image array
        
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
        info['type'] = 'multiband'
    else:
        info['channels'] = 1
        info['type'] = 'grayscale'
        
    return info