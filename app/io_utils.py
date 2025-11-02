"""
Image I/O utilities for reading, writing, and handling different bit depths.
"""
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
import os

def load_image(file_path: str) -> np.ndarray:
    """
    Load image and convert to normalized float array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
        
    pil_image =  Image.open(file_path)
    
    # Convert to RGB if necessary (remove alpha channel)
    if pil_image.mode in ('RGBA', 'LA'):
        # Create white background
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[-1])
        pil_image = background
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
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

def save_image(image: np.ndarray, file_path: str, bit_depth: int = 8):
    """
    Save image array to file.
    
    Args:
        image: Image array (float [0,1] or integer)
        file_path: Output file path
        bit_depth: Output bit depth (8 or 16)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert to appropriate data type
    if image.dtype == np.float32 or image.dtype == np.float64:
        if bit_depth == 8:
            image_save = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == 16:
            image_save = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
    else:
        image_save = image
        
    # Handle single channel vs multi-channel
    if len(image_save.shape) == 3 and image_save.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'L'
        if len(image_save.shape) == 3:
            image_save = image_save.squeeze()
            
    pil_image = Image.fromarray(image_save, mode)
    pil_image.save(file_path)

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
        info['type'] = 'color'
    else:
        info['channels'] = 1
        info['type'] = 'grayscale'
        
    return info