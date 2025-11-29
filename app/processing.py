"""
Core image processing algorithms: smoothing, convolution, Sobel operator, thresholding.
"""
import numpy as np
from typing import Tuple, Optional, Literal, Union
import math
import logging

logger = logging.getLogger(__name__)

def generate_box_kernel(size: int) -> np.ndarray:

    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
        
    kernel = np.ones((size, size), dtype=np.float64)
    return kernel / np.sum(kernel)

def generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:

    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
        
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    
    # Generate Gaussian values
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            
    return kernel / np.sum(kernel)

def convolve_2d(image: np.ndarray, kernel: np.ndarray, use_fft: bool = None) -> np.ndarray:
    
    if use_fft is None:
        # Auto-select: use FFT for large kernels or large images
        kernel_size = max(kernel.shape)
        image_size = max(image.shape[:2])
        # Use FFT for images larger than 256x256 or kernels larger than 7x7
        use_fft = kernel_size > 7 or image_size > 256
        
    if use_fft:
        logger.debug(f"Using FFT convolution for image {image.shape}, kernel {kernel.shape}")
        return convolve_fft(image, kernel)
    else:
        logger.debug(f"Using spatial convolution for image {image.shape}, kernel {kernel.shape}")
        return convolve_spatial(image, kernel)

def convolve_spatial(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    if len(image.shape) == 3:
        # Multi-channel image
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            result[:,:,channel] = _convolve_2d_single(image[:,:,channel], kernel)
        return result
    else:
        # Single channel image
        return _convolve_2d_single(image, kernel)

def _convolve_2d_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve single-channel image with kernel using spatial domain.
    
    Args:
        image: Single-channel input image
        kernel: Convolution kernel
        
    Returns:
        Convolved image
    """
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad image (using zero-padding)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Initialize output
    result = np.zeros_like(image, dtype=np.float64)
    
    # Perform convolution
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k_h, j:j+k_w]
            result[i, j] = np.sum(region * kernel)
            
    return result

def convolve_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve image with kernel using FFT.
    
    Args:
        image: Input image
        kernel: Convolution kernel
        
    Returns:
        Convolved image
    """
    if len(image.shape) == 3:
        # Multi-channel image
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            result[:,:,channel] = _convolve_fft_single(image[:,:,channel], kernel)
        return result
    else:
        # Single channel image
        return _convolve_fft_single(image, kernel)

def _convolve_fft_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve single-channel image with kernel using FFT.
    
    Args:
        image: Single-channel input image
        kernel: Convolution kernel
        
    Returns:
        Convolved image
    """
    h, w = image.shape
    k_h, k_w = kernel.shape
    
    # Pad kernel to match image size
    kernel_padded = np.zeros_like(image, dtype=np.float64)
    kernel_padded[:k_h, :k_w] = kernel
    
    # Shift kernel to center
    kernel_padded = np.roll(kernel_padded, -k_h//2, axis=0)
    kernel_padded = np.roll(kernel_padded, -k_w//2, axis=1)
    
    # Perform FFT convolution
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_padded)
    
    result_fft = image_fft * kernel_fft
    result = np.real(np.fft.ifft2(result_fft))
    
    return result

def process_all_bands(image: np.ndarray) -> np.ndarray:
    """
    Process all bands as color image.
    For images with >3 bands, this should not be called (handled in GUI).
    For images with <3 bands, create pseudo-color.
    
    Args:
        image: Input image
        
    Returns:
        Color image with 3 bands (RGB)
    """
    if len(image.shape) == 3:
        num_bands = image.shape[2]
        if num_bands >= 3:
            # Use first 3 bands as RGB
            return image[:, :, :3]
        elif num_bands == 2:
            # 2 bands - use first for R and G, second for B
            return np.stack([image[:, :, 0], image[:, :, 1], image[:, :, 0]], axis=2)
        else:  # 1 band
            # Single band - duplicate to create grayscale color
            return np.stack([image[:, :, 0]] * 3, axis=2)
    else:
        # Single channel - convert to pseudo-color
        return np.stack([image] * 3, axis=2)

def process_custom_rgb_bands(image: np.ndarray, red_idx: int, green_idx: int, blue_idx: int) -> np.ndarray:
    """
    Create RGB image from custom band selection.
    Handles images with any number of bands.
    
    Args:
        image: Input image
        red_idx: Index for red channel
        green_idx: Index for green channel  
        blue_idx: Index for blue channel
        
    Returns:
        RGB image with custom band assignment
    """
    if len(image.shape) == 3:
        num_bands = image.shape[2]
        # Ensure indices are within bounds
        red_idx = min(red_idx, num_bands - 1)
        green_idx = min(green_idx, num_bands - 1)
        blue_idx = min(blue_idx, num_bands - 1)
        
        red_band = image[:, :, red_idx]
        green_band = image[:, :, green_idx]
        blue_band = image[:, :, blue_idx]
        return np.stack([red_band, green_band, blue_band], axis=2)
    else:
        # Single channel image - use for all channels
        return np.stack([image] * 3, axis=2)

def process_single_band(image: np.ndarray, band_index: int) -> np.ndarray:
    """
    Extract a single band from multi-band image.
    
    Args:
        image: Input image
        band_index: Index of band to extract
        
    Returns:
        Single band image
    """
    if len(image.shape) == 3:
        num_bands = image.shape[2]
        # Ensure index is within bounds
        band_index = min(band_index, num_bands - 1)
        return image[:, :, band_index]
    else:
        return image

def apply_smoothing(image: np.ndarray, smooth_type: str, kernel_size: int, 
                   sigma: Optional[float] = None) -> np.ndarray:
    """
    Apply smoothing filter to image.
    Preserves multi-band structure.
    
    Args:
        image: Input image
        smooth_type: 'box' or 'gaussian'
        kernel_size: Size of smoothing kernel
        sigma: Standard deviation for Gaussian (required if smooth_type='gaussian')
        
    Returns:
        Smoothed image (same number of bands as input)
    """
    # Generate kernel
    if smooth_type == 'box':
        kernel = generate_box_kernel(kernel_size)
    elif smooth_type == 'gaussian':
        if sigma is None:
            raise ValueError("Sigma required for Gaussian smoothing")
        kernel = generate_gaussian_kernel(kernel_size, sigma)
    else:
        raise ValueError(f"Unknown smoothing type: {smooth_type}")
        
    # Apply convolution - this function handles multi-band images
    smoothed = convolve_2d(image, kernel)
    
    return smoothed

def apply_sobel(image: np.ndarray, mode: str = 'both') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Sobel operator to compute image gradients.
    For multi-band images, compute gradient for each band separately.
    
    Args:
        image: Input image (smoothed)
        mode: 'x', 'y', or 'both'
        
    Returns:
        Tuple of (gradient_x, gradient_y, gradient_magnitude)
    """
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float64)
                        
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)
    
    # Handle multi-band images
    if len(image.shape) == 3:
        num_bands = image.shape[2]
        gradient_x = np.zeros_like(image)
        gradient_y = np.zeros_like(image)
        gradient_mag = np.zeros_like(image)
        
        for channel in range(num_bands):
            gx, gy, gm = _apply_sobel_single(image[:, :, channel], sobel_x, sobel_y, mode)
            gradient_x[:, :, channel] = gx
            gradient_y[:, :, channel] = gy
            gradient_mag[:, :, channel] = gm
            
        return gradient_x, gradient_y, gradient_mag
    else:
        # Single channel image
        return _apply_sobel_single(image, sobel_x, sobel_y, mode)

def _apply_sobel_single(image: np.ndarray, sobel_x: np.ndarray, sobel_y: np.ndarray, 
                       mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Sobel to single channel image."""
    gradient_x = None
    gradient_y = None
    gradient_mag = None
    
    if mode in ['x', 'both']:
        gradient_x = convolve_2d(image, sobel_x)
        
    if mode in ['y', 'both']:
        gradient_y = convolve_2d(image, sobel_y)
        
    if mode == 'both':
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    elif mode == 'x':
        gradient_mag = np.abs(gradient_x)
    else:  # mode == 'y'
        gradient_mag = np.abs(gradient_y)
    
    logger.debug(f"Single channel gradient range: [{np.min(gradient_mag):.6f}, {np.max(gradient_mag):.6f}]")
        
    return gradient_x, gradient_y, gradient_mag

def threshold_image(image: np.ndarray, threshold: float, mode: str = 'relative') -> np.ndarray:
    """
    Apply threshold to create binary image.
    For multi-band images, combine bands first using maximum.
    
    Args:
        image: Input image (gradient magnitude)
        threshold: Threshold value
        mode: 'relative' or 'absolute'
        
    Returns:
        Binary image (0 or 255)
    """
    # Handle multi-band images by taking maximum across bands
    if len(image.shape) == 3:
        image_combined = np.max(image, axis=2)
    else:
        image_combined = image
    
    # Handle threshold mode
    if mode == 'relative':
        image_max = np.max(image_combined)
        absolute_threshold = threshold * image_max
    else:  # absolute
        absolute_threshold = threshold
    
    logger.debug(f"Thresholding: mode={mode}, threshold={threshold}, absolute_threshold={absolute_threshold}")
    logger.debug(f"Image range: [{np.min(image_combined):.6f}, {np.max(image_combined):.6f}]")
    
    # Apply threshold
    binary = np.zeros_like(image_combined, dtype=np.uint8)
    binary[image_combined >= absolute_threshold] = 255
    
    edge_pixels = np.sum(binary > 0)
    logger.debug(f"Edge pixels: {edge_pixels} / {binary.size} ({edge_pixels/binary.size*100:.2f}%)")
    
    return binary

def combine_bands_to_grayscale(image: np.ndarray, band_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert multi-band image to grayscale using specified weights.
    
    Args:
        image: Multi-band image (H, W, C)
        band_weights: Weights for each band. If None, use standard luminance.
        
    Returns:
        Grayscale image (H, W)
    """
    if len(image.shape) != 3:
        return image  # Already grayscale
    
    num_bands = image.shape[2]
    
    if band_weights is None:
        # Standard luminance weights for RGB
        if num_bands >= 3:
            band_weights = np.array([0.299, 0.587, 0.114])
        else:
            band_weights = np.ones(num_bands) / num_bands
    
    # Ensure weights match number of bands
    if len(band_weights) != num_bands:
        band_weights = np.ones(num_bands) / num_bands
    
    # Normalize weights
    band_weights = band_weights / np.sum(band_weights)
    
    # Compute weighted sum
    grayscale = np.zeros_like(image[:,:,0])
    for i in range(num_bands):
        grayscale += image[:,:,i] * band_weights[i]
    
    return grayscale
















# def select_bands(image: np.ndarray, band_indices: list[int]) -> np.ndarray:
#     """
#     Select specific bands from a multi-band image.
    
#     Args:
#         image: Multi-band image (H, W, C)
#         band_indices: List of band indices to select
        
#     Returns:
#         Image with selected bands (H, W, len(band_indices))
#     """
#     if len(image.shape) != 3:
#         return image  # Already grayscale
    
#     return image[:,:,band_indices]

