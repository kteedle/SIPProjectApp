"""
Core image processing algorithms: smoothing, convolution, Sobel operator, thresholding.
"""
import numpy as np
from typing import Tuple, Optional, Literal
import math

def generate_box_kernel(size: int) -> np.ndarray:
    """
    Generate a box (averaging) kernel of given size.
    
    Args:
        size: Kernel size (odd integer)
        
    Returns:
        Normalized box kernel
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
        
    kernel = np.ones((size, size), dtype=np.float64)
    return kernel / np.sum(kernel)

def generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        size: Kernel size (odd integer)
        sigma: Standard deviation of Gaussian
        
    Returns:
        Normalized Gaussian kernel
    """
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
    """
    Convolve 2D image with a kernel using either spatial domain or FFT.
    """
    if use_fft is None:
        # Auto-select: use FFT for large kernels or large images
        kernel_size = max(kernel.shape)
        image_size = max(image.shape[:2])
        # Use FFT for images larger than 256x256 or kernels larger than 7x7
        use_fft = kernel_size > 7 or image_size > 256
        
    if use_fft:
        return convolve_fft(image, kernel)
    else:
        return convolve_spatial(image, kernel)

def convolve_spatial(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve image with kernel using spatial domain (nested loops).
    
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

def apply_smoothing(image: np.ndarray, smooth_type: str, kernel_size: int, 
                   sigma: Optional[float] = None) -> np.ndarray:
    """
    Apply smoothing filter to image.
    
    Args:
        image: Input image
        smooth_type: 'box' or 'gaussian'
        kernel_size: Size of smoothing kernel
        sigma: Standard deviation for Gaussian (required if smooth_type='gaussian')
        
    Returns:
        Smoothed image
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
        
    # Apply convolution
    smoothed = convolve_2d(image, kernel)
    
    return smoothed

def apply_sobel(image: np.ndarray, mode: str = 'both') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Sobel operator to compute image gradients.
    
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
    
    # Compute gradients based on mode
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
        
    return gradient_x, gradient_y, gradient_mag

def threshold_image(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply threshold to create binary image.
    
    Args:
        image: Input image
        threshold: Threshold value
        
    Returns:
        Binary image (0 or 1)
    """
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image >= threshold] = 1
    return binary