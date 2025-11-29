"""
Visualization utilities for displaying images and comparisons.
"""
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image

def create_comparison_display(image1: np.ndarray, image2: np.ndarray, 
                             titles: Tuple[str, str] = ('Box Filter', 'Gaussian Filter')) -> np.ndarray:
    """
    Create side-by-side comparison display of two images.
    """
    # Ensure both images are 2D for grayscale
    if len(image1.shape) > 2:
        image1 = image1.squeeze()
    if len(image2.shape) > 2:
        image2 = image2.squeeze()
    
    # Ensure both images are the same size and type
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Use smaller dimensions
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    # Resize images to target dimensions
    if (h1, w1) != (target_h, target_w):
        pil1 = Image.fromarray((image1 * 255).astype(np.uint8))
        pil1 = pil1.resize((target_w, target_h), Image.Resampling.LANCZOS)
        image1 = np.array(pil1).astype(np.float64) / 255.0
        
    if (h2, w2) != (target_h, target_w):
        pil2 = Image.fromarray((image2 * 255).astype(np.uint8))
        pil2 = pil2.resize((target_w, target_h), Image.Resampling.LANCZOS)
        image2 = np.array(pil2).astype(np.float64) / 255.0
    
    # Create combined image
    padding = 10
    combined = np.ones((target_h, target_w * 2 + padding), dtype=np.float64)
    combined[:, :target_w] = image1
    combined[:, target_w + padding:] = image2
    
    return combined

def plot_results(original: np.ndarray, smoothed: np.ndarray, 
                gradient: np.ndarray, edges: np.ndarray,
                titles: Tuple[str, str, str, str] = 
                ('Original', 'Smoothed', 'Gradient', 'Edges')) -> plt.Figure:
    """
    Create a matplotlib figure with all processing results.
    
    Args:
        original: Original image
        smoothed: Smoothed image
        gradient: Gradient magnitude
        edges: Binary edges
        titles: Titles for each subplot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    images = [original, smoothed, gradient, edges]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    return fig

