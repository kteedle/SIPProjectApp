"""
Command-line interface for batch processing.
"""
import argparse
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any

from app.io_utils import load_image, save_image
from app.processing import apply_smoothing, apply_sobel, threshold_image
from app.metrics import compare_edge_maps

def process_single_image(input_path: str, output_dir: str, 
                        kernel_size: int = 3, sigma: float = 1.0,
                        threshold: float = 0.2, smooth_type: str = 'gaussian',
                        band_mode: str = 'grayscale', sobel_mode: str = 'both',
                        threshold_mode: str = 'relative') -> Dict[str, Any]:
    """
    Process a single image with given parameters.
    
    Args:
        input_path: Input image path
        output_dir: Output directory
        kernel_size: Smoothing kernel size
        sigma: Gaussian sigma
        threshold: Threshold value
        smooth_type: 'box' or 'gaussian'
        band_mode: 'grayscale' or 'all'
        sobel_mode: 'x', 'y', or 'both'
        threshold_mode: 'relative' or 'absolute'
        
    Returns:
        Dictionary with results and metrics
    """
    # Load image
    image = load_image(input_path)
    
    # Handle band selection
    if band_mode == "grayscale" and len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:  # Multi-band, use first channel
            gray = image[:,:,0]
        working_image = gray
    else:
        working_image = image
    
    # Apply smoothing
    smoothed = apply_smoothing(working_image, smooth_type, kernel_size, sigma)
    
    # Apply Sobel
    _, _, gradient_mag = apply_sobel(smoothed, sobel_mode)
    
    # Apply threshold
    if threshold_mode == "relative":
        absolute_threshold = threshold * np.max(gradient_mag)
    else:
        absolute_threshold = threshold
        
    edges = threshold_image(gradient_mag, absolute_threshold)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    save_image(smoothed, f"{output_dir}/{base_name}_smoothed.png")
    save_image(gradient_mag, f"{output_dir}/{base_name}_gradient.png")
    save_image(edges, f"{output_dir}/{base_name}_edges.png")
    
    # Compute comparison with box filter if Gaussian was used
    if smooth_type == 'gaussian':
        box_smoothed = apply_smoothing(working_image, 'box', kernel_size)
        _, _, box_gradient_mag = apply_sobel(box_smoothed, sobel_mode)
        box_edges = threshold_image(box_gradient_mag, absolute_threshold)
        
        comparison_metrics = compare_edge_maps(edges, box_edges)
        
        save_image(box_smoothed, f"{output_dir}/{base_name}_box_smoothed.png")
        save_image(box_gradient_mag, f"{output_dir}/{base_name}_box_gradient.png")
        save_image(box_edges, f"{output_dir}/{base_name}_box_edges.png")
    else:
        comparison_metrics = {}
    
    # Prepare results
    results = {
        'input_file': input_path,
        'output_dir': output_dir,
        'parameters': {
            'kernel_size': kernel_size,
            'sigma': sigma,
            'threshold': threshold,
            'smooth_type': smooth_type,
            'band_mode': band_mode,
            'sobel_mode': sobel_mode,
            'threshold_mode': threshold_mode,
            'absolute_threshold': float(absolute_threshold)
        },
        'comparison_metrics': comparison_metrics
    }
    
    # Save metrics
    with open(f"{output_dir}/{base_name}_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Image Edge Processing CLI')
    
    # Required arguments
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    
    # Processing parameters
    parser.add_argument('--kernel-size', type=int, default=3, 
                       help='Smoothing kernel size (odd, default: 3)')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Gaussian sigma (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Threshold value (default: 0.2)')
    parser.add_argument('--smooth-type', choices=['box', 'gaussian'], 
                       default='gaussian', help='Smoothing type (default: gaussian)')
    parser.add_argument('--band-mode', choices=['grayscale', 'all'],
                       default='grayscale', help='Band processing mode (default: grayscale)')
    parser.add_argument('--sobel-mode', choices=['x', 'y', 'both'],
                       default='both', help='Sobel mode (default: both)')
    parser.add_argument('--threshold-mode', choices=['relative', 'absolute'],
                       default='relative', help='Threshold mode (default: relative)')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.kernel_size % 2 == 0 or args.kernel_size < 3:
        parser.error("Kernel size must be odd and >= 3")
    
    if args.sigma <= 0:
        parser.error("Sigma must be positive")
    
    if args.threshold < 0 or args.threshold > 1:
        parser.error("Threshold must be between 0 and 1")
    
    # Process images
    if args.batch:
        if not os.path.isdir(args.input):
            parser.error("Input must be a directory for batch processing")
            
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'))]
        
        all_results = []
        for image_file in image_files:
            input_path = os.path.join(args.input, image_file)
            output_subdir = os.path.join(args.output, 
                                       os.path.splitext(image_file)[0])
            
            print(f"Processing {image_file}...")
            results = process_single_image(
                input_path, output_subdir,
                args.kernel_size, args.sigma, args.threshold,
                args.smooth_type, args.band_mode, args.sobel_mode, args.threshold_mode
            )
            all_results.append(results)
            
        print(f"Processed {len(image_files)} images")
        
    else:
        # Single image processing
        if not os.path.isfile(args.input):
            parser.error("Input file not found")
            
        print(f"Processing {args.input}...")
        results = process_single_image(
            args.input, args.output,
            args.kernel_size, args.sigma, args.threshold,
            args.smooth_type, args.band_mode, args.sobel_mode, args.threshold_mode
        )
        print(f"Processing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()