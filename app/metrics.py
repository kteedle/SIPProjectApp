"""
Metrics for comparing edge maps and evaluating performance.
"""
import numpy as np
from typing import Dict, Tuple, Optional
import math
import logging
from scipy.spatial import cKDTree  # For efficient distance calculations

logger = logging.getLogger(__name__)

def compare_edge_maps(edges1: np.ndarray, edges2: np.ndarray, 
                     max_samples: int = 10000) -> Dict[str, float]:
    """
    Compare two binary edge maps and compute various metrics.
    Optimized for large images using vectorized operations and sampling.
    
    Args:
        edges1: First binary edge map (reference)
        edges2: Second binary edge map (to compare)
        max_samples: Maximum number of samples for Hausdorff distance
        
    Returns:
        Dictionary of comparison metrics
    """
    # Ensure binary arrays
    edges1_bin = (edges1 > 0).astype(np.uint8)
    edges2_bin = (edges2 > 0).astype(np.uint8)
    
    logger.debug(f"Comparing edge maps: {edges1_bin.shape}, edges1 pixels: {np.sum(edges1_bin)}, edges2 pixels: {np.sum(edges2_bin)}")
    
    # Compute confusion matrix using vectorized operations
    tp = np.sum((edges1_bin == 1) & (edges2_bin == 1))
    fp = np.sum((edges1_bin == 0) & (edges2_bin == 1))
    fn = np.sum((edges1_bin == 1) & (edges2_bin == 0))
    tn = np.sum((edges1_bin == 0) & (edges2_bin == 0))
    
    # Basic agreement metrics
    total_pixels = edges1_bin.size
    agreement = (tp + tn) / total_pixels
    disagreement = (fp + fn) / total_pixels
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Jaccard similarity (IoU)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Hausdorff-like distance (optimized)
    h_distance = approximate_hausdorff_distance(edges1_bin, edges2_bin, max_samples)
    
    logger.debug(f"Comparison results: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    return {
        'agreement': agreement,
        'disagreement': disagreement,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'hausdorff_distance': h_distance,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }

def approximate_hausdorff_distance(edges1: np.ndarray, edges2: np.ndarray, 
                                   max_samples: int = 10000,
                                   max_distance: Optional[float] = None) -> float:
    """
    Compute approximate Hausdorff distance between two edge maps efficiently.
    Uses KD-tree for fast nearest neighbor searches and sampling for large images.
    
    Args:
        edges1: First binary edge map
        edges2: Second binary edge map  
        max_samples: Maximum number of edge pixels to sample
        max_distance: Maximum distance to return if edge maps are empty (default: inf)
        
    Returns:
        Approximate Hausdorff distance
    """
    # Get edge coordinates
    coords1 = np.argwhere(edges1 > 0)
    coords2 = np.argwhere(edges2 > 0)
    
    if len(coords1) == 0 or len(coords2) == 0:
        return max_distance if max_distance is not None else float('inf')
    
    # Sample coordinates if too many (for performance)
    if len(coords1) > max_samples:
        indices = np.random.choice(len(coords1), max_samples, replace=False)
        coords1 = coords1[indices]
    
    if len(coords2) > max_samples:
        indices = np.random.choice(len(coords2), max_samples, replace=False)
        coords2 = coords2[indices]
    
    # Use KD-tree for efficient distance computation
    tree1 = cKDTree(coords1)
    tree2 = cKDTree(coords2)
    
    # Compute directed Hausdorff distances
    dist1, _ = tree2.query(coords1, workers=-1)
    dist2, _ = tree1.query(coords2, workers=-1)
    
    h1 = np.max(dist1)
    h2 = np.max(dist2)
    
    return max(h1, h2)

def compute_gradient_metrics(gradient: np.ndarray, edges: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for gradient quality.
    
    Args:
        gradient: Gradient magnitude image
        edges: Binary edge map
        
    Returns:
        Dictionary of gradient metrics
    """
    edge_pixels = gradient[edges > 0]
    non_edge_pixels = gradient[edges == 0]
    
    metrics = {
        'mean_gradient_at_edges': float(np.mean(edge_pixels)) if len(edge_pixels) > 0 else 0.0,
        'mean_gradient_at_non_edges': float(np.mean(non_edge_pixels)) if len(non_edge_pixels) > 0 else 0.0,
        'std_gradient_at_edges': float(np.std(edge_pixels)) if len(edge_pixels) > 0 else 0.0,
        'std_gradient_at_non_edges': float(np.std(non_edge_pixels)) if len(non_edge_pixels) > 0 else 0.0,
    }
    
    # Edge-to-non-edge ratio (avoid division by zero)
    if len(edge_pixels) > 0 and len(non_edge_pixels) > 0 and np.mean(non_edge_pixels) > 0:
        metrics['edge_non_edge_ratio'] = float(np.mean(edge_pixels) / np.mean(non_edge_pixels))
    else:
        metrics['edge_non_edge_ratio'] = 0.0
    
    return metrics



