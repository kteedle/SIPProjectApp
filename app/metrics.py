"""
Metrics for comparing edge maps and evaluating performance.
"""
import numpy as np
from typing import Dict, Tuple
import math

def compare_edge_maps(edges1: np.ndarray, edges2: np.ndarray) -> Dict[str, float]:
    """
    Compare two binary edge maps and compute various metrics.
    
    Args:
        edges1: First binary edge map (reference)
        edges2: Second binary edge map (to compare)
        
    Returns:
        Dictionary of comparison metrics
    """
    # Ensure binary arrays
    edges1_bin = (edges1 > 0).astype(np.uint8)
    edges2_bin = (edges2 > 0).astype(np.uint8)
    
    # Compute confusion matrix
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
    
    # Hausdorff-like distance (approximate)
    h_distance = approximate_hausdorff_distance(edges1_bin, edges2_bin)
    
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
                                  max_distance: float = 50.0) -> float:
    """
    Compute approximate Hausdorff distance between two edge maps.
    
    Args:
        edges1: First binary edge map
        edges2: Second binary edge map
        max_distance: Maximum distance to consider
        
    Returns:
        Approximate Hausdorff distance
    """
    # Get edge coordinates
    coords1 = np.argwhere(edges1 > 0)
    coords2 = np.argwhere(edges2 > 0)
    
    if len(coords1) == 0 or len(coords2) == 0:
        return max_distance
        
    # Compute directed Hausdorff distances
    def directed_hausdorff(coords_a, coords_b):
        max_min_dist = 0
        for point_a in coords_a:
            min_dist = float('inf')
            for point_b in coords_b:
                dist = np.linalg.norm(point_a - point_b)
                if dist < min_dist:
                    min_dist = dist
                    if min_dist == 0:  # Found exact match
                        break
            if min_dist > max_min_dist:
                max_min_dist = min_dist
            if max_min_dist > max_distance:  # Early termination
                break
        return max_min_dist
    
    h1 = directed_hausdorff(coords1, coords2)
    h2 = directed_hausdorff(coords2, coords1)
    
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
        'edge_non_edge_ratio': float(np.mean(edge_pixels) / np.mean(non_edge_pixels)) 
        if len(edge_pixels) > 0 and len(non_edge_pixels) > 0 and np.mean(non_edge_pixels) > 0 else 0.0
    }
    
    return metrics