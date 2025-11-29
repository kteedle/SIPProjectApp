"""
Unit tests for metrics calculations.
"""
import unittest
import numpy as np
from app.metrics import compare_edge_maps, approximate_hausdorff_distance

class TestMetrics(unittest.TestCase):
    
    def test_compare_edge_maps_identical(self):
        """Test comparison of identical edge maps."""
        edges = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        
        metrics = compare_edge_maps(edges, edges)
        
        self.assertEqual(metrics['agreement'], 1.0)
        self.assertEqual(metrics['disagreement'], 0.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        self.assertEqual(metrics['iou'], 1.0)
        
    def test_compare_edge_maps_different(self):
        """Test comparison of different edge maps."""
        edges1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        edges2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        
        metrics = compare_edge_maps(edges1, edges2)
        
        self.assertEqual(metrics['agreement'], 0.5)  # 2/4 pixels agree
        self.assertEqual(metrics['disagreement'], 0.5)
        self.assertEqual(metrics['precision'], 0.0)  # No true positives
        self.assertEqual(metrics['recall'], 0.0)
        
    def test_compare_edge_maps_partial_overlap(self):
        """Test comparison with partial overlap."""
        edges1 = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        edges2 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
        
        metrics = compare_edge_maps(edges1, edges2)
        
        self.assertEqual(metrics['true_positives'], 1)  # Only (0,0) overlaps
        self.assertEqual(metrics['false_positives'], 1)  # (1,1) in edges2 only
        self.assertEqual(metrics['false_negatives'], 1)  # (0,1) in edges1 only
        
    def test_approximate_hausdorff_distance(self):
        """Test approximate Hausdorff distance calculation."""
        # Single point in each set, distance = sqrt(2)
        edges1 = np.zeros((3, 3), dtype=np.uint8)
        edges2 = np.zeros((3, 3), dtype=np.uint8)
        edges1[0, 0] = 1
        edges2[1, 1] = 1
        
        distance = approximate_hausdorff_distance(edges1, edges2)
        expected = np.sqrt(2)
        self.assertAlmostEqual(distance, expected, places=2)
        
    def test_approximate_hausdorff_distance_empty(self):
        """Test Hausdorff distance with empty edge maps."""
        edges1 = np.zeros((3, 3), dtype=np.uint8)
        edges2 = np.zeros((3, 3), dtype=np.uint8)
        
        distance = approximate_hausdorff_distance(edges1, edges2, max_distance=10.0)
        self.assertEqual(distance, 10.0)  # Should return max_distance
        
    def test_approximate_hausdorff_distance_identical(self):
        """Test Hausdorff distance with identical edge maps."""
        edges = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        
        distance = approximate_hausdorff_distance(edges, edges)
        self.assertEqual(distance, 0.0)

if __name__ == '__main__':
    unittest.main()

