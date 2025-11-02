"""
Unit tests for image processing functions.
"""
import unittest
import numpy as np
from app.processing import (
    generate_box_kernel, 
    generate_gaussian_kernel,
    convolve_2d,
    apply_smoothing,
    apply_sobel,
    threshold_image
)

class TestProcessing(unittest.TestCase):
    
    def test_generate_box_kernel(self):
        """Test box kernel generation."""
        kernel = generate_box_kernel(3)
        expected = np.ones((3, 3)) / 9.0
        np.testing.assert_array_equal(kernel, expected)
        
        # Test normalization
        self.assertAlmostEqual(np.sum(kernel), 1.0)
        
        # Test invalid size
        with self.assertRaises(ValueError):
            generate_box_kernel(4)
            
    def test_generate_gaussian_kernel(self):
        """Test Gaussian kernel generation."""
        kernel = generate_gaussian_kernel(3, 1.0)
        
        # Should be symmetric
        self.assertTrue(np.allclose(kernel, kernel.T))
        
        # Should be normalized
        self.assertAlmostEqual(np.sum(kernel), 1.0)
        
        # Center should be maximum
        center = kernel[1, 1]
        self.assertGreater(center, kernel[0, 0])
        self.assertGreater(center, kernel[2, 2])
        
    def test_convolve_2d(self):
        """Test 2D convolution."""
        # Test with identity kernel
        image = np.random.rand(10, 10)
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        
        result = convolve_2d(image, kernel, use_fft=False)
        np.testing.assert_array_almost_equal(result, image)
        
        # Test with averaging kernel
        kernel = np.ones((3, 3)) / 9.0
        result = convolve_2d(image, kernel, use_fft=False)
        
        # Result should be smoothed
        self.assertLess(np.std(result), np.std(image))
        
    def test_apply_smoothing(self):
        """Test smoothing application."""
        image = np.random.rand(20, 20)
        
        # Test box smoothing
        box_smoothed = apply_smoothing(image, 'box', 3)
        self.assertEqual(box_smoothed.shape, image.shape)
        
        # Test Gaussian smoothing
        gaussian_smoothed = apply_smoothing(image, 'gaussian', 3, 1.0)
        self.assertEqual(gaussian_smoothed.shape, image.shape)
        
        # Both should smooth the image
        self.assertLess(np.std(box_smoothed), np.std(image))
        self.assertLess(np.std(gaussian_smoothed), np.std(image))
        
    def test_apply_sobel(self):
        """Test Sobel operator."""
        # Create test image with vertical edge
        image = np.zeros((10, 10))
        image[:, 5:] = 1.0
        
        gx, gy, mag = apply_sobel(image, 'both')
        
        # Should detect vertical edge
        self.assertGreater(np.max(np.abs(gx)), 0)
        self.assertEqual(gx.shape, image.shape)
        self.assertEqual(gy.shape, image.shape)
        self.assertEqual(mag.shape, image.shape)
        
        # Test x-only mode
        gx_only, _, mag_x = apply_sobel(image, 'x')
        np.testing.assert_array_equal(gx_only, gx)
        np.testing.assert_array_equal(mag_x, np.abs(gx))
        
    def test_threshold_image(self):
        """Test image thresholding."""
        image = np.array([0.1, 0.5, 0.9])
        binary = threshold_image(image, 0.5)
        expected = np.array([0, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(binary, expected)

if __name__ == '__main__':
    unittest.main()