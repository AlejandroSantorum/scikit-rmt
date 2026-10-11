import numpy as np
import pytest

from skrmt.denoise.utils import norm_img_0_255, normalize_imgs_0_255


def test_norm_img_0_255_standard():
    """Test standard normalization of a 2D image."""
    img = np.array([[10, 20], [30, 40]], dtype=float)
    normalized = norm_img_0_255(img)
    
    assert normalized.shape == img.shape
    assert np.isclose(np.min(normalized), 0.0)
    assert np.isclose(np.max(normalized), 255.0)
    
    # Expected: min=10, max=40, range=30 -> scaled by 255 / 30
    expected = np.array([[0.0, 85.0], [170.0, 255.0]])
    np.testing.assert_allclose(normalized, expected)


def test_norm_img_0_255_flat():
    """Test normalization when all pixel values are identical (flat image)."""
    img = np.array([[5, 5], [5, 5]], dtype=float)
    normalized = norm_img_0_255(img)
    
    assert normalized.shape == img.shape
    np.testing.assert_array_equal(normalized, np.zeros_like(img, dtype=np.float64))


def test_norm_img_0_255_negative():
    """Test normalization on an image with negative values."""
    img = np.array([[-10, 0], [10, 20]], dtype=float)
    normalized = norm_img_0_255(img)
    
    assert np.isclose(np.min(normalized), 0.0)
    assert np.isclose(np.max(normalized), 255.0)
    
    # Expected: min=-10, max=20, range=30
    expected = np.array([[0.0, 85.0], [170.0, 255.0]])
    np.testing.assert_allclose(normalized, expected)


def test_normalize_imgs_0_255_standard():
    """Test batch normalization of a 3D stack of images."""
    snapshots = np.array([
        [[0, 10], [20, 30]],       # Image 1 (min=0, max=30)
        [[100, 100], [200, 300]]   # Image 2 (min=100, max=300)
    ], dtype=float)
    
    normalized = normalize_imgs_0_255(snapshots)
    
    assert normalized.shape == snapshots.shape
    
    # Check first image normalization
    img1_expected = np.array([[0.0, 85.0], [170.0, 255.0]])
    np.testing.assert_allclose(normalized[0], img1_expected)
    
    # Check second image normalization (min=100, max=300, range=200)
    img2_expected = np.array([[0.0, 0.0], [127.5, 255.0]])
    np.testing.assert_allclose(normalized[1], img2_expected)


def test_normalize_imgs_0_255_with_flat():
    """Test batch normalization when one of the snapshots in the stack is flat."""
    snapshots = np.array([
        [[5, 5], [5, 5]],          # Flat image (min=5, max=5)
        [[0, 10], [20, 30]]        # Normal image
    ], dtype=float)
    
    normalized = normalize_imgs_0_255(snapshots)
    
    assert normalized.shape == snapshots.shape
    
    # Flat image should be safely handled and returned as zeros
    np.testing.assert_array_equal(normalized[0], np.zeros((2, 2)))
    
    # Second image should be normalized normally
    img2_expected = np.array([[0.0, 85.0], [170.0, 255.0]])
    np.testing.assert_allclose(normalized[1], img2_expected)
