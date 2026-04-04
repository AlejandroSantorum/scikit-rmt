"""Denoise Utils functions

This sub-module contains several useful functions to run and manage various
simulations related to the denoising of images using RMT-based techniques.
"""

import numpy as np


def norm_img_0_255(img: np.ndarray) -> np.ndarray:
    """Normalizes a 2D image (numpy array) pixel intensities between 0 and 255.

    The value 0 corresponds to the least intense pixel, and 255 to the pixel with
    the highest intensity.

    Args:
        img (ndarray): 2D numpy array representing the image to be normalized.

    Returns:
        ndarray: Normalized image with pixel intensities between 0 and 255.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.float64)
    return 255.0 * (img - min_val) / (max_val - min_val)


def normalize_imgs_0_255(snapshots: np.ndarray) -> np.ndarray:
    """Normalizes a set of 2D images (numpy arrays) pixel intensities between 0 and 255.

    The value 0 corresponds to the least intense pixel, and 255 to the pixel with
    the highest intensity.

    Args:
        snapshots (ndarray): 3D numpy array of shape (num_images, height, width) representing
            the set of images to be normalized.

    Returns:
        ndarray: Normalized images with pixel intensities between 0 and 255, of shape
            (num_images, height, width).
    """
    min_vals = snapshots.min(axis=(1, 2), keepdims=True)
    max_vals = snapshots.max(axis=(1, 2), keepdims=True)
    ranges = max_vals - min_vals
    # Avoid division by zero for flat images
    ranges[ranges == 0] = 1
    return 255.0 * (snapshots - min_vals) / ranges
