"""Denoising Performance Metrics Module

This module provides standard image quality metrics for evaluating the
output of denoising algorithms such as MarchenkoPasturPCADenoiser.

Each metric is available in two flavours:

    * A **single-image** function that accepts a pair of 2-D arrays and returns
        a scalar float.
    * A **batch** function that accepts a pair of (p, height, width) stacks and
        returns a 1-D array of per-image values, one entry per snapshot.
    * An **average** function that accepts a pair of (p, height, width) stacks
        and returns a single scalar float representing the mean metric across all
        image pairs.

References
----------
- Wang, Z. et al.
    "Image quality assessment: from error visibility to structural similarity".
    IEEE Transactions on Image Processing. 13.4 (2004): 600-612.
- Sage, D.
    "SNR, PSNR, RMSE, MAE".
    Biomedical Image Group, EPFL.
    http://bigwww.epfl.ch/sage/soft/snr/
- D. Sage, M. Unser.
    "Teaching Image-Processing Programming in Java".
    IEEE Signal Processing Magazine, vol. 20, no. 6, pp. 43-52, November 2003.
    http://bigwww.epfl.ch/publications/sage0303.html

"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as _skimage_ssim


#
# Internal helpers
#

def _check_image_pair(ref_img: np.ndarray, test_img: np.ndarray) -> None:
    """Validate that two images are conformant 2-D arrays of the same shape.

    Args:
        ref_img (numpy array): reference image; must be 2-D.
        test_img (numpy array): test (denoised) image; must be 2-D and match
            the shape of ref_img.

    Raises:
        ValueError: if either array is not 2-D, or if the shapes do not match.
    """
    if ref_img.ndim != 2:
        raise ValueError(
            f"ref_img must be a 2-D array, got shape {ref_img.shape}."
        )
    if test_img.ndim != 2:
        raise ValueError(
            f"test_img must be a 2-D array, got shape {test_img.shape}."
        )
    if ref_img.shape != test_img.shape:
        raise ValueError(
            f"ref_img and test_img must have the same shape; "
            f"got {ref_img.shape} and {test_img.shape}."
        )


def _check_image_stack_pair(
    ref_imgs: np.ndarray,
    test_imgs: np.ndarray,
) -> None:
    """Validate that two image stacks are conformant 3-D arrays of the same shape.

    Args:
        ref_imgs (numpy array): reference image stack of shape (p, height, width).
        test_imgs (numpy array): test (denoised) image stack; must match the
            shape of ref_imgs exactly.

    Raises:
        ValueError: if either array is not 3-D, or if the shapes do not match.
    """
    if ref_imgs.ndim != 3:
        raise ValueError(
            f"ref_imgs must be a 3-D array (p, height, width), "
            f"got shape {ref_imgs.shape}."
        )
    if test_imgs.ndim != 3:
        raise ValueError(
            f"test_imgs must be a 3-D array (p, height, width), "
            f"got shape {test_imgs.shape}."
        )
    if ref_imgs.shape != test_imgs.shape:
        raise ValueError(
            f"ref_imgs and test_imgs must have the same shape; "
            f"got {ref_imgs.shape} and {test_imgs.shape}."
        )


def _broadcast_ref_to_stack(ref_imgs: np.ndarray, p: int) -> np.ndarray:
    """Expand a single 2-D reference image into a (p, height, width) stack.

    If ref_imgs is already 3-D it is returned unchanged. If it is 2-D it is
    broadcast (zero-copy) to a (p, height, width) view so that every image
    pair shares the same reference.

    Args:
        ref_imgs (numpy array): either a 2-D (height, width) reference image
            or a 3-D (p, height, width) reference stack.
        p (int): number of test images; used only when ref_imgs is 2-D.

    Returns:
        (numpy array) 3-D array of shape (p, height, width).
    """
    if ref_imgs.ndim == 2:
        return np.broadcast_to(
            ref_imgs[np.newaxis, :, :],
            (p, ref_imgs.shape[0], ref_imgs.shape[1])
        )
    return ref_imgs


#
# Single-image metrics
#

def snr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """Compute the Signal-to-Noise Ratio (SNR) between two 2-D images in dB.

    SNR measures the ratio of signal power (energy of the reference image) to
    noise power (energy of the residual ``ref_img - test_img``):

        SNR = 10 * log10(sum(ref^2) / sum((ref - test)^2))

    Args:
        ref_img (numpy array): 2-D reference (ground-truth) image.
        test_img (numpy array): 2-D test (denoised) image to evaluate.

    Returns:
        (float) SNR value in dB. Higher is better.

    Raises:
        ValueError: if the inputs are not 2-D arrays of the same shape.

    References:
        - Sage, D. "SNR, PSNR, RMSE, MAE".
            http://bigwww.epfl.ch/sage/soft/snr/
    """
    _check_image_pair(ref_img, test_img)
    signal_power = np.sum(ref_img ** 2)
    noise_power = np.sum((ref_img - test_img) ** 2)
    if noise_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def psnr(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    max_pixel_value: float = 255.0,
) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two 2-D images in dB.

    PSNR uses the maximum representable pixel value as the signal peak:

        PSNR = 20 * log10(max_pixel_value / RMSE)

    Args:
        ref_img (numpy array): 2-D reference (ground-truth) image.
        test_img (numpy array): 2-D test (denoised) image to evaluate.
        max_pixel_value (float, default=255.0): maximum possible pixel value in
            the image. Use ``1.0`` for images normalised to [0, 1], or ``255.0``
            for standard 8-bit images.

    Returns:
        (float) PSNR value in dB. Higher is better.

    Raises:
        ValueError: if the inputs are not 2-D arrays of the same shape.

    References:
        - https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _check_image_pair(ref_img, test_img)
    mse_val = mean_squared_error(ref_img.ravel(), test_img.ravel())
    if mse_val == 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_pixel_value / np.sqrt(mse_val)))


def mae(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """Compute the Mean Absolute Error (MAE) between two 2-D images.

    MAE measures the average absolute pixel-level deviation between the
    reference and the test image:

        MAE = mean(|ref - test|)

    Computed via ``sklearn.metrics.mean_absolute_error``.

    Args:
        ref_img (numpy array): 2-D reference (ground-truth) image.
        test_img (numpy array): 2-D test (denoised) image to evaluate.

    Returns:
        (float) MAE value. Lower is better.

    Raises:
        ValueError: if the inputs are not 2-D arrays of the same shape.
    """
    _check_image_pair(ref_img, test_img)
    return float(mean_absolute_error(ref_img.ravel(), test_img.ravel()))


def rmse(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """Compute the Root Mean Squared Error (RMSE) between two 2-D images.

    RMSE penalises larger errors more heavily than MAE through the quadratic
    term and is defined as:

        RMSE = sqrt(mean((ref - test)^2))

    Computed as the square root of ``sklearn.metrics.mean_squared_error``.

    Args:
        ref_img (numpy array): 2-D reference (ground-truth) image.
        test_img (numpy array): 2-D test (denoised) image to evaluate.

    Returns:
        (float) RMSE value. Lower is better.

    Raises:
        ValueError: if the inputs are not 2-D arrays of the same shape.
    """
    _check_image_pair(ref_img, test_img)
    return float(np.sqrt(mean_squared_error(ref_img.ravel(), test_img.ravel())))


def ssim(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    data_range: float = 255.0,
) -> float:
    """Compute the Structural Similarity Index (SSIM) between two 2-D images.

    SSIM is a perceptual metric that quantifies image quality degradation caused
    by denoising or compression. Unlike pixel-level metrics (SNR, PSNR, MAE,
    RMSE), SSIM captures structural information, luminance, and contrast
    similarity, making it a valuable complementary measure for evaluating
    denoising performance.

    SSIM values range from -1 to 1, where 1 indicates perfect structural
    similarity. The metric is computed via ``skimage.metrics.structural_similarity``.

    Args:
        ref_img (numpy array): 2-D reference (ground-truth) image.
        test_img (numpy array): 2-D test (denoised) image to evaluate.
        data_range (float, default=255.0): the dynamic range of the images
            (i.e. ``max_value - min_value``). Use ``255.0`` for standard 8-bit
            images and ``1.0`` for images normalised to [0, 1].

    Returns:
        (float) SSIM value in [-1, 1]. Higher is better.

    Raises:
        ValueError: if the inputs are not 2-D arrays of the same shape.

    References:
        - Wang, Z. et al.
            "Image quality assessment: from error visibility to structural similarity".
            IEEE Transactions on Image Processing. 13.4 (2004): 600-612.
    """
    _check_image_pair(ref_img, test_img)
    return float(_skimage_ssim(ref_img, test_img, data_range=data_range))


#
# Batch metrics (image stacks)
#

def batch_snr(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> np.ndarray:
    """Compute SNR for each pair of images in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image stack of shape (p, height, width).
        test_imgs (numpy array): denoised image stack of shape (p, height, width).

    Returns:
        (numpy array) 1-D array of shape (p,) containing the SNR in dB for
        each image pair. Higher is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    _check_image_stack_pair(ref_imgs, test_imgs)
    return np.array([
        snr(ref_imgs[i], test_imgs[i]) for i in range(ref_imgs.shape[0])
    ])


def batch_psnr(
    ref_imgs: np.ndarray,
    test_imgs: np.ndarray,
    max_pixel_value: float = 255.0,
) -> np.ndarray:
    """Compute PSNR for each pair of images in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image stack of shape (p, height, width).
        test_imgs (numpy array): denoised image stack of shape (p, height, width).
        max_pixel_value (float, default=255.0): maximum possible pixel value.

    Returns:
        (numpy array) 1-D array of shape (p,) containing the PSNR in dB for
        each image pair. Higher is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    _check_image_stack_pair(ref_imgs, test_imgs)
    return np.array([
        psnr(ref_imgs[i], test_imgs[i], max_pixel_value=max_pixel_value)
        for i in range(ref_imgs.shape[0])
    ])


def batch_mae(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> np.ndarray:
    """Compute MAE for each pair of images in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image stack of shape (p, height, width).
        test_imgs (numpy array): denoised image stack of shape (p, height, width).

    Returns:
        (numpy array) 1-D array of shape (p,) containing the MAE for each
        image pair. Lower is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    _check_image_stack_pair(ref_imgs, test_imgs)
    return np.array([
        mae(ref_imgs[i], test_imgs[i]) for i in range(ref_imgs.shape[0])
    ])


def batch_rmse(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> np.ndarray:
    """Compute RMSE for each pair of images in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image stack of shape (p, height, width).
        test_imgs (numpy array): denoised image stack of shape (p, height, width).

    Returns:
        (numpy array) 1-D array of shape (p,) containing the RMSE for each
        image pair. Lower is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    _check_image_stack_pair(ref_imgs, test_imgs)
    return np.array([
        rmse(ref_imgs[i], test_imgs[i]) for i in range(ref_imgs.shape[0])
    ])


def batch_ssim(
    ref_imgs: np.ndarray,
    test_imgs: np.ndarray,
    data_range: float = 255.0,
) -> np.ndarray:
    """Compute SSIM for each pair of images in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image stack of shape (p, height, width).
        test_imgs (numpy array): denoised image stack of shape (p, height, width).
        data_range (float, default=255.0): the dynamic range of the images.

    Returns:
        (numpy array) 1-D array of shape (p,) containing the SSIM in [-1, 1]
        for each image pair. Higher is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    _check_image_stack_pair(ref_imgs, test_imgs)
    return np.array([
        ssim(ref_imgs[i], test_imgs[i], data_range=data_range)
        for i in range(ref_imgs.shape[0])
    ])


#
# Average metrics (mean across an image stack)
#

def average_snr(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> float:
    """Compute the mean SNR across all image pairs in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image of shape (height, width) or
            reference image stack of shape (p, height, width). If 2-D, the
            single image is used as the reference for every test image.
        test_imgs (numpy array): denoised image stack of shape (p, height, width).

    Returns:
        (float) Mean SNR in dB across all p image pairs. Higher is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    ref_imgs = _broadcast_ref_to_stack(np.asarray(ref_imgs), test_imgs.shape[0])
    return float(np.mean(batch_snr(ref_imgs, test_imgs)))


def average_psnr(
    ref_imgs: np.ndarray,
    test_imgs: np.ndarray,
    max_pixel_value: float = 255.0,
) -> float:
    """Compute the mean PSNR across all image pairs in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image of shape (height, width) or
            reference image stack of shape (p, height, width). If 2-D, the
            single image is used as the reference for every test image.
        test_imgs (numpy array): denoised image stack of shape (p, height, width).
        max_pixel_value (float, default=255.0): maximum possible pixel value.

    Returns:
        (float) Mean PSNR in dB across all p image pairs. Higher is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    ref_imgs = _broadcast_ref_to_stack(np.asarray(ref_imgs), test_imgs.shape[0])
    return float(np.mean(batch_psnr(ref_imgs, test_imgs, max_pixel_value=max_pixel_value)))


def average_mae(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> float:
    """Compute the mean MAE across all image pairs in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image of shape (height, width) or
            reference image stack of shape (p, height, width). If 2-D, the
            single image is used as the reference for every test image.
        test_imgs (numpy array): denoised image stack of shape (p, height, width).

    Returns:
        (float) Mean MAE across all p image pairs. Lower is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    ref_imgs = _broadcast_ref_to_stack(np.asarray(ref_imgs), test_imgs.shape[0])
    return float(np.mean(batch_mae(ref_imgs, test_imgs)))


def average_rmse(ref_imgs: np.ndarray, test_imgs: np.ndarray) -> float:
    """Compute the mean RMSE across all image pairs in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image of shape (height, width) or
            reference image stack of shape (p, height, width). If 2-D, the
            single image is used as the reference for every test image.
        test_imgs (numpy array): denoised image stack of shape (p, height, width).

    Returns:
        (float) Mean RMSE across all p image pairs. Lower is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    ref_imgs = _broadcast_ref_to_stack(np.asarray(ref_imgs), test_imgs.shape[0])
    return float(np.mean(batch_rmse(ref_imgs, test_imgs)))


def average_ssim(
    ref_imgs: np.ndarray,
    test_imgs: np.ndarray,
    data_range: float = 255.0,
) -> float:
    """Compute the mean SSIM across all image pairs in two aligned stacks.

    Args:
        ref_imgs (numpy array): reference image of shape (height, width) or
            reference image stack of shape (p, height, width). If 2-D, the
            single image is used as the reference for every test image.
        test_imgs (numpy array): denoised image stack of shape (p, height, width).
        data_range (float, default=255.0): the dynamic range of the images.

    Returns:
        (float) Mean SSIM in [-1, 1] across all p image pairs. Higher is better.

    Raises:
        ValueError: if the stacks are not 3-D arrays of the same shape.
    """
    ref_imgs = _broadcast_ref_to_stack(np.asarray(ref_imgs), test_imgs.shape[0])
    return float(np.mean(batch_ssim(ref_imgs, test_imgs, data_range=data_range)))
