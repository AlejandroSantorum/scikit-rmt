"""Marchenko-Pastur PCA (MP-PCA) Denoiser Module

This module contains the implementation of the Marchenko-Pastur PCA (MP-PCA)
denoiser, a patch-based image denoising method grounded in Random Matrix Theory.
The denoiser leverages the Marchenko-Pastur spectral law (Wishart ensemble, beta=1)
to distinguish true signal singular values from noise in the SVD of local image patches.

"""

from typing import Union

import numpy as np
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skrmt.ensemble import WishartEnsemble
from skrmt.denoise.utils import normalize_imgs_0_255


def _fit_mp_bulk(eigenvals: np.ndarray, gamma: float) -> float:
    """Estimate noise variance by numerically fitting the Marchenko-Pastur bulk median.

    Finds the value of sigma squared such that the observed median of the
    eigenvalues that fall inside the MP bulk (below lambda_plus) matches the
    expected midpoint of the MP support, sigma squared times (1 + gamma).

    Note: this function is intentionally kept as a plain module-level function
    and not a method of MarchenkoPasturPCADenoiser because it is called inside a SciPy
    minimisation loop where constructing a class instance on every iteration
    would be prohibitively expensive. See MarchenkoPasturPCADenoiser._estimate_sigma for
    the higher-level estimator that delegates to this function.

    Args:
        eigenvals (numpy array): 1-D array of sample-covariance eigenvalues,
            i.e. squared singular values of the normalised matrix (1/sqrt(n)) * X.
        gamma (float): aspect ratio p/n.

    Returns:
        (float) Estimated noise variance sigma squared (always positive).
    """
    def loss(x):
        sigma2 = float(x[0])
        lambda_max = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
        bulk = eigenvals[eigenvals <= lambda_max]
        if len(bulk) == 0:
            # All eigenvalues lie above the predicted bulk edge; push sigma squared up.
            return 1e10
        # Expected midpoint of the MP support is sigma squared times (1 + gamma).
        expected_median = sigma2 * (1.0 + gamma)
        return (np.median(bulk) - expected_median) ** 2

    x0 = max(float(np.median(eigenvals)) / (1.0 + gamma), 1e-12)
    result = minimize(loss, x0=[x0], bounds=[(1e-12, None)])
    return float(result.x[0])


#########################################################################
### MP-PCA Denoiser

class MarchenkoPasturPCADenoiser(BaseEstimator, TransformerMixin):
    """Patch-based image denoiser using Marchenko-Pastur PCA (MP-PCA).

    Denoises a stack of p noisy acquisitions of the same scene (e.g. repeated
    MRI scans of the same brain region) by exploiting the redundancy across
    acquisitions. For every window_size x window_size spatial patch, the
    (p, window_size squared) data matrix is denoised via economy SVD with hard
    thresholding of singular values against the Marchenko-Pastur (MP) upper
    spectral edge lambda_plus = sigma^2 * (1 + sqrt(gamma))^2,
    where gamma = p / window_size^2.

    Singular values whose square lies in the MP noise bulk (at or below
    lambda_plus) are zeroed out; those above are treated as true signal and
    preserved. The upper spectral edge is obtained from WishartEnsemble (beta=1),
    which implements the validated Marchenko-Pastur formula. Overlapping patches
    are averaged at the end to eliminate block artefacts.

    This class follows the scikit-learn BaseEstimator / TransformerMixin
    interface, making it composable in Pipeline objects and compatible with
    GridSearchCV for hyperparameter tuning.

    Attributes:
        sigma (float or None): noise standard deviation supplied at construction
            time, or None if it is to be estimated from the data during fit.
        sigma_estimator (str): method used to estimate sigma when sigma is None.
            One of "median", "bulk_mean", "min_eigen", or "mp_fit".
        window_size (int): side length in pixels of each square patch.
        sigma_ (float): noise standard deviation used for denoising; either the
            value passed as sigma or the estimate learned during fit.
        lambda_plus_ (float): Marchenko-Pastur upper spectral edge computed from
            sigma_ and the aspect ratio of the full flattened stack. Informational
            only; per-patch thresholds are recomputed in transform using the actual
            patch dimensions via WishartEnsemble.
        n_snapshots_ (int): number of images p seen during fit.
        image_shape_ (tuple of int): spatial dimensions (height, width) of the
            images seen during fit.

    References:
        - Marchenko, V.A. and Pastur, L.A.
            "Distribution of eigenvalues for some sets of random matrices".
            Mathematics of the USSR-Sbornik. 1.4 (1967): 457-483.
        - Veraart, J. et al.
            "Denoising of diffusion MRI using random matrix theory".
            NeuroImage. 142 (2016): 394-406.

    """

    def __init__(
        self,
        sigma: float = None,
        sigma_estimator: str = "median",
        window_size: int = 16,
        normalize_output: bool = True,
    ) -> None:
        """Constructor for MarchenkoPasturPCADenoiser class.

        Initializes an instance of this class with the given parameters.

        Args:
            sigma (float, default=None): noise standard deviation, assumed common
                to all images and spatial positions. If None, sigma is estimated
                from the eigenvalue bulk of the flattened image stack during fit
                using the method selected by sigma_estimator.
            sigma_estimator (str, default="median"): method used to estimate sigma
                when it is not provided. One of "median" (bulk median divided by
                1 + gamma; fast and robust), "bulk_mean" (mean eigenvalue; unbiased
                but sensitive to signal spikes), "min_eigen" (smallest eigenvalue
                divided by (1 - sqrt(gamma)) squared; accurate for small gamma but
                unstable near gamma = 1), or "mp_fit" (least-squares fit of the MP
                bulk median via SciPy; slowest but most accurate). Ignored when
                sigma is explicitly provided.
            window_size (int, default=16): side length in pixels of each square
                patch. Larger windows improve eigenvalue statistics but reduce
                spatial adaptivity.
            normalize_output (bool, default=True): whether to normalize the
                denoised output images to the 0-255 range. If False, the output
                images will be returned in the same scale as the input; this may be
                desirable if the input images are already in a standard range (e.g. 0-1)
                or if the user wants to preserve the original intensity scale for
                downstream analysis. If True, the output images will be normalized to
                the 0-255 range.

        """
        self.sigma = sigma
        self.sigma_estimator = sigma_estimator
        self.window_size = window_size
        self.normalize_output = normalize_output

    def fit(self, X: np.ndarray, y=None) -> "MarchenkoPasturPCADenoiser":
        """Learn the noise level from the image stack X.

        If sigma was provided at construction time it is stored directly as
        sigma_. Otherwise sigma is estimated from the global eigenvalue
        distribution of the flattened stack using the method chosen by
        sigma_estimator.

        Args:
            X (numpy array): noisy image stack of shape (p, height, width).
            y: ignored; present only for scikit-learn API compatibility.

        Returns:
            (MarchenkoPasturPCADenoiser) self
        """
        X = self._check_input(X)
        p, h, w = X.shape

        # Store metadata about the training data for validation in transform.
        self.n_snapshots_ = p
        self.image_shape_ = (h, w)

        if self.sigma is not None:
            # Caller supplied sigma: use it directly, no estimation needed.
            self.sigma_ = float(self.sigma)
        else:
            # Estimate sigma globally from the full (p, h*w) flattened matrix.
            # Normalise by 1/sqrt(h*w) so squared singular values equal the
            # eigenvalues of the sample covariance - the quantity that the
            # Marchenko-Pastur law describes.
            n_pixels = h * w
            gamma = p / n_pixels
            X_flat = X.reshape(p, n_pixels)
            _, S, _ = np.linalg.svd(X_flat / np.sqrt(n_pixels), full_matrices=False)
            eigenvals = S ** 2
            self.sigma_ = self._estimate_sigma(eigenvals, gamma, self.sigma_estimator)

        # Compute and expose the global MP upper edge via WishartEnsemble
        # (beta=1, real-valued entries). This is informational; per-patch
        # thresholds are recomputed with the correct patch gamma in transform.
        wre = WishartEnsemble(beta=1, p=p, n=h * w, sigma=self.sigma_)
        self.lambda_plus_ = wre.lambda_plus

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Denoise the image stack using patch-based MP-PCA.

        Applies sliding-window SVD hard-thresholding with the noise level
        sigma_ learned during fit. Every window_size x window_size patch is
        denoised independently; overlapping patches are averaged to suppress
        block artefacts and reduce residual variance.
        If normalize_output is True, the final denoised images are normalized to the 0-255 range.

        Args:
            X (numpy array): noisy image stack of shape (p, height, width).
                The spatial dimensions must match those seen during fit.

        Returns:
            (numpy array) Denoised image stack with the same shape and dtype as X.

        Raises:
            sklearn.exceptions.NotFittedError: if called before fit.
            ValueError: if the spatial dimensions of X differ from those seen
                during fit, or if window_size exceeds the image dimensions.
        """
        check_is_fitted(self)
        X = self._check_input(X, expected_shape=self.image_shape_)
        X_denoised = self._sliding_window_denoise(X)
        if self.normalize_output:
            X_denoised = normalize_imgs_0_255(X_denoised)
        return X_denoised

    # fit_transform is inherited from TransformerMixin: it calls fit(X) then
    # transform(X), which is correct and avoids redundant computation.

    @staticmethod
    def _estimate_sigma(eigenvals: np.ndarray, gamma: float, method: str) -> float:
        """Estimate the noise standard deviation from the empirical eigenvalue spectrum.

        Uses the Marchenko-Pastur (MP) law as a reference: for a pure-noise (p x n)
        matrix with i.i.d. N(0, sigma squared) entries, the sample-covariance
        eigenvalues are entirely supported on
        [lambda_minus, lambda_plus] = [sigma^2*(1-sqrt(gamma))^2, sigma^2*(1+sqrt(gamma))^2].
        Each estimator below exploits a different property of this support.

        Args:
            eigenvals (numpy array): 1-D array of sample-covariance eigenvalues, i.e.
                the squared singular values of the normalised matrix (1/sqrt(n)) * X.
            gamma (float): aspect ratio p/n.
            method (str): estimation strategy. One of:

                ``"median"``
                    The midpoint of the MP support is sigma squared times (1 + gamma).
                    Using the empirical median as a proxy for that midpoint gives
                    sigma squared approximately equal to median(eigenvals) / (1 + gamma).
                    Fast, robust, and works well in the typical regime gamma < 1.

                ``"bulk_mean"``
                    The mean eigenvalue of the MP distribution equals sigma squared
                    (the population variance), so sigma squared is approximately
                    mean(eigenvals). Simple but sensitive to outlier spikes above the bulk.

                ``"min_eigen"``
                    The lower MP edge is lambda_minus = sigma^2 * (1 - sqrt(gamma))^2,
                    so sigma squared is approximately min(eigenvals) / (1 - sqrt(gamma))^2.
                    Accurate when the smallest eigenvalue sits close to lambda_minus,
                    but numerically unstable when gamma is close to 1 (lower edge
                    approaches 0); falls back to ``"bulk_mean"`` in that case.

                ``"mp_fit"``
                    Numerically minimises the discrepancy between the observed bulk
                    median and its theoretical MP prediction (uses SciPy).
                    Most accurate but slowest; suitable when patches are large.

        Returns:
            (float) Estimated noise standard deviation sigma (strictly positive).

        Raises:
            ValueError: if method is not one of the four supported options.
        """
        if method == "median":
            # Bulk median ~ sigma^2 * (1 + gamma)  ->  sigma^2 ~ median / (1 + gamma)
            sigma2 = float(np.median(eigenvals)) / (1.0 + gamma)

        elif method == "bulk_mean":
            # E[lambda] = sigma^2 under the MP law
            sigma2 = float(np.mean(eigenvals))

        elif method == "min_eigen":
            # Lower MP edge: lambda_minus = sigma^2 * (1 - sqrt(gamma))^2
            denom = (1.0 - np.sqrt(gamma)) ** 2
            if denom < 1e-10:
                # gamma ~ 1: lower edge collapses to 0, estimator is ill-defined;
                # fall back to the mean estimator which remains well-behaved.
                sigma2 = float(np.mean(eigenvals))
            else:
                sigma2 = float(np.min(eigenvals)) / denom

        elif method == "mp_fit":
            sigma2 = _fit_mp_bulk(eigenvals, gamma)

        else:
            raise ValueError(
                f"Unknown sigma_estimator: '{method}'. "
                "Valid options are: 'median', 'bulk_mean', 'min_eigen', 'mp_fit'."
            )

        # Guard against non-positive values caused by degenerate inputs.
        return float(np.sqrt(max(sigma2, 1e-12)))

    @staticmethod
    def _check_input(
        X: np.ndarray,
        expected_shape: Union[tuple, None] = None,
    ) -> np.ndarray:
        """Validate and coerce the image stack array.

        Args:
            X (numpy array): input array; must be 3-D of shape (p, height, width).
            expected_shape (tuple, default=None): if provided, the (height, width)
                that X must match.

        Returns:
            (numpy array) X cast to float64.

        Raises:
            ValueError: on wrong number of dimensions or mismatched spatial shape.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(
                f"X must be a 3-D array (p, height, width), got shape {X.shape}."
            )
        if expected_shape is not None:
            _, h, w = X.shape
            if (h, w) != expected_shape:
                raise ValueError(
                    f"Spatial dimensions of X ({h}x{w}) do not match the "
                    f"dimensions seen during fit "
                    f"({expected_shape[0]}x{expected_shape[1]})."
                )
        return X

    def _denoise_patch(self, patch: np.ndarray) -> np.ndarray:
        """Denoise a single (p, n_pixels) patch matrix.

        Normalises the patch so that squared singular values equal sample-
        covariance eigenvalues, uses WishartEnsemble (beta=1) to obtain the
        validated MP upper spectral edge for the patch's own aspect ratio
        gamma = p / n_pixels, zeros out noise singular values, and reconstructs
        the denoised patch at the original (un-normalised) scale.

        Args:
            patch (numpy array): (p, n_pixels) float array for one spatial window.

        Returns:
            (numpy array) Denoised (p, n_pixels) array.
        """
        p, n = patch.shape

        # Normalise: squared singular values become sample-covariance eigenvalues.
        U, S, Vh = np.linalg.svd(patch / np.sqrt(n), full_matrices=False)

        # Use WishartEnsemble (beta=1, real entries) to compute the validated
        # MP upper edge for this patch's specific aspect ratio gamma = p/n.
        # lambda_plus = sigma_^2 * (1 + sqrt(gamma))^2
        wre = WishartEnsemble(beta=1, p=p, n=n, sigma=self.sigma_)

        # Hard threshold: zero out singular values inside the noise bulk.
        denoised_S = np.where(S <= np.sqrt(wre.lambda_plus), 0.0, S)

        # Rescale back to the original (non-normalised) domain and reconstruct.
        return np.sqrt(n) * (U * denoised_S) @ Vh

    def _sliding_window_denoise(self, X: np.ndarray) -> np.ndarray:
        """Apply the sliding-window MP-PCA denoising loop.

        Iterates over all window_size x window_size overlapping patches,
        denoises each one via _denoise_patch, and accumulates contributions
        into a running sum. Each pixel is finally divided by the number of
        patches that covered it (the overlap-average step).

        Args:
            X (numpy array): validated float array of shape (p, height, width).

        Returns:
            (numpy array) Denoised array with the same shape and dtype as X.

        Raises:
            ValueError: if window_size exceeds either spatial dimension.
        """
        p, img_height, img_width = X.shape
        ws = self.window_size

        if ws > img_height or ws > img_width:
            raise ValueError(
                f"window_size ({ws}) exceeds image dimensions "
                f"({img_height}x{img_width})."
            )

        sigma_info = (
            f"sigma = {self.sigma:.6g}" if self.sigma is not None
            else f"sigma_estimator = '{self.sigma_estimator}' -> sigma_ = {self.sigma_:.6g}"
        )
        print(
            f"Denoising {p} snapshots of size {img_height}x{img_width} "
            f"({sigma_info}, window_size = {ws})."
        )

        # Running sum of all denoised patch contributions for each pixel.
        denoised_sum = np.zeros_like(X, dtype=float)
        # Number of patches that covered each spatial pixel (for averaging).
        patch_count = np.zeros((img_height, img_width), dtype=float)

        for i in range(img_height - ws + 1):
            for j in range(img_width - ws + 1):
                # Flatten the spatial window into a (p, ws*ws) matrix.
                patch = X[:, i:i + ws, j:j + ws].reshape(p, ws * ws)
                denoised_patch = self._denoise_patch(patch)

                # Accumulate into the running sum.
                denoised_sum[:, i:i + ws, j:j + ws] += denoised_patch.reshape(
                    p, ws, ws
                )
                # Record that these pixels received one more patch contribution.
                patch_count[i:i + ws, j:j + ws] += 1.0

        # Average: divide each pixel by the number of overlapping patches.
        # patch_count is (height, width); broadcast over the p-axis with newaxis.
        denoised = denoised_sum / patch_count[np.newaxis, :, :]

        return denoised.astype(X.dtype)
