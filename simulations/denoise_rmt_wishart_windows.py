import numpy as np
from scipy.optimize import minimize
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    # Package scikit-rmt installed
    from skrmt.ensemble import WishartEnsemble
except ModuleNotFoundError:
    # Running directly from the simulations/ directory
    import sys
    sys.path.append("..")
    from skrmt.ensemble import WishartEnsemble


# ---------------------------------------------------------------------------
# Module-level private helpers – noise variance estimation
#
# These are kept as plain functions rather than methods because they are
# stateless utilities with no dependency on the class, and _fit_mp_bulk is
# called inside a SciPy optimisation loop where constructing a class instance
# on every iteration would be prohibitively expensive.
# ---------------------------------------------------------------------------

def _fit_mp_bulk(eigenvals: np.ndarray, gamma: float) -> float:
    """Estimate sigma² by numerically fitting the Marchenko-Pastur bulk median.

    The minimisation finds the value of sigma² such that the observed median
    of the eigenvalues that fall inside the MP bulk (i.e. below lambda_+)
    matches the expected midpoint of the MP support, sigma²·(1 + gamma).

    Args:
        eigenvals: 1-D array of sample-covariance eigenvalues (S²).
        gamma: aspect ratio p/n.

    Returns:
        Estimated noise variance sigma² (always positive).
    """
    def loss(x):
        sigma2 = float(x[0])
        lambda_max = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
        bulk = eigenvals[eigenvals <= lambda_max]
        if len(bulk) == 0:
            # All eigenvalues above the predicted bulk edge – push sigma² up.
            return 1e10
        # Expected midpoint of the MP support: (lambda_+ + lambda_-) / 2 = sigma²·(1 + gamma)
        expected_median = sigma2 * (1.0 + gamma)
        return (np.median(bulk) - expected_median) ** 2

    x0 = max(float(np.median(eigenvals)) / (1.0 + gamma), 1e-12)
    result = minimize(loss, x0=[x0], bounds=[(1e-12, None)])
    return float(result.x[0])


def _estimate_sigma(eigenvals: np.ndarray, gamma: float, method: str) -> float:
    """Estimate the noise standard deviation sigma from the empirical eigenvalue
    spectrum of the sample covariance matrix, guided by the Marchenko-Pastur law.

    Under the MP law, a pure-noise (p x n) matrix with i.i.d. N(0, sigma²)
    entries produces sample-covariance eigenvalues whose distribution is
    entirely supported on [lambda_-, lambda_+] = [sigma²·(1-√gamma)², sigma²·(1+√gamma)²].
    Each estimator below exploits a different property of this support.

    Args:
        eigenvals: 1-D array of sample-covariance eigenvalues, i.e. the squared
            singular values of the normalised matrix (1/√n)·X.
        gamma: aspect ratio p/n.
        method: estimation strategy. One of:

            ``"median"``
                The midpoint of the MP support is sigma²·(1 + gamma).
                Using the empirical median as a proxy for that midpoint gives
                sigma² ≈ median(eigenvals) / (1 + gamma).
                Fast, robust, and works well in the typical regime gamma < 1.

            ``"bulk_mean"``
                The mean eigenvalue of the MP distribution equals sigma²
                (the population variance), so sigma² ≈ mean(eigenvals).
                Simple but sensitive to outlier spikes above the bulk.

            ``"min_eigen"``
                The lower MP edge is lambda_- = sigma²·(1 - √gamma)², so
                sigma² ≈ min(eigenvals) / (1 - √gamma)².
                Accurate when the smallest eigenvalue sits close to lambda_-,
                but numerically unstable when gamma ≈ 1 (lower edge → 0);
                falls back to ``"bulk_mean"`` in that case.

            ``"mp_fit"``
                Numerically minimises the discrepancy between the observed bulk
                median and its theoretical MP prediction (uses SciPy).
                Most accurate but slowest; suitable when patches are large.

    Returns:
        Estimated noise standard deviation sigma (strictly positive).

    Raises:
        ValueError: if *method* is not one of the four supported options.
    """
    if method == "median":
        # Bulk median ≈ sigma²·(1 + gamma)  →  sigma² ≈ median / (1 + gamma)
        sigma2 = float(np.median(eigenvals)) / (1.0 + gamma)

    elif method == "bulk_mean":
        # E[lambda] = sigma²  under the MP law
        sigma2 = float(np.mean(eigenvals))

    elif method == "min_eigen":
        # Lower MP edge: lambda_- = sigma²·(1 - √gamma)²
        denom = (1.0 - np.sqrt(gamma)) ** 2
        if denom < 1e-10:
            # gamma ≈ 1: lower edge collapses to 0, estimator is ill-defined;
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

    # Guard against non-positive values caused by degenerate inputs
    return float(np.sqrt(max(sigma2, 1e-12)))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MPPCADenoiser(BaseEstimator, TransformerMixin):
    """Patch-based image denoiser using Marchenko-Pastur PCA (MP-PCA).

    Denoises a stack of *p* noisy acquisitions of the same scene (e.g.
    repeated MRI scans of the same brain region) by exploiting the redundancy
    across acquisitions.  For every ``window_size × window_size`` spatial
    patch, the ``(p, window_size²)`` data matrix is denoised via economy SVD
    with hard thresholding of singular values against the Marchenko-Pastur
    (MP) upper spectral edge:
        lambda_+ = sigma²·(1 + √gamma)²,   gamma = p / (window_size²).

    Singular values whose square lies in the MP noise bulk (≤ lambda_+) are
    zeroed out; those above are treated as true signal and preserved.
    Overlapping patches are averaged at the end to eliminate block artefacts.

    This class follows the scikit-learn ``BaseEstimator`` / ``TransformerMixin``
    interface, making it composable in ``Pipeline`` objects and compatible with
    ``GridSearchCV`` for hyperparameter tuning.

    Parameters
    ----------
    sigma : float or None, default=None
        Noise standard deviation, assumed common to all images and spatial
        positions.  If ``None``, *sigma* is estimated from the eigenvalue bulk
        of the (flattened) image stack during ``fit`` using the method selected
        by ``sigma_estimator``.
    sigma_estimator : str, default="median"
        Method used to estimate *sigma* when it is not provided.  One of:

        * ``"median"``    – bulk median / (1 + gamma).  Fast and robust.
        * ``"bulk_mean"`` – mean eigenvalue; unbiased but spike-sensitive.
        * ``"min_eigen"`` – smallest eigenvalue / (1 − √gamma)²; accurate
                            for small gamma, unstable near gamma = 1.
        * ``"mp_fit"``    – least-squares fit of the MP bulk median (SciPy);
                            slowest but most accurate.

        Ignored when *sigma* is explicitly provided.
    window_size : int, default=16
        Side length in pixels of each square patch.  Larger windows improve
        eigenvalue statistics but reduce spatial adaptivity.

    Attributes
    ----------
    sigma_ : float
        Noise standard deviation used for denoising; either the value passed
        as ``sigma`` or the estimate learned during ``fit``.
    lambda_plus_ : float
        Marchenko-Pastur upper spectral edge computed from ``sigma_`` and the
        aspect ratio of the full (flattened) stack.  Informational only;
        per-patch thresholds are recomputed in ``transform`` using the actual
        patch dimensions via ``WishartEnsemble``.
    n_snapshots_ : int
        Number of images *p* seen during ``fit``.
    image_shape_ : tuple of int
        Spatial dimensions ``(height, width)`` of the images seen during ``fit``.

    Examples
    --------
    >>> denoiser = MPPCADenoiser(window_size=16)
    >>> denoised = denoiser.fit_transform(noisy_snapshots)   # (p, H, W)

    >>> # Known noise level – skip estimation:
    >>> denoiser = MPPCADenoiser(sigma=0.1, window_size=8)
    >>> denoiser.fit(train_stack)
    >>> denoised_test = denoiser.transform(test_stack)
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        sigma_estimator: str = "median",
        window_size: int = 16,
    ):
        self.sigma = sigma
        self.sigma_estimator = sigma_estimator
        self.window_size = window_size

    def fit(self, X: np.ndarray, y=None) -> "MPPCADenoiser":
        """Learn the noise level from the image stack *X*.

        If ``sigma`` was provided at construction time it is stored directly as
        ``sigma_``.  Otherwise *sigma* is estimated from the global eigenvalue
        distribution of the flattened stack using the method chosen by
        ``sigma_estimator``.

        Args:
            X: noisy image stack, shape ``(p, height, width)``.
            y: ignored; present only for scikit-learn API compatibility.

        Returns:
            self
        """
        X = self._check_input(X)
        p, h, w = X.shape

        # Store metadata about the training data for validation in transform
        self.n_snapshots_ = p
        self.image_shape_ = (h, w)

        if self.sigma is not None:
            # Caller supplied sigma: use it directly, no estimation needed.
            self.sigma_ = float(self.sigma)
        else:
            # Estimate sigma globally from the full (p, h*w) flattened matrix.
            # Normalise by 1/√(h*w) so squared singular values equal the
            # eigenvalues of the sample covariance — the quantity that the
            # Marchenko-Pastur law describes.
            n_pixels = h * w
            gamma = p / n_pixels
            X_flat = X.reshape(p, n_pixels)
            _, S, _ = np.linalg.svd(X_flat / np.sqrt(n_pixels), full_matrices=False)
            eigenvals = S ** 2
            self.sigma_ = _estimate_sigma(eigenvals, gamma, self.sigma_estimator)

        # Compute and expose the global MP upper edge via WishartEnsemble
        # (beta=1, real-valued entries).  This is informational; per-patch
        # thresholds are recomputed with the correct patch gamma in transform.
        wre = WishartEnsemble(beta=1, p=p, n=h * w, sigma=self.sigma_)
        self.lambda_plus_ = wre.lambda_plus

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Denoise the image stack using patch-based MP-PCA.

        Applies sliding-window SVD hard-thresholding with the noise level
        ``sigma_`` learned during ``fit``.  Every ``window_size × window_size``
        patch is denoised independently; overlapping patches are averaged to
        suppress block artefacts and reduce residual variance.

        Args:
            X: noisy image stack, shape ``(p, height, width)``.  The spatial
                dimensions must match those seen during ``fit``.

        Returns:
            Denoised image stack with the same shape and dtype as *X*.

        Raises:
            sklearn.exceptions.NotFittedError: if called before ``fit``.
            ValueError: if the spatial dimensions of *X* differ from those
                seen during ``fit``, or if ``window_size`` exceeds the image
                dimensions.
        """
        check_is_fitted(self)
        X = self._check_input(X, expected_shape=self.image_shape_)
        return self._sliding_window_denoise(X)

    # fit_transform is inherited from TransformerMixin: it calls fit(X) then
    # transform(X), which is correct and efficient for this estimator.

    @staticmethod
    def _check_input(
        X: np.ndarray,
        expected_shape: Optional[tuple] = None,
    ) -> np.ndarray:
        """Validate and coerce the image stack array.

        Args:
            X: input array; must be 3-D (p, height, width).
            expected_shape: if provided, ``(height, width)`` that X must match.

        Returns:
            X cast to float64.

        Raises:
            ValueError: on wrong number of dimensions or mismatched shape.
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
                    f"Spatial dimensions of X ({h}×{w}) do not match the "
                    f"dimensions seen during fit "
                    f"({expected_shape[0]}×{expected_shape[1]})."
                )
        return X

    def _denoise_patch(self, patch: np.ndarray) -> np.ndarray:
        """Denoise a single ``(p, n_pixels)`` patch matrix.

        Normalises the patch so that squared singular values equal sample-
        covariance eigenvalues, uses ``WishartEnsemble`` (beta=1) to obtain
        the validated MP upper spectral edge for the patch's own aspect ratio
        gamma = p / n_pixels, zeros out noise singular values, and reconstructs
        the denoised patch at the original (un-normalised) scale.

        Args:
            patch: ``(p, n_pixels)`` float array for one spatial window.

        Returns:
            Denoised ``(p, n_pixels)`` array.
        """
        p, n = patch.shape

        # Normalise: squared singular values become sample-covariance eigenvalues
        U, S, Vh = np.linalg.svd(patch / np.sqrt(n), full_matrices=False)

        # Use WishartEnsemble (beta=1, real entries) to compute the validated
        # MP upper edge for this patch's specific aspect ratio gamma = p/n.
        # lambda_+ = sigma_²·(1 + √gamma)²
        wre = WishartEnsemble(beta=1, p=p, n=n, sigma=self.sigma_)

        # Hard threshold: zero out singular values inside the noise bulk
        denoised_S = np.where(S <= np.sqrt(wre.lambda_plus), 0.0, S)

        # Rescale back to the original (non-normalised) domain and reconstruct
        return np.sqrt(n) * (U * denoised_S) @ Vh

    def _sliding_window_denoise(self, X: np.ndarray) -> np.ndarray:
        """Apply the sliding-window MP-PCA denoising loop.

        Iterates over all ``window_size × window_size`` overlapping patches,
        denoises each one via ``_denoise_patch``, and accumulates contributions
        into a running sum.  Each pixel is finally divided by the number of
        patches that covered it (the overlap-average step).

        Args:
            X: validated float array of shape ``(p, height, width)``.

        Returns:
            Denoised array with the same shape and dtype as *X*.

        Raises:
            ValueError: if ``window_size`` exceeds either spatial dimension.
        """
        p, img_height, img_width = X.shape
        ws = self.window_size

        if ws > img_height or ws > img_width:
            raise ValueError(
                f"window_size ({ws}) exceeds image dimensions "
                f"({img_height}×{img_width})."
            )

        sigma_info = (
            f"sigma = {self.sigma:.6g}" if self.sigma is not None
            else f"sigma_estimator = '{self.sigma_estimator}' → sigma_ = {self.sigma_:.6g}"
        )
        print(
            f"Denoising {p} snapshots of size {img_height}×{img_width} "
            f"({sigma_info}, window_size = {ws})."
        )

        # Running sum of all denoised patch contributions for each pixel
        denoised_sum = np.zeros_like(X, dtype=float)
        # Number of patches that covered each spatial pixel (for averaging)
        patch_count = np.zeros((img_height, img_width), dtype=float)

        for i in range(img_height - ws + 1):
            for j in range(img_width - ws + 1):
                # Flatten the spatial window into a (p, ws²) matrix
                patch = X[:, i:i + ws, j:j + ws].reshape(p, ws * ws)
                denoised_patch = self._denoise_patch(patch)

                # Accumulate into the running sum
                denoised_sum[:, i:i + ws, j:j + ws] += denoised_patch.reshape(
                    p, ws, ws
                )
                # Record that these pixels received one more patch contribution
                patch_count[i:i + ws, j:j + ws] += 1.0

        # Average: divide each pixel by the number of overlapping patches.
        # patch_count is (height, width); broadcast over the p-axis with newaxis.
        denoised = denoised_sum / patch_count[np.newaxis, :, :]

        return denoised.astype(X.dtype)
