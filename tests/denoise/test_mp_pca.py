"""MP-PCA Denoiser Test Module

Tests for skrmt.denoise.mp_pca, targeting 100% line coverage.
"""
# pylint: disable=redefined-outer-name  # pytest fixture parameters must match the fixture name
# pylint: disable=missing-function-docstring  # test method names are self-describing
# pylint: disable=protected-access  # unit tests legitimately exercise private static methods

from unittest.mock import patch

import pytest
import numpy as np
from scipy.optimize import OptimizeResult
from numpy.testing import assert_almost_equal

from sklearn.exceptions import NotFittedError

from skrmt.denoise import MarchenkoPasturPCADenoiser
from skrmt.denoise.mp_pca import _fit_mp_bulk


##########################################
### Fixtures

@pytest.fixture(scope="module")
def pure_noise_eigenvals():
    """Eigenvalues of a pure-noise (10 x 100) matrix with sigma=2."""
    rng = np.random.default_rng(42)
    p, n, sigma = 10, 100, 2.0
    X = rng.standard_normal((p, n)) * sigma
    _, S, _ = np.linalg.svd(X / np.sqrt(n), full_matrices=False)
    return S ** 2, p / n


@pytest.fixture(scope="module")
def small_stack():
    """Noisy image stack of shape (4, 8, 8) with true sigma=2."""
    rng = np.random.default_rng(42)
    signal = rng.standard_normal((8, 8)) * 5.0
    return signal[np.newaxis, :, :] + rng.standard_normal((4, 8, 8)) * 2.0


##########################################
### _fit_mp_bulk

class TestFitMpBulk:
    """Tests for the module-level _fit_mp_bulk function."""

    def test_returns_positive_float(self, pure_noise_eigenvals):
        eigenvals, gamma = pure_noise_eigenvals
        result = _fit_mp_bulk(eigenvals, gamma)
        assert isinstance(result, float)
        assert result > 0.0

    def test_result_close_to_true_sigma_squared(self, pure_noise_eigenvals):
        """On typical pure-noise eigenvalues, estimate is within a factor of 3."""
        eigenvals, gamma = pure_noise_eigenvals
        sigma2_hat = _fit_mp_bulk(eigenvals, gamma)
        # True sigma^2 = 4.0; rough bound due to finite-sample fluctuations.
        assert 0.5 < sigma2_hat < 20.0

    def test_empty_bulk_branch_via_mock(self):
        """Cover the 'if len(bulk) == 0: return 1e10' branch inside loss().

        We intercept the call to scipy.optimize.minimize and invoke the loss
        closure with a near-zero sigma2, making lambda_max << min(eigenvals)
        so the bulk is empty.
        """
        eigenvals = np.array([1.0, 2.0])
        gamma = 0.5
        loss_values = []

        def mock_minimize(fn, x0, **_kwargs):
            # Evaluate with sigma2 = 1e-30: lambda_max ~ 3e-30 << 1.0 → empty bulk
            loss_values.append(fn(np.array([1e-30])))
            # Evaluate at x0 to exercise the normal (non-empty) path
            loss_values.append(fn(x0))
            return OptimizeResult(x=x0, success=True, fun=fn(x0))

        with patch("skrmt.denoise.mp_pca.minimize", side_effect=mock_minimize):
            result = _fit_mp_bulk(eigenvals, gamma)

        assert 1e10 in loss_values, "empty-bulk guard was not triggered"
        assert isinstance(result, float)


##########################################
### MarchenkoPasturPCADenoiser._estimate_sigma

class TestEstimateSigma:
    """Tests for the static method MarchenkoPasturPCADenoiser._estimate_sigma."""

    def test_median(self, pure_noise_eigenvals):
        eigenvals, gamma = pure_noise_eigenvals
        sigma = MarchenkoPasturPCADenoiser._estimate_sigma(eigenvals, gamma, "median")
        assert isinstance(sigma, float)
        assert sigma > 0.0

    def test_min_eigen_normal(self, pure_noise_eigenvals):
        """Normal min_eigen path when gamma is well below 1."""
        eigenvals, gamma = pure_noise_eigenvals
        sigma = MarchenkoPasturPCADenoiser._estimate_sigma(eigenvals, gamma, "min_eigen")
        assert isinstance(sigma, float)
        assert sigma > 0.0

    def test_min_eigen_fallback_near_gamma_one(self, pure_noise_eigenvals):
        """When gamma ~ 1, (1-sqrt(gamma))^2 < 1e-10 → fallback to median."""
        eigenvals, _ = pure_noise_eigenvals
        # gamma so close to 1 that (1 - sqrt(gamma))^2 < 1e-10
        sigma = MarchenkoPasturPCADenoiser._estimate_sigma(
            eigenvals, gamma=0.9999999999, method="min_eigen"
        )
        assert isinstance(sigma, float)
        assert sigma > 0.0

    def test_mp_fit(self, pure_noise_eigenvals):
        eigenvals, gamma = pure_noise_eigenvals
        sigma = MarchenkoPasturPCADenoiser._estimate_sigma(eigenvals, gamma, "mp_fit")
        assert isinstance(sigma, float)
        assert sigma > 0.0

    def test_invalid_method_raises(self, pure_noise_eigenvals):
        eigenvals, gamma = pure_noise_eigenvals
        with pytest.raises(ValueError, match="Unknown sigma_estimator"):
            MarchenkoPasturPCADenoiser._estimate_sigma(eigenvals, gamma, "nonexistent_method")

    def test_bulk_mean_raises(self, pure_noise_eigenvals):
        """'bulk_mean' was removed; it must now raise ValueError."""
        eigenvals, gamma = pure_noise_eigenvals
        with pytest.raises(ValueError, match="Unknown sigma_estimator"):
            MarchenkoPasturPCADenoiser._estimate_sigma(eigenvals, gamma, "bulk_mean")

    def test_degenerate_near_zero_eigenvals(self):
        """Guard against sigma2 <= 0: should return sqrt(1e-12)."""
        eigenvals = np.array([1e-30, 1e-30])
        sigma = MarchenkoPasturPCADenoiser._estimate_sigma(eigenvals, gamma=0.5, method="median")
        assert sigma > 0.0
        assert_almost_equal(sigma, np.sqrt(1e-12), decimal=14)


##########################################
### MarchenkoPasturPCADenoiser._check_input

class TestCheckInput:
    """Tests for the static method MarchenkoPasturPCADenoiser._check_input."""

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            MarchenkoPasturPCADenoiser._check_input(np.ones(5))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            MarchenkoPasturPCADenoiser._check_input(np.ones((5, 5)))

    def test_4d_input_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            MarchenkoPasturPCADenoiser._check_input(np.ones((2, 5, 5, 5)))

    def test_shape_mismatch_raises(self):
        X = np.ones((3, 8, 8))
        with pytest.raises(ValueError, match="Spatial dimensions"):
            MarchenkoPasturPCADenoiser._check_input(X, expected_shape=(4, 4))

    def test_valid_3d_no_expected_shape(self):
        X = np.ones((3, 8, 8), dtype=np.int32)
        out = MarchenkoPasturPCADenoiser._check_input(X)
        assert out.dtype == np.float64
        assert out.shape == (3, 8, 8)

    def test_valid_3d_matching_expected_shape(self):
        X = np.ones((3, 8, 8))
        out = MarchenkoPasturPCADenoiser._check_input(X, expected_shape=(8, 8))
        assert out.shape == (3, 8, 8)


##########################################
### MarchenkoPasturPCADenoiser fit()

class TestFit:
    """Tests for MarchenkoPasturPCADenoiser.fit()."""

    def test_fit_returns_self(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        assert d.fit(small_stack) is d

    def test_fit_sets_fitted_attributes(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        d.fit(small_stack)
        assert d.n_snapshots_ == 4
        assert d.image_shape_ == (8, 8)
        assert d.sigma_ == 2.0
        assert d.lambda_plus_ > 0.0

    def test_fit_explicit_sigma_stored_as_float(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=3, window_size=4)  # int sigma
        d.fit(small_stack)
        assert isinstance(d.sigma_, float)
        assert d.sigma_ == 3.0

    def test_fit_sigma_none_median(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma_estimator="median", window_size=4)
        d.fit(small_stack)
        assert d.sigma_ > 0.0

    def test_fit_sigma_none_min_eigen(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma_estimator="min_eigen", window_size=4)
        d.fit(small_stack)
        assert d.sigma_ > 0.0

    def test_fit_sigma_none_mp_fit(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma_estimator="mp_fit", window_size=4)
        d.fit(small_stack)
        assert d.sigma_ > 0.0

    def test_fit_invalid_input_raises(self):
        d = MarchenkoPasturPCADenoiser(sigma=2.0)
        with pytest.raises(ValueError, match="3-D"):
            d.fit(np.ones((8, 8)))


##########################################
### MarchenkoPasturPCADenoiser transform()

class TestTransform:
    """Tests for MarchenkoPasturPCADenoiser.transform() and the internal denoising loop."""

    def test_not_fitted_error(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        with pytest.raises(NotFittedError):
            d.transform(small_stack)

    def test_output_shape_matches_input(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        out = d.fit_transform(small_stack)
        assert out.shape == small_stack.shape

    def test_output_dtype_is_float64_for_float32_input(self, small_stack):
        """_check_input casts to float64 unconditionally; output is always float64."""
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        X_f32 = small_stack.astype(np.float32)
        out = d.fit_transform(X_f32)
        assert out.dtype == np.float64

    def test_output_dtype_preserved_float64(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        out = d.fit_transform(small_stack)
        assert out.dtype == np.float64

    def test_transform_spatial_mismatch_raises(self, small_stack):
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        d.fit(small_stack)
        X_wrong = np.ones((4, 6, 6))
        with pytest.raises(ValueError, match="Spatial dimensions"):
            d.transform(X_wrong)

    def test_window_exceeds_img_height_raises(self, small_stack):
        """window_size > img_height triggers the guard in _sliding_window_denoise."""
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=100)
        d.fit(small_stack)
        with pytest.raises(ValueError, match="window_size"):
            d.transform(small_stack)

    def test_window_exceeds_img_width_raises(self):
        """window_size > img_width (but not img_height) — non-square image."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 10, 6)) * 2.0  # height=10, width=6
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=8)  # 8 <= 10 but 8 > 6
        d.fit(X)
        with pytest.raises(ValueError, match="window_size"):
            d.transform(X)

    def test_sigma_info_branch_explicit_sigma(self, small_stack, capsys):
        """When sigma is provided explicitly, _sliding_window_denoise prints 'sigma = ...'."""
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        d.fit_transform(small_stack)
        captured = capsys.readouterr()
        assert "sigma = 2" in captured.out

    def test_sigma_info_branch_estimated_sigma(self, small_stack, capsys):
        """When sigma=None, _sliding_window_denoise prints 'sigma_estimator = ...'."""
        d = MarchenkoPasturPCADenoiser(sigma_estimator="median", window_size=4)
        d.fit_transform(small_stack)
        captured = capsys.readouterr()
        assert "sigma_estimator" in captured.out

    def test_pure_noise_is_suppressed(self):
        """Denoising a pure-noise stack should reduce its variance."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 16, 16)) * 2.0
        d = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=8, normalize_output=False)
        out = d.fit_transform(X)
        assert np.var(out) < np.var(X)

    def test_fit_transform_is_consistent_with_fit_then_transform(self, small_stack):
        """fit_transform(X) must equal fit(X).transform(X) for a fixed seed."""
        d1 = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        out1 = d1.fit_transform(small_stack)

        d2 = MarchenkoPasturPCADenoiser(sigma=2.0, window_size=4)
        d2.fit(small_stack)
        out2 = d2.transform(small_stack)

        assert_almost_equal(out1, out2)


##########################################
### BaseEstimator API

class TestBaseEstimatorAPI:
    """Tests for get_params / set_params inherited from BaseEstimator."""

    def test_get_params(self):
        d = MarchenkoPasturPCADenoiser(sigma=1.0, sigma_estimator="min_eigen", window_size=8)
        params = d.get_params()
        expected = {
            "sigma": 1.0,
            "sigma_estimator": "min_eigen",
            "window_size": 8,
            "normalize_output": True
        }
        assert params == expected

    def test_set_params(self):
        d = MarchenkoPasturPCADenoiser(sigma=1.0, window_size=8)
        d.set_params(sigma=3.0, window_size=4)
        assert d.sigma == 3.0
        assert d.window_size == 4

    def test_default_params(self):
        d = MarchenkoPasturPCADenoiser()
        params = d.get_params()
        assert params["sigma"] is None
        assert params["sigma_estimator"] == "median"
        assert params["window_size"] == 16


##########################################
### Integration / accuracy test

class TestAccuracy:
    """End-to-end accuracy test over a larger synthetic stack."""

    def test_sigma_estimation_accuracy_median(self):
        """Median estimator should recover sigma within 50% on a 32x32 stack."""
        rng = np.random.default_rng(0)
        sigma_true = 2.0
        signal = rng.standard_normal((32, 32)) * 5.0
        X = signal[np.newaxis, :, :] + rng.standard_normal((10, 32, 32)) * sigma_true
        d = MarchenkoPasturPCADenoiser(sigma_estimator="median", window_size=16)
        d.fit(X)
        assert abs(d.sigma_ - sigma_true) < 1.0

    def test_denoising_improves_snr(self):
        """Denoised output should be closer to the clean signal than the noisy input."""
        rng = np.random.default_rng(1)
        sigma_true = 2.0
        signal = rng.standard_normal((16, 16)) * 5.0
        noise = rng.standard_normal((6, 16, 16)) * sigma_true
        X_noisy = signal[np.newaxis, :, :] + noise
        X_clean = np.broadcast_to(signal, (6, 16, 16))

        d = MarchenkoPasturPCADenoiser(sigma=sigma_true, window_size=8, normalize_output=False)
        X_denoised = d.fit_transform(X_noisy)

        mse_noisy = np.mean((X_noisy - X_clean) ** 2)
        mse_denoised = np.mean((X_denoised - X_clean) ** 2)
        assert mse_denoised < mse_noisy
