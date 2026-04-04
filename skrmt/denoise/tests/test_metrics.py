"""Metrics Test Module

Tests for skrmt.denoise.metrics, targeting 100% line coverage.
"""
# pylint: disable=redefined-outer-name  # pytest fixture parameters must match the fixture name
# pylint: disable=missing-function-docstring  # test method names are self-describing

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from skrmt.denoise.metrics import (
    # helpers
    _check_image_pair,
    _check_image_stack_pair,
    # single-image
    snr,
    psnr,
    mae,
    rmse,
    ssim,
    # batch
    batch_snr,
    batch_psnr,
    batch_mae,
    batch_rmse,
    batch_ssim,
    # average
    average_snr,
    average_psnr,
    average_mae,
    average_rmse,
    average_ssim,
)


##########################################
### Fixtures

@pytest.fixture(scope="module")
def image_pair():
    """A deterministic (32, 32) reference image and a slightly noisy test image."""
    rng = np.random.default_rng(42)
    ref = rng.random((32, 32)) * 255.0
    test = ref + rng.standard_normal((32, 32)) * 5.0
    return ref, test


@pytest.fixture(scope="module")
def perfect_pair(image_pair):
    """A reference image paired with itself (zero error)."""
    ref, _ = image_pair
    return ref, ref.copy()


@pytest.fixture(scope="module")
def image_stack_pair():
    """Deterministic stacks of shape (4, 32, 32)."""
    rng = np.random.default_rng(42)
    refs = rng.random((4, 32, 32)) * 255.0
    tests = refs + rng.standard_normal((4, 32, 32)) * 5.0
    return refs, tests


##########################################
### _check_image_pair

class TestCheckImagePair:
    """Tests for the _check_image_pair helper."""

    def test_1d_ref_raises(self):
        with pytest.raises(ValueError, match="ref_img must be a 2-D array"):
            _check_image_pair(np.ones(5), np.ones((5, 5)))

    def test_1d_test_raises(self):
        with pytest.raises(ValueError, match="test_img must be a 2-D array"):
            _check_image_pair(np.ones((5, 5)), np.ones(5))

    def test_3d_ref_raises(self):
        with pytest.raises(ValueError, match="ref_img must be a 2-D array"):
            _check_image_pair(np.ones((2, 5, 5)), np.ones((5, 5)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same shape"):
            _check_image_pair(np.ones((4, 4)), np.ones((8, 8)))

    def test_valid_pair_passes(self, image_pair):
        ref, test = image_pair
        _check_image_pair(ref, test)  # must not raise


##########################################
### _check_image_stack_pair

class TestCheckImageStackPair:
    """Tests for the _check_image_stack_pair helper."""

    def test_2d_ref_raises(self):
        with pytest.raises(ValueError, match="ref_imgs must be a 3-D array"):
            _check_image_stack_pair(np.ones((4, 4)), np.ones((2, 4, 4)))

    def test_2d_test_raises(self):
        with pytest.raises(ValueError, match="test_imgs must be a 3-D array"):
            _check_image_stack_pair(np.ones((2, 4, 4)), np.ones((4, 4)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same shape"):
            _check_image_stack_pair(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_valid_pair_passes(self, image_stack_pair):
        refs, tests = image_stack_pair
        _check_image_stack_pair(refs, tests)  # must not raise


##########################################
### snr

class TestSNR:
    """Tests for the single-image snr function."""

    def test_returns_float(self, image_pair):
        ref, test = image_pair
        assert isinstance(snr(ref, test), float)

    def test_positive_for_noisy_image(self, image_pair):
        ref, test = image_pair
        assert snr(ref, test) > 0.0

    def test_perfect_reconstruction_returns_inf(self, perfect_pair):
        ref, test = perfect_pair
        assert snr(ref, test) == float("inf")

    def test_higher_noise_lower_snr(self, image_pair):
        ref, _ = image_pair
        rng = np.random.default_rng(0)
        low_noise = ref + rng.standard_normal(ref.shape) * 2.0
        high_noise = ref + rng.standard_normal(ref.shape) * 20.0
        assert snr(ref, low_noise) > snr(ref, high_noise)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            snr(np.ones((4, 4)), np.ones((8, 8)))


##########################################
### psnr

class TestPSNR:
    """Tests for the single-image psnr function."""

    def test_returns_float(self, image_pair):
        ref, test = image_pair
        assert isinstance(psnr(ref, test), float)

    def test_positive_for_noisy_image(self, image_pair):
        ref, test = image_pair
        assert psnr(ref, test) > 0.0

    def test_perfect_reconstruction_returns_inf(self, perfect_pair):
        ref, test = perfect_pair
        assert psnr(ref, test) == float("inf")

    def test_custom_max_pixel_value(self, image_pair):
        """psnr with max_pixel_value=1.0 should differ from the default 255.0."""
        ref, test = image_pair
        p_default = psnr(ref, test, max_pixel_value=255.0)
        p_unit = psnr(ref / 255.0, test / 255.0, max_pixel_value=1.0)
        assert_almost_equal(p_default, p_unit, decimal=10)

    def test_higher_noise_lower_psnr(self, image_pair):
        ref, _ = image_pair
        rng = np.random.default_rng(1)
        low_noise = ref + rng.standard_normal(ref.shape) * 2.0
        high_noise = ref + rng.standard_normal(ref.shape) * 20.0
        assert psnr(ref, low_noise) > psnr(ref, high_noise)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            psnr(np.ones((4, 4)), np.ones((8, 8)))


##########################################
### mae

class TestMAE:
    """Tests for the single-image mae function."""

    def test_returns_float(self, image_pair):
        ref, test = image_pair
        assert isinstance(mae(ref, test), float)

    def test_zero_for_identical_images(self, perfect_pair):
        ref, test = perfect_pair
        assert mae(ref, test) == 0.0

    def test_non_negative(self, image_pair):
        ref, test = image_pair
        assert mae(ref, test) >= 0.0

    def test_higher_noise_higher_mae(self, image_pair):
        ref, _ = image_pair
        rng = np.random.default_rng(2)
        low_noise = ref + rng.standard_normal(ref.shape) * 2.0
        high_noise = ref + rng.standard_normal(ref.shape) * 20.0
        assert mae(ref, low_noise) < mae(ref, high_noise)

    def test_known_value(self):
        """MAE of a constant offset of 3 across all pixels must equal 3."""
        ref = np.zeros((4, 4))
        test = ref + 3.0
        assert_almost_equal(mae(ref, test), 3.0)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            mae(np.ones((4, 4)), np.ones((8, 8)))


##########################################
### rmse

class TestRMSE:
    """Tests for the single-image rmse function."""

    def test_returns_float(self, image_pair):
        ref, test = image_pair
        assert isinstance(rmse(ref, test), float)

    def test_zero_for_identical_images(self, perfect_pair):
        ref, test = perfect_pair
        assert rmse(ref, test) == 0.0

    def test_non_negative(self, image_pair):
        ref, test = image_pair
        assert rmse(ref, test) >= 0.0

    def test_higher_noise_higher_rmse(self, image_pair):
        ref, _ = image_pair
        rng = np.random.default_rng(3)
        low_noise = ref + rng.standard_normal(ref.shape) * 2.0
        high_noise = ref + rng.standard_normal(ref.shape) * 20.0
        assert rmse(ref, low_noise) < rmse(ref, high_noise)

    def test_known_value(self):
        """RMSE of a constant offset of 4 across all pixels must equal 4."""
        ref = np.zeros((4, 4))
        test = ref + 4.0
        assert_almost_equal(rmse(ref, test), 4.0)

    def test_rmse_geq_mae(self, image_pair):
        """By the QM-AM inequality, RMSE >= MAE for any input."""
        ref, test = image_pair
        assert rmse(ref, test) >= mae(ref, test)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            rmse(np.ones((4, 4)), np.ones((8, 8)))


##########################################
### ssim

class TestSSIM:
    """Tests for the single-image ssim function."""

    def test_returns_float(self, image_pair):
        ref, test = image_pair
        assert isinstance(ssim(ref, test), float)

    def test_perfect_reconstruction_returns_one(self, perfect_pair):
        ref, test = perfect_pair
        assert_almost_equal(ssim(ref, test), 1.0, decimal=10)

    def test_in_valid_range(self, image_pair):
        ref, test = image_pair
        val = ssim(ref, test)
        assert -1.0 <= val <= 1.0

    def test_close_to_one_for_small_noise(self, image_pair):
        ref, test = image_pair
        assert ssim(ref, test) > 0.9

    def test_custom_data_range(self, image_pair):
        """ssim with data_range=1.0 on [0,1]-normalised images must match."""
        ref, test = image_pair
        val_255 = ssim(ref, test, data_range=255.0)
        val_1 = ssim(ref / 255.0, test / 255.0, data_range=1.0)
        assert_almost_equal(val_255, val_1, decimal=10)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            ssim(np.ones((4, 4)), np.ones((8, 8)))


##########################################
### batch functions

class TestBatchMetrics:
    """Tests for the batch_* functions."""

    def test_batch_snr_shape(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_snr(refs, tests)
        assert result.shape == (refs.shape[0],)

    def test_batch_psnr_shape(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_psnr(refs, tests)
        assert result.shape == (refs.shape[0],)

    def test_batch_mae_shape(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_mae(refs, tests)
        assert result.shape == (refs.shape[0],)

    def test_batch_rmse_shape(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_rmse(refs, tests)
        assert result.shape == (refs.shape[0],)

    def test_batch_ssim_shape(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_ssim(refs, tests)
        assert result.shape == (refs.shape[0],)

    def test_batch_snr_values_match_single(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_snr(refs, tests)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], snr(refs[i], tests[i]))

    def test_batch_psnr_values_match_single(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_psnr(refs, tests)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], psnr(refs[i], tests[i]))

    def test_batch_mae_values_match_single(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_mae(refs, tests)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], mae(refs[i], tests[i]))

    def test_batch_rmse_values_match_single(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_rmse(refs, tests)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], rmse(refs[i], tests[i]))

    def test_batch_ssim_values_match_single(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_ssim(refs, tests)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], ssim(refs[i], tests[i]))

    def test_batch_snr_invalid_raises(self):
        with pytest.raises(ValueError):
            batch_snr(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_batch_psnr_invalid_raises(self):
        with pytest.raises(ValueError):
            batch_psnr(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_batch_mae_invalid_raises(self):
        with pytest.raises(ValueError):
            batch_mae(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_batch_rmse_invalid_raises(self):
        with pytest.raises(ValueError):
            batch_rmse(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_batch_ssim_invalid_raises(self):
        with pytest.raises(ValueError):
            batch_ssim(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_batch_psnr_custom_max_pixel_value(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_psnr(refs, tests, max_pixel_value=128.0)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], psnr(refs[i], tests[i], max_pixel_value=128.0))

    def test_batch_ssim_custom_data_range(self, image_stack_pair):
        refs, tests = image_stack_pair
        result = batch_ssim(refs, tests, data_range=128.0)
        for i in range(refs.shape[0]):
            assert_almost_equal(result[i], ssim(refs[i], tests[i], data_range=128.0))


##########################################
### average functions

class TestAverageMetrics:
    """Tests for the average_* functions."""

    def test_average_snr_returns_float(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert isinstance(average_snr(refs, tests), float)

    def test_average_psnr_returns_float(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert isinstance(average_psnr(refs, tests), float)

    def test_average_mae_returns_float(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert isinstance(average_mae(refs, tests), float)

    def test_average_rmse_returns_float(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert isinstance(average_rmse(refs, tests), float)

    def test_average_ssim_returns_float(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert isinstance(average_ssim(refs, tests), float)

    def test_average_snr_equals_mean_of_batch(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert_almost_equal(average_snr(refs, tests), float(np.mean(batch_snr(refs, tests))))

    def test_average_psnr_equals_mean_of_batch(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert_almost_equal(average_psnr(refs, tests), float(np.mean(batch_psnr(refs, tests))))

    def test_average_mae_equals_mean_of_batch(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert_almost_equal(average_mae(refs, tests), float(np.mean(batch_mae(refs, tests))))

    def test_average_rmse_equals_mean_of_batch(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert_almost_equal(average_rmse(refs, tests), float(np.mean(batch_rmse(refs, tests))))

    def test_average_ssim_equals_mean_of_batch(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert_almost_equal(average_ssim(refs, tests), float(np.mean(batch_ssim(refs, tests))))

    def test_average_snr_perfect_stack_returns_inf(self):
        """A perfect (zero-error) stack yields average SNR of inf."""
        refs = np.random.default_rng(0).random((3, 16, 16)) * 255.0
        assert average_snr(refs, refs.copy()) == float("inf")

    def test_average_mae_zero_for_identical_stack(self):
        refs = np.random.default_rng(0).random((3, 16, 16)) * 255.0
        assert average_mae(refs, refs.copy()) == 0.0

    def test_average_rmse_zero_for_identical_stack(self):
        refs = np.random.default_rng(0).random((3, 16, 16)) * 255.0
        assert average_rmse(refs, refs.copy()) == 0.0

    def test_average_ssim_one_for_identical_stack(self):
        refs = np.random.default_rng(0).random((3, 16, 16)) * 255.0
        assert_almost_equal(average_ssim(refs, refs.copy()), 1.0, decimal=10)

    def test_average_psnr_custom_max_pixel_value(self, image_stack_pair):
        refs, tests = image_stack_pair
        expected = float(np.mean(batch_psnr(refs, tests, max_pixel_value=128.0)))
        assert_almost_equal(average_psnr(refs, tests, max_pixel_value=128.0), expected)

    def test_average_ssim_custom_data_range(self, image_stack_pair):
        refs, tests = image_stack_pair
        expected = float(np.mean(batch_ssim(refs, tests, data_range=128.0)))
        assert_almost_equal(average_ssim(refs, tests, data_range=128.0), expected)

    def test_average_snr_invalid_raises(self):
        with pytest.raises(ValueError):
            average_snr(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    def test_average_mae_invalid_raises(self):
        with pytest.raises(ValueError):
            average_mae(np.ones((2, 4, 4)), np.ones((2, 8, 8)))

    #
    # 2-D reference image broadcasting
    #

    def test_average_snr_2d_ref_matches_broadcast_stack(self, image_stack_pair):
        """A 2-D ref must produce the same result as a manually broadcast stack."""
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(average_snr(ref_2d, tests), average_snr(ref_3d, tests))

    def test_average_psnr_2d_ref_matches_broadcast_stack(self, image_stack_pair):
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(average_psnr(ref_2d, tests), average_psnr(ref_3d, tests))

    def test_average_mae_2d_ref_matches_broadcast_stack(self, image_stack_pair):
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(average_mae(ref_2d, tests), average_mae(ref_3d, tests))

    def test_average_rmse_2d_ref_matches_broadcast_stack(self, image_stack_pair):
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(average_rmse(ref_2d, tests), average_rmse(ref_3d, tests))

    def test_average_ssim_2d_ref_matches_broadcast_stack(self, image_stack_pair):
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(average_ssim(ref_2d, tests), average_ssim(ref_3d, tests))

    def test_average_snr_2d_ref_returns_float(self, image_stack_pair):
        refs, tests = image_stack_pair
        assert isinstance(average_snr(refs[0], tests), float)

    def test_average_mae_2d_ref_zero_for_identical(self):
        """2-D ref equal to every slice in the test stack → MAE = 0."""
        ref_2d = np.random.default_rng(7).random((16, 16)) * 255.0
        tests = np.broadcast_to(ref_2d[np.newaxis], (3, 16, 16)).copy()
        assert average_mae(ref_2d, tests) == 0.0

    def test_average_psnr_2d_ref_custom_max_pixel_value(self, image_stack_pair):
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(
            average_psnr(ref_2d, tests, max_pixel_value=128.0),
            average_psnr(ref_3d, tests, max_pixel_value=128.0),
        )

    def test_average_ssim_2d_ref_custom_data_range(self, image_stack_pair):
        refs, tests = image_stack_pair
        ref_2d = refs[0]
        ref_3d = np.broadcast_to(ref_2d[np.newaxis], tests.shape).copy()
        assert_almost_equal(
            average_ssim(ref_2d, tests, data_range=128.0),
            average_ssim(ref_3d, tests, data_range=128.0),
        )
