'''Miscellaneous Utils Test module

Testing misc sub-module functions
'''
import os
import logging
from unittest.mock import patch
import numpy as np
import pytest

from skrmt.ensemble.misc import (
    plot_func,
    relu,
    indicator,
    get_bins_centers_and_contour,
    get_logger,
)

TMP_DIR_PATH = None


@pytest.fixture(scope="module", autouse=True)
def _setup_tmp_dir(tmp_path_factory):
    '''Create a pytest-managed directory for this module's plot files.'''
    global TMP_DIR_PATH
    TMP_DIR_PATH = str(tmp_path_factory.mktemp("misc_utils"))


def test_plot_func_raises():
    """Testing plot_func raises ValueError for non-tuple interval."""
    with pytest.raises(ValueError):
        plot_func(interval="not-a-tuple", func=np.sin)


def test_plot_func_save():
    """Testing plot_func saves file successfully with title and ylabel."""
    fig_path = os.path.join(TMP_DIR_PATH, "plot_func.png")
    plot_func(
        interval=(0.0, 1.0),
        func=np.sin,
        num_x_vals=10,
        plot_title="Sine Wave",
        plot_ylabel="Amplitude",
        savefig_path=fig_path,
    )
    assert os.path.isfile(fig_path)


@patch("skrmt.ensemble.misc.plt.show")
def test_plot_func_show(mock_show):
    """Testing plot_func falls back to plt.show when savefig_path is None."""
    plot_func(interval=(0.0, 1.0), func=np.cos, savefig_path=None)
    mock_show.assert_called_once()


def test_relu():
    """Testing element-wise relu function."""
    x = np.array([-2.0, 0.0, 3.0])
    res = relu(x)
    np.testing.assert_array_equal(res, [0.0, 0.0, 3.0])


def test_indicator_raises():
    """Testing indicator function error raising for missing limits and invalid inclusive types."""
    with pytest.raises(ValueError, match="provide start and/or stop"):
        indicator(0.5, start=None, stop=None)
    with pytest.raises(ValueError, match="invalid interval inclusive parameter"):
        indicator(0.5, start=0.0, stop=1.0, inclusive="invalid")


def test_indicator_branches():
    """Testing indicator function across all boundary conditions and inclusive options."""
    x = np.array([0.0, 1.0, 2.0])
    
    # Start only branches
    np.testing.assert_array_equal(indicator(x, start=1.0, inclusive="both"), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(indicator(x, start=1.0, inclusive="neither"), [0.0, 0.0, 1.0])

    # Stop only branches
    np.testing.assert_array_equal(indicator(x, stop=1.0, inclusive="both"), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(indicator(x, stop=1.0, inclusive="neither"), [1.0, 0.0, 0.0])

    # Start and stop combined branches (all inclusive options)
    np.testing.assert_array_equal(indicator(x, start=0.0, stop=2.0, inclusive="both"), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(indicator(x, start=0.0, stop=2.0, inclusive="left"), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(indicator(x, start=0.0, stop=2.0, inclusive="right"), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(indicator(x, start=0.0, stop=2.0, inclusive="neither"), [0.0, 1.0, 0.0])


def test_get_bins_centers_and_contour():
    """Testing bin centers and contour calculation."""
    edges = [0.0, 2.0, 4.0]
    centers = get_bins_centers_and_contour(edges)
    assert centers == [0.0, 1.0, 3.0, 4.0]


def test_get_logger():
    """Testing logger getter helper."""
    logger = get_logger("skrmt.test_misc_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "skrmt.test_misc_logger"
