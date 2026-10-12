'''Base Ensemble Test module

Testing BaseEnsemble abstract class
'''

import os
from unittest.mock import patch
import pytest

from skrmt.ensemble.base_ensemble import BaseEnsemble
from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble


TMP_DIR_PATH = None


@pytest.fixture(scope="module", autouse=True)
def _setup_tmp_dir(tmp_path_factory):
    '''Create a pytest-managed directory for this module's plot files.'''
    global TMP_DIR_PATH
    TMP_DIR_PATH = str(tmp_path_factory.mktemp("base_ensemble"))


def test_init_exception():
    """Testing the abstract class cannot be instantiated
    """
    with pytest.raises(TypeError):
        _ = BaseEnsemble()  # pylint: disable=abstract-class-instantiated


def test_str_method():
    """Testing the __str__ method of BaseEnsemble
    """
    goe = GaussianEnsemble(beta=1, n=10, tridiagonal_form=False)
    str_repr = str(goe)
    assert str_repr == "<GaussianEnsemble shape=(10, 10)>"

    goe.matrix = None
    str_repr = str(goe)
    assert str_repr == "<GaussianEnsemble shape=(None, None)>"


def test_base_set_eigval_norm_const():
    """Testing setting a custom eigenvalue normalization constant
    """
    goe = GaussianEnsemble(beta=1, n=10, tridiagonal_form=False)

    assert goe.eigval_norm_const is not None

    goe._set_eigval_norm_const(100.0)  # pylint: disable=protected-access
    assert goe.eigval_norm_const == 100.0

    goe._set_eigval_norm_const(0.1)  # pylint: disable=protected-access
    assert goe.eigval_norm_const == 0.1


def test_base_ens_plot():
    """Testing plot eigval hist
    """
    fig_name = "test_base_ens_plot_eigval_hist.png"
    goe = GaussianEnsemble(beta=1, n=100, tridiagonal_form=False)
    goe.plot_eigval_hist(savefig_path=TMP_DIR_PATH+"/"+fig_name)
    assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name))


def test_base_ens_eigval_hist_raises():
    """Testing raising an exception for invalid ``interval`` type.
    """
    goe = GaussianEnsemble(beta=1, n=10, tridiagonal_form=False)
    with pytest.raises(ValueError):
        goe.eigval_hist(bins=10, interval="invalid interval type")


def test_base_ens_plot_eigval_hist_raises():
    """Testing raising an exception for invalid ``interval`` type.
    """
    goe = GaussianEnsemble(beta=1, n=10, tridiagonal_form=False)
    with pytest.raises(ValueError):
        goe.plot_eigval_hist(bins=10, interval="invalid interval type")


def test_base_ens_eigval_hist_valid_interval_and_avoid_img():
    """Testing eigval_hist with a valid tuple interval and avoid_img=True.
    """
    goe = GaussianEnsemble(beta=1, n=5, tridiagonal_form=False)
    observed, bin_edges = goe.eigval_hist(bins=5, interval=(-3.0, 3.0), avoid_img=True)
    assert len(observed) == 5
    assert len(bin_edges) == 6


@patch("skrmt.ensemble.base_ensemble.plt.show")
def test_base_ens_plot_eigval_hist_density_and_show(mock_show):
    """Testing plot_eigval_hist with density=True and savefig_path=None (plt.show).
    """
    goe = GaussianEnsemble(beta=1, n=5, tridiagonal_form=False)
    goe.plot_eigval_hist(bins=5, density=True, savefig_path=None)
    mock_show.assert_called_once()
