'''Utils Test module

Testing utils sub-module
'''
import os
import pytest

from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.utils import (
    plot_spectral_hist_and_law,
    standard_vs_tridiag_hist,
)


TMP_DIR_PATH = None


@pytest.fixture(scope="module", autouse=True)
def _setup_tmp_dir(tmp_path_factory):
    '''Create a pytest-managed directory for this module's plot files.'''
    global TMP_DIR_PATH
    TMP_DIR_PATH = str(tmp_path_factory.mktemp("ensemble_utils"))


class TestUtils:
    """Test scikit-rmt ensemble utils
    """

    def test_plot_spectral_hist_and_law(self):
        """Testing plotting the spectral histogram of a random matrix ensemble
        alongside the PDF of the corresponding spectral law.
        """
        fig_name = "test_test_plot_spectral_hist_and_law.png"

        goe = GaussianEnsemble(beta=1, n=10, random_state=1)
        plot_spectral_hist_and_law(
            ensemble=goe,
            bins=20,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name))

    def test_standard_vs_tridiag_hist(self):
        """Testing plotting the spectral histogram of a random matrix ensemble
        in its standard form vs its corresponding tridiagonal form.
        """
        fig_name = "test_standard_vs_tridiag_hist.png"

        goe = GaussianEnsemble(beta=1, n=5)
        standard_vs_tridiag_hist(
            ensemble=goe,
            bins=10,
            random_state=1,
            savefig_path=TMP_DIR_PATH+"/"+fig_name
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name))
