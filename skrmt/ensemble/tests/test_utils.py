'''Utils Test module

Testing utils sub-module
'''
import os
import pytest
import shutil

from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.utils import (
    plot_spectral_hist_and_law,
    standard_vs_tridiag_hist,
)


TMP_DIR_PATH = os.path.join(os.getcwd(), "skrmt/ensemble/tests/tmp")


@pytest.fixture(scope="module", autouse=True)
def _setup_tmp_dir(request):
    '''Function that is run before all tests in this script.

    It creates a temporary folder in order to store useful files
    for the following tests.
    '''
    # if the directory already exists, it is deleted
    if os.path.exists(TMP_DIR_PATH):
        shutil.rmtree(TMP_DIR_PATH)
    # creating temporary directory  
    os.mkdir(TMP_DIR_PATH)

    # specifying a function that will be run after all tests are completed
    request.addfinalizer(_remove_tmp_dir)


def _remove_tmp_dir():
    '''Function that removes the created temporary directory.

    The function is run when all tests in this module are completed.
    '''
    shutil.rmtree(TMP_DIR_PATH)


class TestUtils:

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
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
    
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
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
