'''Utils Test module

Testing utils sub-module
'''
import os
import pytest
import shutil

from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.utils import (
    rand_mtx_max_eigvals,
    plot_spectral_hist_and_law,
    plot_max_eigvals_tracy_widom,
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


    def test_rand_mtx_max_eigvals(self):
        """Testing getting maximum eigenvalues of an Ensemble object
        """
        goe = GaussianEnsemble(beta=4, n=10, random_state=1)

        max_eigval = goe.eigvals().max()

        max_vals = rand_mtx_max_eigvals(goe, n_eigvals=1, normalize=False, random_state=1)
        assert max_eigval == max_vals[0]

        max_vals_norm = rand_mtx_max_eigvals(goe, n_eigvals=1, normalize=True, random_state=1)
        # now it has to be different since it was normalized by Tracy-Widom distr. constants
        assert max_eigval != max_vals_norm[0]


    def test_plot_max_eigvals_tracy_widom(self):
        '''Testing plotting max eigenvalues histogram of an ensemble and comparing it
        with Tracy-Widom distribution
        '''
        fig_name = "test_plot_max_eigvals_tracy_widom.png"

        ens = GaussianEnsemble(beta=1, n=10)

        plot_max_eigvals_tracy_widom(
            ensemble=ens,
            n_eigvals=1,
            bins=10,
            random_state=1,
            savefig_path=TMP_DIR_PATH+"/"+fig_name,
        )
        assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
