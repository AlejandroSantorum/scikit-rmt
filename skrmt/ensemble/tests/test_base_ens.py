'''Base Ensemble Test module

Testing _Ensemble abstract class
'''

import os
import pytest
import shutil

from skrmt.ensemble.base_ensemble import _Ensemble
from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble


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


def test_init_exception():
    """Testing the abstract class cannot be instantiated
    """
    with pytest.raises(TypeError):
        _ = _Ensemble()


def test_base_set_eigval_norm_const():
    """Testing setting a custom eigenvalue normalization constant
    """
    goe = GaussianEnsemble(beta=1, n=10, tridiagonal_form=False)

    assert goe.eigval_norm_const is not None

    goe.set_eigval_norm_const(100.0)
    assert goe.eigval_norm_const == 100.0

    goe.set_eigval_norm_const(0.1)
    assert goe.eigval_norm_const == 0.1



def test_base_ens_plot():
    """Testing plot eigval hist
    """
    fig_name = "test_base_ens_plot_eigval_hist.png"
    goe = GaussianEnsemble(beta=1, n=100, tridiagonal_form=False)
    goe.plot_eigval_hist(savefig_path=TMP_DIR_PATH+"/"+fig_name)
    assert os.path.isfile(os.path.join(TMP_DIR_PATH, fig_name)) == True
