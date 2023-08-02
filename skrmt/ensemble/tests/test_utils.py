'''Utils Test module

Testing utils sub-module
'''

from skrmt.ensemble.utils import rand_mtx_max_eigvals
from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble


def test_rand_mtx_max_eigvals():
    """Testing getting maximum eigenvalues of an Ensemble object
    """
    goe = GaussianEnsemble(beta=1, n=10, random_state=1)

    max_eigval = goe.eigvals().max()
    max_vals = rand_mtx_max_eigvals(goe, size=1, random_state=1)
    assert max_eigval == max_vals[0]
