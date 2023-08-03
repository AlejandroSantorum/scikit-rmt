'''Utils Test module

Testing utils sub-module
'''

from skrmt.ensemble.utils import rand_mtx_max_eigvals
from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble


def test_rand_mtx_max_eigvals():
    """Testing getting maximum eigenvalues of an Ensemble object
    """
    goe = GaussianEnsemble(beta=4, n=10, random_state=1)

    max_eigval = goe.eigvals().max()

    max_vals = rand_mtx_max_eigvals(goe, n_eigvals=1, normalize=False, random_state=1)
    assert max_eigval == max_vals[0]

    max_vals_norm = rand_mtx_max_eigvals(goe, n_eigvals=1, normalize=True, random_state=1)
    # now it has to be different since it was normalized by Tracy-Widom distr. constants
    assert max_eigval != max_vals_norm[0]
