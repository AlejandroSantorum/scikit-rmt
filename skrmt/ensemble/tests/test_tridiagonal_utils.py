'''Tridiagonal Utils Test module

Testing tridiagonalization utils.
'''

import pytest
import numpy as np

from skrmt.ensemble.tridiagonal_utils import tridiag_eigval_hist


def test_tridiag_eigval_hist_raise_interval_type():
    with pytest.raises(ValueError):
        tridiag_eigval_hist(
            tridiag_mtx=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            interval="invalid interval type",
            bins=10,
            density=True,
        )

def test_tridiag_eigval_hist_raise_bins_type():
    with pytest.raises(ValueError):
        tridiag_eigval_hist(
            tridiag_mtx=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            interval=(0,1),
            bins="invalid bins type",
            density=True,
        )

def test_tridiag_eigval_hist_not_raise_bins_type():
    tridiag_eigval_hist(
        tridiag_mtx=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        interval=(-10,10),
        bins=(-10,10),
        density=False,
    )