import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from rmtpy.ensemble import GOE, GUE, GSE


def test_goe_init():
    np.random.seed(1)
    goe = GOE(3)

    assert_almost_equal(goe.matrix, np.array([[ 1.62434536, -0.84236252, 0.60832001],
                                              [-0.84236252, 0.86540763, -1.5313728 ],
                                              [ 0.60832001, -1.5313728, 0.3190391 ]]),
                        decimal=4)


def test_gue_init():
    np.random.seed(1)
    gue = GUE(3)

    assert_almost_equal(gue.matrix, np.array([[1.62434536-0.24937038j, -0.84236252+0.56984537j, 0.60832001-1.58001599j],
                                             [-0.84236252+0.56984537j, 0.86540763-0.38405435j, -1.5313728 +0.48067062j],
                                             [0.60832001-1.58001599j, -1.5313728 +0.48067062j, 0.3190391 -0.87785842j]]),
                        decimal=4)


def test_gse_init():
    np.random.seed(1)
    gse = GSE(2)

    assert_almost_equal(gse.matrix, np.array([[1.62434536+0.86540763j, -0.56996408-0.27836347j, 0.-0.3224172j, -0.85573916+0.37485754j],
                                             [-0.56996408-0.27836347j, -1.07296862-0.7612069j, 0.85573916+0.37485754j, 0.-1.09989127j],
                                             [0.-0.3224172j, 0.85573916+0.37485754j, 1.62434536-0.86540763j, -0.56996408+0.27836347j],
                                             [-0.85573916+0.37485754j, 0.-1.09989127j, -0.56996408+0.27836347j, -1.07296862+0.7612069j ]]),
                        decimal=4)