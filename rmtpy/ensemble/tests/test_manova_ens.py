import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from rmtpy.ensemble import ManovaReal, ManovaComplex, ManovaQuaternion


def test_manovaReal_init():
    M = 3
    N1 = 5
    N2 = 10

    np.random.seed(1)
    mr = ManovaReal(m=M, n1=N1, n2=N2)

    assert(mr.matrix.shape == (M,M))

    assert_almost_equal(mr.matrix, np.array([[0.30694308, -0.20376032, 0.12255971],
                                            [-0.15432555, 0.57689371, -0.1121342 ],
                                            [0.20435358, -0.32120057, 0.26201473]]),
                        decimal=4)


def test_manovaComplex_init():
    M = 2
    N1 = 1
    N2 = 10

    np.random.seed(1)
    mc = ManovaComplex(m=M, n1=N1, n2=N2)

    assert(mc.matrix.shape == (M,M))

    assert_almost_equal(mc.matrix, np.array([[0.15676774-0.11549436j, 0.53659572+0.43077891j],
                                            [-0.10473074-0.09411069j, 0.2265188 -0.44303466j]]),
                        decimal=4)


def test_manovaQuatern_init():
    M = 2
    N1 = 10
    N2 = 20

    np.random.seed(1)
    mq = ManovaQuaternion(m=M, n1=N1, n2=N2)

    assert(mq.matrix.shape == (2*M,2*M))

    assert_almost_equal(mq.matrix, np.array([[0.89445195+0.37755429j, -0.44197811+0.30301297j, 0.42617549-1.46493341j, 0.5261217-2.14539756j],
                                            [0.33995013-0.08355132j, 0.17411644-0.08552232j, 0.11361442-0.79709277j, -0.0351003-1.2905895j ],
                                            [-0.42617549-1.46493341j, -0.5261217-2.14539756j, 0.89445195-0.37755429j, -0.44197811-0.30301297j],
                                            [-0.11361442-0.79709277j, 0.0351003-1.2905895j, 0.33995013+0.08355132j, 0.17411644+0.08552232j]]),
                        decimal=4)