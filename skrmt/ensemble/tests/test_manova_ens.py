import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from skrmt.ensemble import ManovaEnsemble


##########################################
### Manova Real Ensemble = MRE

def test_manovaReal_init():
    M = 3
    N1 = 5
    N2 = 10

    np.random.seed(1)
    mr = ManovaEnsemble(beta=1, m=M, n1=N1, n2=N2)

    assert(mr.matrix.shape == (M,M))

    assert_almost_equal(mr.matrix, np.array([[0.30694308, -0.20376032, 0.12255971],
                                            [-0.15432555, 0.57689371, -0.1121342 ],
                                            [0.20435358, -0.32120057, 0.26201473]]),
                        decimal=7)


def test_mre_set_size():
    M1, N11, N12 = 3, 5, 8
    M2, N21, N22 = 2, 4, 6

    mr = ManovaEnsemble(beta=1, m=M1, n1=N11, n2=N12)
    assert(mr.m == M1)
    assert(mr.n1 == N11)
    assert(mr.n2 == N12)
    assert(mr.matrix.shape == (M1,M1))

    mr.set_size(m=M2, n1=N21, n2=N22, resample_mtx=False)
    assert(mr.m == M2)
    assert(mr.n1 == N21)
    assert(mr.n2 == N22)
    assert(mr.matrix.shape == (M1,M1))

    mr.set_size(m=M2, n1=N21, n2=N22, resample_mtx=True)
    assert(mr.m == M2)
    assert(mr.n1 == N21)
    assert(mr.n2 == N22)
    assert(mr.matrix.shape == (M2,M2))


##########################################
### Manova Complex Ensemble = MCE

def test_manovaComplex_init():
    M = 2
    N1 = 1
    N2 = 10

    np.random.seed(1)
    mc = ManovaEnsemble(beta=2, m=M, n1=N1, n2=N2)

    assert(mc.matrix.shape == (M,M))

    assert_almost_equal(mc.matrix, np.array([[0.15676774-0.11549436j, 0.53659572+0.43077891j],
                                            [-0.10473074-0.09411069j, 0.2265188 -0.44303466j]]),
                        decimal=7)


def test_mce_set_size():
    M1, N11, N12 = 4, 6, 8
    M2, N21, N22 = 3, 9, 12

    mc = ManovaEnsemble(beta=2, m=M1, n1=N11, n2=N12)
    assert(mc.m == M1)
    assert(mc.n1 == N11)
    assert(mc.n2 == N12)
    assert(mc.matrix.shape == (M1,M1))

    mc.set_size(m=M2, n1=N21, n2=N22, resample_mtx=False)
    assert(mc.m == M2)
    assert(mc.n1 == N21)
    assert(mc.n2 == N22)
    assert(mc.matrix.shape == (M1,M1))

    mc.set_size(m=M2, n1=N21, n2=N22, resample_mtx=True)
    assert(mc.m == M2)
    assert(mc.n1 == N21)
    assert(mc.n2 == N22)
    assert(mc.matrix.shape == (M2,M2))


##########################################
### Manova Quaternion Ensemble = MQE

def test_manovaQuatern_init():
    M = 2
    N1 = 10
    N2 = 20

    np.random.seed(1)
    mq = ManovaEnsemble(beta=4, m=M, n1=N1, n2=N2)

    assert(mq.matrix.shape == (2*M,2*M))

    assert_almost_equal(mq.matrix, np.array([[0.89445195+0.37755429j, -0.44197811+0.30301297j, 0.42617549-1.46493341j, 0.5261217-2.14539756j],
                                            [0.33995013-0.08355132j, 0.17411644-0.08552232j, 0.11361442-0.79709277j, -0.0351003-1.2905895j ],
                                            [-0.42617549-1.46493341j, -0.5261217-2.14539756j, 0.89445195-0.37755429j, -0.44197811-0.30301297j],
                                            [-0.11361442-0.79709277j, 0.0351003-1.2905895j, 0.33995013+0.08355132j, 0.17411644+0.08552232j]]),
                        decimal=7)


def test_mqe_set_size():
    M1, N11, N12 = 2, 5, 7
    M2, N21, N22 = 4, 5, 6

    mq = ManovaEnsemble(beta=4, m=M1, n1=N11, n2=N12)
    assert(mq.m == M1)
    assert(mq.n1 == N11)
    assert(mq.n2 == N12)
    assert(mq.matrix.shape == (2*M1,2*M1))

    mq.set_size(m=M2, n1=N21, n2=N22, resample_mtx=False)
    assert(mq.m == M2)
    assert(mq.n1 == N21)
    assert(mq.n2 == N22)
    assert(mq.matrix.shape == (2*M1,2*M1))

    mq.set_size(m=M2, n1=N21, n2=N22, resample_mtx=True)
    assert(mq.m == M2)
    assert(mq.n1 == N21)
    assert(mq.n2 == N22)
    assert(mq.matrix.shape == (2*M2,2*M2))