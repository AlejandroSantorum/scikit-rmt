'''Manova Ensemble Test module

Testing ManovaEnsemble module
'''

import numpy as np
from numpy.testing import (
    assert_almost_equal,
)

from skrmt.ensemble import ManovaEnsemble


##########################################
### Manova Real Ensemble = MRE

def test_manova_real_init():
    '''Testing MRE init
    '''
    m_size = 3
    n1_size = 5
    n2_size = 10

    np.random.seed(1)
    mre = ManovaEnsemble(beta=1, m=m_size, n1=n1_size, n2=n2_size)

    assert mre.matrix.shape == (m_size,m_size)

    assert_almost_equal(mre.matrix, np.array([[0.30694308, -0.20376032, 0.12255971],
                                              [-0.15432555, 0.57689371, -0.1121342 ],
                                              [0.20435358, -0.32120057, 0.26201473]]),
                        decimal=7)


def test_mre_set_size():
    '''Testing setter to change matrix sizes of MRE
    '''
    m1_size, n11_size, n12_size = 3, 5, 8
    m2_size, n21_size, n22_size = 2, 4, 6

    mre = ManovaEnsemble(beta=1, m=m1_size, n1=n11_size, n2=n12_size)
    assert mre.m == m1_size
    assert mre.n1 == n11_size
    assert mre.n2 == n12_size
    assert mre.matrix.shape == (m1_size,m1_size)

    mre.set_size(m=m2_size, n1=n21_size, n2=n22_size, resample_mtx=False)
    assert mre.m == m2_size
    assert mre.n1 == n21_size
    assert mre.n2 == n22_size
    assert mre.matrix.shape == (m1_size,m1_size)

    mre.set_size(m=m2_size, n1=n21_size, n2=n22_size, resample_mtx=True)
    assert mre.m == m2_size
    assert mre.n1 == n21_size
    assert mre.n2 == n22_size
    assert mre.matrix.shape == (m2_size,m2_size)


def test_beta1_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    m_size, n1_size, n2_size = 4, 6, 8
    mre = ManovaEnsemble(beta=1, m=m_size, n1=n1_size, n2=n2_size)

    mre.matrix = np.zeros((m_size,m_size))
    assert mre.eigval_pdf() == 0.0

    mre.matrix = np.eye(m_size)
    assert mre.eigval_pdf() == 0.0


##########################################
### Manova Complex Ensemble = MCE

def test_manova_complex_init():
    '''Testing MCE init
    '''
    m_size = 2
    n1_size = 1
    n2_size = 10

    np.random.seed(1)
    mce = ManovaEnsemble(beta=2, m=m_size, n1=n1_size, n2=n2_size)

    assert mce.matrix.shape == (m_size,m_size)

    assert_almost_equal(mce.matrix, np.array([[0.15676774-0.11549436j, 0.53659572+0.43077891j],
                                              [-0.10473074-0.09411069j, 0.2265188 -0.44303466j]]),
                        decimal=7)


def test_mce_set_size():
    '''Testing setter to change matrix sizes of MCE
    '''
    m1_size, n11_size, n12_size = 4, 6, 8
    m2_size, n21_size, n22_size = 3, 9, 12

    mce = ManovaEnsemble(beta=2, m=m1_size, n1=n11_size, n2=n12_size)
    assert mce.m == m1_size
    assert mce.n1 == n11_size
    assert mce.n2 == n12_size
    assert mce.matrix.shape == (m1_size,m1_size)

    mce.set_size(m=m2_size, n1=n21_size, n2=n22_size, resample_mtx=False)
    assert mce.m == m2_size
    assert mce.n1 == n21_size
    assert mce.n2 == n22_size
    assert mce.matrix.shape == (m1_size,m1_size)

    mce.set_size(m=m2_size, n1=n21_size, n2=n22_size, resample_mtx=True)
    assert mce.m == m2_size
    assert mce.n1 == n21_size
    assert mce.n2 == n22_size
    assert mce.matrix.shape == (m2_size,m2_size)


##########################################
### Manova Quaternion Ensemble = MQE

def test_manova_quatern_init():
    '''Testing MQE init
    '''
    m_size = 2
    n1_size = 10
    n2_size = 20

    np.random.seed(1)
    mqe = ManovaEnsemble(beta=4, m=m_size, n1=n1_size, n2=n2_size)

    assert mqe.matrix.shape == (2*m_size,2*m_size)

    mtx_sol = [[0.89445195+0.37755429j, -0.44197811+0.30301297j, \
                0.42617549-1.46493341j, 0.5261217-2.14539756j],
               [0.33995013-0.08355132j, 0.17411644-0.08552232j, \
                0.11361442-0.79709277j, -0.0351003-1.2905895j ],
               [-0.42617549-1.46493341j, -0.5261217-2.14539756j, \
                0.89445195-0.37755429j, -0.44197811-0.30301297j],
               [-0.11361442-0.79709277j, 0.0351003-1.2905895j, \
                0.33995013+0.08355132j, 0.17411644+0.08552232j]]

    assert_almost_equal(mqe.matrix, np.array(mtx_sol), decimal=7)


def test_mqe_set_size():
    '''Testing setter to change matrix sizes of MQE
    '''
    m1_size, n11_size, n12_size = 2, 5, 7
    m2_size, n21_size, n22_size = 4, 5, 6

    mqe = ManovaEnsemble(beta=4, m=m1_size, n1=n11_size, n2=n12_size)
    assert mqe.m == m1_size
    assert mqe.n1 == n11_size
    assert mqe.n2 == n12_size
    assert mqe.matrix.shape == (2*m1_size,2*m1_size)

    mqe.set_size(m=m2_size, n1=n21_size, n2=n22_size, resample_mtx=False)
    assert mqe.m == m2_size
    assert mqe.n1 == n21_size
    assert mqe.n2 == n22_size
    assert mqe.matrix.shape == (2*m1_size,2*m1_size)

    mqe.set_size(m=m2_size, n1=n21_size, n2=n22_size, resample_mtx=True)
    assert mqe.m == m2_size
    assert mqe.n1 == n21_size
    assert mqe.n2 == n22_size
    assert mqe.matrix.shape == (2*m2_size,2*m2_size)
