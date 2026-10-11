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


def test_beta1_joint_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    m_size, n1_size, n2_size = 4, 6, 8
    mre = ManovaEnsemble(beta=1, m=m_size, n1=n1_size, n2=n2_size)

    mre.matrix = np.zeros((m_size,m_size))
    assert mre.joint_eigval_pdf() == 0.0

    mre.matrix = np.eye(m_size)
    assert mre.joint_eigval_pdf() == 0.0


def test_manova_resample_random_state_and_cache():
    """Test reproducible resampling and resetting cached eigenvalues."""
    mre = ManovaEnsemble(
        beta=1, m=3, n1=5, n2=8, random_state=123
    )

    mre.eigvals()
    assert mre._eigvals is not None

    sampled_matrix = mre.resample(random_state=42)

    expected = ManovaEnsemble(
        beta=1, m=3, n1=5, n2=8, random_state=42
    )

    assert_almost_equal(sampled_matrix, expected.matrix)
    assert mre._eigvals is None


def test_manova_eigvals_cache_and_normalization():
    """Test cached eigenvalues and normalized eigenvalues."""
    mre = ManovaEnsemble(
        beta=1, m=3, n1=5, n2=8, random_state=123
    )

    vals = mre.eigvals()

    # Exercise the cached eigenvalue path.
    assert_almost_equal(mre.eigvals(), vals)

    # Exercise the normalized eigenvalue path.
    normalized_vals = mre.eigvals(normalize=True)
    assert_almost_equal(
        normalized_vals,
        vals * mre.eigval_norm_const,
    )


def test_manova_invalid_beta():
    """Test sampling with an unsupported beta."""
    mre = ManovaEnsemble(
        beta=1, m=3, n1=5, n2=8, random_state=123
    )

    mre.beta = 3

    # No sampling method matches beta=3, so sample returns None.
    assert mre.sample(random_state=42) is None


def test_manova_plot_eigval_hist_delegation(monkeypatch):
    """Test delegation to the base class plotting methods."""
    mre = ManovaEnsemble(
        beta=1, m=3, n1=5, n2=8, random_state=123
    )

    base_class = ManovaEnsemble.__mro__[1]
    hist_calls = {}
    plot_calls = {}

    def fake_hist(self, **kwargs):
        hist_calls.update(kwargs)

    def fake_plot(self, **kwargs):
        plot_calls.update(kwargs)

    # Test the protected plotting wrapper.
    monkeypatch.setattr(base_class, "_plot_eigval_hist", fake_hist)

    mre._plot_eigval_hist(
        bins=10,
        interval=(0, 1),
        density=True,
        normalize=True,
        avoid_img=False,
    )

    assert hist_calls == {
        "bins": 10,
        "interval": (0, 1),
        "density": True,
        "normalize": True,
        "avoid_img": False,
    }

    # Test the public plotting wrapper.
    monkeypatch.setattr(base_class, "plot_eigval_hist", fake_plot)

    mre.plot_eigval_hist(
        bins=10,
        interval=(0, 1),
        density=True,
        normalize=True,
        savefig_path="test_manova.png",
    )

    assert plot_calls == {
        "bins": 10,
        "interval": (0, 1),
        "density": True,
        "normalize": True,
        "savefig_path": "test_manova.png",
        "avoid_img": True,
    }


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

    assert_almost_equal(mce.matrix, np.array([[0.1027935+0.00612663j, -0.00875725+0.08423236j],
                                              [-0.01070604-0.07368927j, 0.06093032-0.00612663j]]),
                        decimal=7)


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

    mtx_sol = [[0.36695654+0.01615832j, 0.05808572-0.08831798j, \
                0.00268224-0.00172003j, -0.01413889-0.0228631j],
               [0.05675906+0.06962803j, 0.30129103-0.01333821j, \
                0.0081629+0.01974845j, -0.00731881-0.00630646j],
               [-0.00268224-0.00172003j, 0.01413889-0.0228631j, \
                0.36695654-0.01615832j, 0.05808572+0.08831798j],
               [-0.0081629+0.01974845j, 0.00731881-0.00630646j, \
                0.05675906-0.06962803j, 0.30129103+0.01333821j]]

    assert_almost_equal(mqe.matrix, np.array(mtx_sol), decimal=7)


def test_manova_joint_eigval_pdf_all_betas():
    """Test the joint PDF using explicitly supplied eigenvalues."""
    eigvals = np.array([0.2, 0.45, 0.8])

    for beta in (1, 2, 4):
        ensemble = ManovaEnsemble(
            beta=beta,
            m=3,
            n1=6,
            n2=7,
            random_state=123,
        )

        pdf = ensemble.joint_eigval_pdf(eigvals=eigvals)

        assert np.isfinite(pdf)
        assert pdf > 0.0
