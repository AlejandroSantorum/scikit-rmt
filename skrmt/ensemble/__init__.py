"""
The :mod:`skrmt.ensemble` module includes models based on
random matrix ensembles.
"""

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble
from .manova_ensemble import ManovaEnsemble
from .circular_ensemble import CircularEnsemble

from .spectral_law import (
    WignerSemicircleDistribution,
    MarchenkoPasturDistribution,
    TracyWidomDistribution,
    ManovaSpectrumDistribution,
)

from .utils import (
    standard_vs_tridiag_hist,
    plot_spectral_hist_and_law,
)

from .tridiagonal_utils import (
    tridiag_eigval_hist, householder_reduction
)


__all__ = [
    "GaussianEnsemble",
    "WishartEnsemble",
    "ManovaEnsemble",
    "CircularEnsemble",
    "WignerSemicircleDistribution",
    "MarchenkoPasturDistribution",
    "TracyWidomDistribution",
    "ManovaSpectrumDistribution",
    "standard_vs_tridiag_hist",
    "plot_spectral_hist_and_law",
    "tridiag_eigval_hist",
    "householder_reduction",
]
