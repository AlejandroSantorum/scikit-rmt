"""
The :mod:`skrmt.ensemble` module includes models based on
random matrix ensembles.
"""

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble
from .manova_ensemble import ManovaEnsemble
from .circular_ensemble import CircularEnsemble

from .spectral_law import WignerSemicircleDistribution
from .spectral_law import MarchenkoPasturDistribution
from .spectral_law import TracyWidomDistribution
from .spectral_law import ManovaSpectrumDistribution

from .plot_law import wigner_semicircle
from .plot_law import marchenko_pastur
from .plot_law import tracy_widom
from .plot_law import manova_spectrum

from .tridiagonal_utils import tridiag_eigval_neg
from .tridiagonal_utils import tridiag_eigval_hist
from .tridiagonal_utils import householder_reduction


__all__ = ["GaussianEnsemble", "WishartEnsemble",
           "ManovaEnsemble", "CircularEnsemble",
           "WignerSemicircleDistribution",
           "MarchenkoPasturDistribution",
           "TracyWidomDistribution",
           "ManovaSpectrumDistribution",
           "wigner_semicircle",
           "marchenko_pastur",
           "tracy_widom",
           "manova_spectrum",
           "tridiag_eigval_neg",
           "tridiag_eigval_hist",
           "householder_reduction"
        ]
