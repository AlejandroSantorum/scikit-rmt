"""
The :mod:`skrmt.ensemble` module includes models based on
random matrix ensembles.
"""

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble
from .manova_ensemble import ManovaEnsemble
from .circular_ensemble import CircularEnsemble


__all__ = ["GaussianEnsemble", "WishartEnsemble",
           "ManovaEnsemble", "CircularEnsemble"]
