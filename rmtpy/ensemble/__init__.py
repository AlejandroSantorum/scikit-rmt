"""
The :mod:`rmtpy.ensemble` module includes models based on
random matrix ensembles.
"""

from .gaussian_ensemble import GaussianEnsemble
from .gaussian_ensemble import GOE, GUE, GSE

from .wishart_ensemble import WishartEnsemble
from .wishart_ensemble import WishartReal
from .wishart_ensemble import WishartComplex
from .wishart_ensemble import WishartQuaternion

__all__ = ["GaussianEnsemble",
           "GOE", "GUE", "GSE",
           "WishartReal",
           "WishartComplex",
           "WishartQuaternion"]