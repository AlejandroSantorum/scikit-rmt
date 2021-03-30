"""
The :mod:`rmtpy.ensemble` module includes models based on
random matrix ensembles.
"""

from .gaussian_ensemble import GaussianEnsemble
from .gaussian_ensemble import GOE, GUE, GSE

__all__ = ["GaussianEnsemble",
           "GOE", "GUE", "GSE"]