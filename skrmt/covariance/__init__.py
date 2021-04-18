"""
The :mod:`skrmt.covariance` module implements different methods
to estimate covariance matrices.
"""

from .analytical_shrinkage import AnalyticalShrinkage
from .estimator import SampleEstimator


__all__ = ["AnalyticalShrinkage", "SampleEstimator",]