"""
The :mod:`skrmt.covariance` module implements different methods
to estimate covariance matrices.
"""

from .estimator import sample_estimator
from .estimator import fsopt_estimator
from .estimator import linear_shrinkage_estimator
from .estimator import analytical_shrinkage_estimator
from .estimator import empirical_bayesian_estimator
from .estimator import minimax_estimator

from .metrics import loss_mv, loss_frobenius
from .metrics import prial_mv


__all__ = ["sample_estimator", "fsopt_estimator",
           "linear_shrinkage_estimator",
           "analytical_shrinkage_estimator",
           "empirical_bayesian_estimator",
           "minimax_estimator",
           "loss_mv", "loss_frobenius",
           "prial_mv"]
