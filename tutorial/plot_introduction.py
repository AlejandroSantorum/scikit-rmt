"""
Introduction
============

In this section, we will briefly explain what is Random Matrix Theory (RMT),
and we will introduce scikit-rmt, a library that provides RMT tools to
mathematicians, engineers, physicists, economists or any interested scientist
in the Python scientific ecosystem.

"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

##############################################################################
# What is Random Matrix Theory?
# ------------------------------
# 
# Random mathematical objects, such as graphs or tensors, have always been
# deeply studied because of their applications in economics, physics and
# engineering. The growing computational capacity allows to work and
# simulate this class of objects efficiently to improve in many disciplines.
# 
# In particular, two-dimensional random tensors are known as random matrices.
# Random Matrix Theory is the is the branch of statistics that studies the
# properties of matrices whose inputs are random variables. Many random matrices
# have similar behaviour and properties, especially if we analyze their
# eigenvalue spectrum, that is why they are usually classified into ensembles
# of random matrices, which are sets of them that share common features.

##############################################################################
# What is scikit-rmt?
# -------------------
#
# scikit-rmt is a Python library containing classes and functions that allow
# you to perform RMT tasks. Using it you can:
#
#     - Sample many types of random matrices of the main ensembles: Gaussian
#       ensemble, Wishart ensemble, Manova ensemble and Circular ensemble.
#     - Plot and analyze random matrix eigenvalue spectrum of the sampled
#       ensembles.
#     - Calculate eigenvalue joint probability density function of the sampled
#       random matrices.
#     - Plot and study main spectrum laws: Wigner Semicircle Law,
#       Marchenko-Pastur Law and Tracy-Widom Law.
#     - Estimation of covariance matrices with several methods, such as
#       non-lineal shrinkage analytical estimator.

##############################################################################
# As an example, the following code shows how to sample the most known
# ensemble of random matrices: Gaussian Orthogonal Ensemble (GOE).

from skrmt.ensemble import GaussianEnsemble

goe = GaussianEnsemble(beta=1, n=5)
print(goe.matrix)

##############################################################################
# To study its eigenvalue spectrum, its size should be a little bit larger.

from skrmt.ensemble import GaussianEnsemble

goe = GaussianEnsemble(beta=1, n=1000)
goe.plot_eigval_hist(bins=60, interval=(-2,2))