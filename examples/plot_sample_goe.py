"""
Sampling GOE
============

Shows the sampling method of GOE, the most known random matrix ensemble.
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import GaussianEnsemble

##############################################################################
# First, we create a GaussianEnsemble object
goe = GaussianEnsemble(beta=1, n=5)

##############################################################################
# Showing the sampled matrix
print(goe.matrix)

##############################################################################
# Resampling it
goe.sample()
print(goe.matrix)
