"""
Sampling GOE
============
Shows the sampling method of GOE
"""

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
