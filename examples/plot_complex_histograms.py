"""
Complex histograms
==================

Shows how spectrum of random matrices with complex eigenvalues is plotted.
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import CircularEnsemble

##############################################################################
# Several ensembles have symmetric or hermitian matrices, so their eigenvalues
# are real. However, there are some cases where random matrices are not
# symmetric either hermitian. Therefore, their eigenvalues can be complex.
#
# This is the case of Circular Unitary Ensemble (CUE) and Circular Symplectic
# Ensemble (CSE).
#
# scikit-rmt takes into account this possibility and uses histograms in 2D in
# order to represent spectrum distribution of given ensembles.

##############################################################################
# In this example, eigenvalue spectrum of CUE random matrix is shown.

cue = CircularEnsemble(beta=2, n=1000)
cue.plot_eigval_hist(bins=80, interval=(-2.2,2.2))

##############################################################################
# And the eigenvalue spectrum of CSE.

cse = CircularEnsemble(beta=4, n=1000)
cse.plot_eigval_hist(bins=80, interval=(-2.2,2.2))

##############################################################################
# The provided heatmap gives an illustration of the eigenvalue accumulation.


