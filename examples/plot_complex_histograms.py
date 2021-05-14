"""
Complex histograms
==================

Shows how spectrum of random matrices with complex eigenvalues is plotted.
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import ManovaEnsemble, CircularEnsemble

##############################################################################
# Several ensembles have symmetric or hermitian matrices, so their eigenvalues
# are real. However, there are some cases where random matrices are not
# symmetric either hermitian. Therefore, their eigenvalues can be complex.
#
# This is the case of Manova Complex Ensemble (MCE), Manova Quaternion Ensemble
# (MQE), Circular Unitary Ensemble (CUE) and Circular Symplectic Ensemble (CSE).
#
# scikit-rmt takes into account this possibility and uses histograms in 2D in
# order to represent spectrum distribution of given ensembles.

##############################################################################
# In this example, we plot the eigenvalue histogram of a MCE random matrix.

mce = ManovaEnsemble(beta=2, m=1000, n1=3000, n2=3000)
mce.plot_eigval_hist(bins=80, interval=(-2,2))

##############################################################################
# And now the eigenvalue histogram of a MQE random matrix.

mqe = ManovaEnsemble(beta=4, m=1000, n1=3000, n2=3000)
mqe.plot_eigval_hist(bins=80, interval=(-2,2))

##############################################################################
# In this example, eigenvalue spectrum of CUE random matrix is shown.

cue = CircularEnsemble(beta=2, n=1000)
cue.plot_eigval_hist(bins=80, interval=(-2.2,2.2))

##############################################################################
# And the eigenvalue spectrum of CSE.

cse = CircularEnsemble(beta=4, n=1000)
cse.plot_eigval_hist(bins=80, interval=(-2.2,2.2))


