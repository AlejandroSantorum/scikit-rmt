"""
Gaussian Ensemble
=================

Defines Gaussian Ensemble random matrices
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import GaussianEnsemble

##############################################################################
# Gaussian Ensemble contains random matrices whose inputs are gaussian
# distributed.

##############################################################################
# Gaussian Orthogonal Ensemble (GOE)
# ----------------------------------
#
# Random matrices of GOE have real entries gaussian distributed. These
# matrices are invariant under orthogonal conjugation, i.e., if
# :math:`\mathbf{X} \in \text{GOE}` and :math:`\mathbf{O}` is an orthogonal
# matrix, then :math:`\mathbf{O} \mathbf{X} \mathbf{O}^T` is equally
# distributed as :math:`\mathbf{X}`.
#
# They are also known as 1-Hermite random matrices (beta = 1).

##############################################################################
# A random matrix of Gaussian Orthogonal Ensemble can be sampled using
# scikit-rmt with the following code.

goe = GaussianEnsemble(beta=1, n=4)
print(goe.matrix)

##############################################################################
# Gaussian Unitary Ensemble (GUE)
# -------------------------------
#
# Random matrices of GUE have complex entries gaussian distributed, i.e.,
# their real part and their complex part are gaussian distributed. These
# matrices are invariant under unitary conjugation, i.e., if
# :math:`\mathbf{X} \in \text{GUE}` and :math:`\mathbf{O}` is an unitary
# matrix, then :math:`\mathbf{O} \mathbf{X} \mathbf{O}^T` is equally
# distributed as :math:`\mathbf{X}`.
#
# They are also known as 2-Hermite random matrices (beta = 2).

##############################################################################
# A random matrix of Gaussian Unitary Ensemble can be sampled using
# scikit-rmt with the following code.

gue = GaussianEnsemble(beta=2, n=4)
print(gue.matrix)

##############################################################################
# Gaussian Symplectic Ensemble (GSE)
# ----------------------------------
#
# Random matrices of GSE have quaternionic entries gaussian distributed, i.e.,
# their four dimensions are gaussian distributed. These matrices are invariant
# under conjugation by the symplectic group.
#
# They are also known as 4-Hermite random matrices (beta = 4).

##############################################################################
# A random matrix of Gaussian Symplectic Ensemble can be sampled using
# scikit-rmt with the following code.

gse = GaussianEnsemble(beta=4, n=2)
print(gse.matrix)