"""
Circular Ensemble
=================

Defines Circular Ensemble random matrices
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import CircularEnsemble

##############################################################################
# Circular Ensemble contains random matrices introduced by Freeman Dyson as
# modifications of the Gaussian Ensemble.

##############################################################################
# Circular Orthogonal Ensemble (COE)
# ----------------------------------
#
# Random matrices of COE have real entries gaussian distributed. These
# matrices are invariant under orthogonal conjugation, i.e., if
# :math:`\mathbf{X} \in \text{COE}` and :math:`\mathbf{O}` is an orthogonal
# matrix, then :math:`\mathbf{O} \mathbf{X} \mathbf{O}^T` is equally
# distributed as :math:`\mathbf{X}`.
#
# They are also known as 1-Dyson random matrices (beta = 1).

##############################################################################
# A random matrix of Circular Orthogonal Ensemble can be sampled using
# scikit-rmt with the following code.

coe = CircularEnsemble(beta=1, n=4)
print(coe.matrix)

##############################################################################
# Circular Unitary Ensemble (CUE)
# -------------------------------
#
# Random matrices of CUE have complex entries gaussian distributed, i.e.,
# their real part and their complex part are gaussian distributed. These
# matrices are invariant under unitary conjugation, i.e., if
# :math:`\mathbf{X} \in \text{CUE}` and :math:`\mathbf{O}` is an unitary
# matrix, then :math:`\mathbf{O} \mathbf{X} \mathbf{O}^T` is equally
# distributed as :math:`\mathbf{X}`.
#
# They are also known as 2-Dyson random matrices (beta = 2).

##############################################################################
# A random matrix of Circular Unitary Ensemble can be sampled using
# scikit-rmt with the following code.

cue = CircularEnsemble(beta=2, n=4)
print(cue.matrix)

##############################################################################
# Circular Symplectic Ensemble (CSE)
# ----------------------------------
#
# Random matrices of CSE  are invariant under conjugation by the symplectic
# group.
#
# They are also known as 4-Dyson random matrices (beta = 4).

##############################################################################
# A random matrix of Circular Symplectic Ensemble can be sampled using
# scikit-rmt with the following code.

cse = CircularEnsemble(beta=4, n=2)
print(cse.matrix)