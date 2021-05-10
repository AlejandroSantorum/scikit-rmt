"""
Wishart Ensemble
=================

Defines Wishart Ensemble random matrices
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import WishartEnsemble

##############################################################################
# Wishart Ensemble contains random matrices formed by the multiplication
# of two random matrices whose entries are gaussian distributed.

##############################################################################
# Wishart Real Ensemble (WRE)
# ---------------------------
#
# Random matrices of WRE are formed by multiplying a random real standard
# gaussian matrix of size :math:`p \times n` by its transpose.
#
# They are also known as 1-Laguerre random matrices (beta = 1).

##############################################################################
# A random matrix of Wishart Real Ensemble can be sampled using
# scikit-rmt with the following code.

wre = WishartEnsemble(beta=1, p=3, n=5)
print(wre.matrix)

##############################################################################
# Wishart Complex Ensemble (WCE)
# ------------------------------
#
# Random matrices of WCE are formed by multiplying a random complex
# standard gaussian matrix of size :math:`p \times n` by its transpose.
#
# They are also known as 2-Laguerre random matrices (beta = 2).

##############################################################################
# A random matrix of Wishart Complex Ensemble can be sampled using
# scikit-rmt with the following code.

wce = WishartEnsemble(beta=2, p=3, n=5)
print(wce.matrix)

##############################################################################
# Wishart Quaternion Ensemble (WQE)
# ---------------------------------
#
# Random matrices of WQE are formed by sampling two random complex standard
# guassian matrices (:math:`\mathbf{X}` and :math:`\mathbf{Y}`), stacking
# them to create matrix :math:`\mathbf{A} = (\mathbf{X}\ \mathbf{Y}; -\mathbf{Y}^*\ \mathbf{X}^*)`.
# Finally matrix :math:`\mathbf{A}` is multiplied by its transpose to generate
# a matrix WQE randon matrix.
#
# They are also known as 4-Laguerre random matrices (beta = 4).

##############################################################################
# A random matrix of Wishart Quaternion Ensemble can be sampled using
# scikit-rmt with the following code.

wqe = WishartEnsemble(beta=4, p=2, n=5)
print(wqe.matrix)