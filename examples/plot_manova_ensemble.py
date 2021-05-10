"""
Manova Ensemble
=================

Defines Manova Ensemble random matrices
"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

from skrmt.ensemble import ManovaEnsemble

##############################################################################
# Manova Ensemble random matrices are considered to be double Wishart random
# matrices.

##############################################################################
# Manova Real Ensemble (MRE)
# ---------------------------
#
# Random matrices of MRE are formed by sampling two random real standard
# guassian matrices (:math:`\mathbf{X}` and :math:`\mathbf{Y}`) of size
# :math:`m \times n_1` and :math:`m \times n_2` respectively. Then, matrix
# :math:`\mathbf{A} = \dfrac{\mathbf{X}\mathbf{X}^T}{\mathbf{X}\mathbf{X}^T  + \mathbf{Y}\mathbf{Y}^T}`
# generates a matrix of the MRE.
#
# They are also known as 1-Jacobi random matrices (beta = 1).

##############################################################################
# A random matrix of Manova Real Ensemble can be sampled using
# scikit-rmt with the following code.

mre = ManovaEnsemble(beta=1, m=3, n1=5, n2=5)
print(mre.matrix)

##############################################################################
# Manova Complex Ensemble (MCE)
# ------------------------------
#
# Random matrices of MCE are formed by sampling two random complex standard
# guassian matrices (:math:`\mathbf{X}` and :math:`\mathbf{Y}`) of size
# :math:`m \times n_1` and :math:`m \times n_2` respectively. Then, matrix
# :math:`\mathbf{A} = \dfrac{\mathbf{X}\mathbf{X}^T}{\mathbf{X}\mathbf{X}^T  + \mathbf{Y}\mathbf{Y}^T}`
# generates a matrix of the MCE.
#
# They are also known as 2-Jacobi random matrices (beta = 2).

##############################################################################
# A random matrix of Manova Complex Ensemble can be sampled using
# scikit-rmt with the following code.

mce = ManovaEnsemble(beta=2, m=3, n1=5, n2=5)
print(mce.matrix)

##############################################################################
# Manova Quaternion Ensemble (MQE)
# ---------------------------------
#
# Random matrices of MQE are formed by sampling two random complex standard 
# guassian matrices (:math:`\mathbf{X_1}` and :math:`\mathbf{X_2}`), both of
# size :math:`m \times n_1`. Another two random complex standard guassian matrices
# (:math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}`), both of size :math:`m \times n_2`,
# are sampled. They are stacked forming matrices :math:`\mathbf{X}` and
# :math:`\mathbf{Y}`:
# 
# :math:`\mathbf{X} = (\mathbf{X_1}\ \mathbf{X_2}; -\mathbf{X_2}^*\ \mathbf{X_1}^*)`
# 
# :math:`\mathbf{Y} = (\mathbf{Y_1}\ \mathbf{Y_2}; -\mathbf{Y_2}^*\ \mathbf{Y_1}^*)`
#
# Finally, matrix :math:`\mathbf{A} = \dfrac{\mathbf{X}\mathbf{X}^T}{\mathbf{X}\mathbf{X}^T  + \mathbf{Y}\mathbf{Y}^T}`
# generates a matrix of the MQE.
#
# They are also known as 4-Jacobi random matrices (beta = 4).

##############################################################################
# A random matrix of Manova Quaternion Ensemble can be sampled using
# scikit-rmt with the following code.

mqe = ManovaEnsemble(beta=4, m=2, n1=5, n2=5)
print(mqe.matrix)