"""
Analyzing spectral laws 
=======================

In this section, we will briefly explain what is the spectral distribution
of a random matrix and how to plot it in order to study its properties.
Most ensembles spectral distribution are explained through spectral laws,
that are going to be detailed as well. 

"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

##############################################################################
# Spectral distribution
# ------------------------------
# 
# Most of the information of a random matrix is explained through their
# eigenvalues. Random matrix eigenvalues or random matrix spectrum follows
# a certain distribution depending on the ensemble. We would like to study
# that distribution in order to have insight of the random matrix limiting
# behaviour.
#
# We can plot the eigenvalue density of any random matrix ensemble implemented
# in scikit-rmt. However, some technical details must be taken into account.

##############################################################################
# As an example, we can try to plot the spectral density of the Gaussian
# Orthogonal Ensemble (GOE) in the interval (-10, 10).

from skrmt.ensemble import GaussianEnsemble

goe = GaussianEnsemble(beta=1, n=1000)
goe.plot_eigval_hist(bins=80, interval=(-10, 10), norm_const=1)

##############################################################################
# As we can see, the histogram does not explain barely anything about
# the ensemble. That is because we are not using correctly the theory
# behind the model.
#
# Most of the ensembles need their eigenvalues to be normalized by a
# certain constant in order to perceive their limiting density properly.
# By default, the scikit-rmt is implemented to normalize eigenvalues
# with the more suitable constant depending on the ensemble.

##############################################################################
# In the former example, GOE matrices of size :math:`n \times n` should be
# normalized by :math:`1/\sqrt{n}`. 

import numpy as np
from skrmt.ensemble import GaussianEnsemble

n=1000
goe = GaussianEnsemble(beta=1, n=n)
goe.plot_eigval_hist(bins=80, interval=(-10, 10), norm_const=1/np.sqrt(n))

##############################################################################
# Additionally, we can control the plotting interval to get a better picture
# of the spectral density. In the GOE case, the spectral distribution is
# concentrated in the (-2,2) interval.

import numpy as np
from skrmt.ensemble import GaussianEnsemble

n=1000
goe = GaussianEnsemble(beta=1, n=n)
goe.plot_eigval_hist(bins=80, interval=(-2, 2), norm_const=1/np.sqrt(n))

##############################################################################
# This example would be equivalent for Gaussian Unitary Ensemble (GUE) and
# Gaussian Symplectic Ensemble (GSE).

##############################################################################
# Another example would be using the Wishart Ensemble. In the following
# snippet of code Wishart Real Ensemble (WRE) is sampled.

from skrmt.ensemble import WishartEnsemble

p, n = 1000, 3000
wre = WishartEnsemble(beta=1, p=p, n=n)
wre.plot_eigval_hist(bins=80, interval=(-5,5))

##############################################################################
# By not specifying the normalization constant, the library has taken charge
# of it by itself. The plotting interval still can be improved.

from skrmt.ensemble import WishartEnsemble

p, n = 1000, 3000
wre = WishartEnsemble(beta=1, p=p, n=n)
wre.plot_eigval_hist(bins=80, interval=(0.2, 2.2))

##############################################################################
# This example would be equivalent for Wishart Complex Ensemble (WCE) and
# Wishart Quaternion Ensemble (WQE), with the detail of controlling its
# representation interval.

##############################################################################
# So far we have analyzed spectral distributions of GOE and WRE. In the first
# example it is easy to see that its eigenvalue distribution is quite similar
# to a semicircle. Thats because Gaussian Ensemble random matrices spectrum
# follow *Wigner's Semicircle Law*. In the second example, the shown density
# has a very particular shape too, it is known as the *Marchenko-Pastur Law*.
# But, what is a spectral law?

##############################################################################
# Spectral laws
# ------------------------------
# 
# Spectral laws define random matrix eigenvalue distribution when its size
# goes to infinity. So the spectral laws describes the limiting behavior
# of the spectrum of a random matrix ensemble.
#
# The most known spectral laws, and implemented by scikit-rmt, are the
# following:
#
# - **Winger Semicircle Law**: describes the limiting behaviour of Gaussian
#   Ensemble random matrix spectrum.
#
# - **Marchenko-Pastur Law**: describes the limiting behaviour of Wishart
#   Ensemble random matrix spectrum.
#
# - **Tracy-Widom Law**: describes the limiting behaviour of the largest
#   eigenvalue of a Gaussian Ensemble random matrix.

##############################################################################
# Anyone can forget about the specific plotting details and directly study
# those laws by using the functions provided in scikit-rmt.

##############################################################################
# We can easily analyze **Wigner's Semicircle Law** as follows.

from skrmt.ensemble import wigner_semicircular_law

wigner_semicircular_law(ensemble='goe', n_size=2000, bins=80)

##############################################################################
# We can also study **Marchenko-Pastur Law** as by:

from skrmt.ensemble import marchenko_pastur_law

marchenko_pastur_law(ensemble='wre', p_size=2000, n_size=6000, bins=80)

##############################################################################
# Finally, **Tracy-Widom Law** can be represented using:

from skrmt.ensemble import tracy_widom_law

tracy_widom_law(ensemble='goe', n_size=100, times=10000, bins=80)

##############################################################################
# Spectral Laws analytical expression
# -----------------------------------
# 
# The spectral laws described so far have been proven to converge to certain
# analytical functions that defines the limiting behaviour of the eigenvalue
# distribution of the random matrices. The functions of scikit-rmt that are
# capable of plotting the spectral laws also support the representation of the
# theoretical eigenvalue pdf.

##############################################################################
# The analytical probability function for the Gaussian Ensemble, known as
# Wigner Semicircle Law, supported on :math:`[-R, R]` and centered at :math:`(0,0)`
# is :math:`f(x) = \frac{2}{\pi R^2} \sqrt{R^2 - x^2}`.

from skrmt.ensemble import wigner_semicircular_law

wigner_semicircular_law(ensemble='goe', n_size=2000, bins=80, density=True, limit_pdf=True)

##############################################################################
# The analytical probability function for the Wishart Ensemble known as
# Marchenko-Pastur Law with parameter :math:`\lambda = p/n \in (0,1]` is
# :math:`f_{\lambda}(x) = \frac{1}{2\pi \sigma^2}\frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{\lambda x}`,
# where :math:`\lambda_{\pm} = \sigma^2 (1 \pm \sqrt{\lambda})^2`. 
# If :math:`\lambda > 1` then the limiting distribution has an additional
# mass probability point in the origin of size :math:`1 - \frac{1}{\lambda}`.

from skrmt.ensemble import marchenko_pastur_law

marchenko_pastur_law(ensemble='wre', p_size=2000, n_size=6000, bins=80, density=True, limit_pdf=True)

##############################################################################
# In the other hand, the Tracy-Widom Law has a complex analytical expression,
# that is the solution of a particular non-linear differential equation, described
# in detail in:
# S. Bauman. "The Tracy-Widom Distribution and its Application to Statistical Physics".
# MIT Department of Physics. 2017.
# The package scikit-rmt represents a precise
# approximation of the theoretical Tracy-Widom pdf.

from skrmt.ensemble import tracy_widom_law

tracy_widom_law(ensemble='goe', n_size=10, times=5000, bins=80, density=True, limit_pdf=True)

##############################################################################
# Finally, the limiting distribution of the Manova Ensemble is not described
# by any famous Law, but its expression has been determined. This library
# provides functionality to show it on top of the empirical histogram of the
# eigenvalue spectrum.

from skrmt.ensemble import manova_spectrum_distr

manova_spectrum_distr(ensemble='mre', m_size=1000, n1_size=3000, n2_size=3000,
                      bins=80, density=True, limit_pdf=True)
