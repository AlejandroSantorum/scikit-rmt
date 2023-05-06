"""
Random matrix spectrum distribution
===================================

In this section we describe the tools provided by **scikit-rmt** to study,
analyze and compute the distribution of the spectral laws of some common
random matrix ensembles.

"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause


##############################################################################
# Spectral laws
# --------------------------
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
#
# - **Manova Spectrum distribution**: Although the random matrices of the
#   Manova ensemble have a defined limiting behaviour, the analytical law
#   that describes it is not named after anyone like the previous mentioned
#   laws. It was introduced by Wachter in *The Limiting Empirical Measure
#   of Multiple Discriminant Ratios* (1980).


##############################################################################
# Spectral laws for Wigner matrices
# ---------------------------------
#
# Wigner Semicircle Law
# =====================
#
# **Wigner's Semicircle Law** characterizes the density of eigenvalues of
# sufficiently large Wigner matrices with second moment :math:`\rho`.
#
#   .. math:: \mu_{sc}(dx) = \frac{1}{2\pi \rho} \sqrt{4\rho - x^2}\mathbf{1}_{|x| \le 2\sqrt{\rho}} dx.
#
# The previous equation can be re-formulated by using that the radius of the
# semicircle is :math:`$R = 2\sqrt{\rho}$`:
#
#   .. math:: \mu_{sc}(dx) = \frac{2}{\pi R^2} \sqrt{R^2 - x^2}\mathbf{1}_{|x| \le R} dx.
#
# Random matrices from a Gaussian ensemble are of the Wigner type. Therefore,
# distribution of their eigenvalues scaled by :math:`1 / \sqrt{n}`, where
# :math:`n` is the size of the matrix, approaches asymptotically Wigner's Law
# at rate :math:`O(n^{-1/2})`.
#
# The *probability density function* (PDF) of the Wigner's Semicircle Law can
# be plotted using **scikit-rmt** with the **WignerSemicircleDistribution** class.

import numpy as np
import matplotlib.pyplot as plt
from skrmt.ensemble.law import WignerSemicircleDistribution


x1 = np.linspace(-5, 5, num=1000)
x2 = np.linspace(-10, 10, num=2000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

for sigma in [0.5, 1.0, 2.0, 4.0]:
    wsd = WignerSemicircleDistribution(beta=1, center=0.0, sigma=sigma)

    y1 = wsd.pdf(x1)
    y2 = wsd.pdf(x2)

    ax1.plot(x1, y1, label=f"$\sigma$ = {sigma} (R = ${wsd.radius}$)")
    ax2.plot(x2, y2, label=f"$\sigma$ = {sigma} (R = ${wsd.radius}$)")

ax1.legend()
ax1.set_xlabel("x", fontweight="bold")
ax1.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax1.set_ylabel("density", fontweight="bold")

ax2.legend()
ax2.set_xlabel("x", fontweight="bold")
ax2.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
ax2.set_ylabel("density", fontweight="bold")

fig.suptitle("Wigner Semicircle probability density function (PDF)", fontweight="bold")
plt.show()

# Similarly, the *cumulative distribution function* (CDF) of the Wigner's Semicircle
# Law can also be plotted using **WignerSemicircleDistribution** class.

import numpy as np
import matplotlib.pyplot as plt
from skrmt.ensemble.law import WignerSemicircleDistribution


x1 = np.linspace(-5, 5, num=2000)
x2 = np.linspace(-10, 10, num=4000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

for sigma in [0.5, 1.0, 2.0, 4.0]:
    wsd = WignerSemicircleDistribution(beta=1, center=0.0, sigma=sigma)

    y1 = wsd.cdf(x1)
    y2 = wsd.cdf(x2)

    ax1.plot(x1, y1, label=f"$\sigma$ = {sigma} (R = ${wsd.radius}$)")
    ax2.plot(x2, y2, label=f"$\sigma$ = {sigma} (R = ${wsd.radius}$)")

ax1.legend()
ax1.set_xlabel("x", fontweight="bold")
ax1.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
ax1.set_ylabel("distribution", fontweight="bold")

ax2.legend()
ax2.set_xlabel("x", fontweight="bold")
ax2.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
ax2.set_ylabel("distribution", fontweight="bold")

fig.suptitle("Wigner Semicircle cumulative distribution function (CDF)", fontweight="bold")
plt.show()

#
# Tracy-Widom Law
# ================
#
# 