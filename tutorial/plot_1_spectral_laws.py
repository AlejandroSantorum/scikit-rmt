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
# -------------
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
# Wigner Semicircle Law
# ---------------------
#
# **Wigner's Semicircle Law** characterizes the density of eigenvalues of
# sufficiently large Wigner matrices with second moment :math:`\rho`.
#
#   .. math:: \mu_{sc}(dx) = \frac{1}{2\pi \rho} \sqrt{4\rho - x^2}\mathbf{1}_{|x| \le 2\sqrt{\rho}} dx.
#
# The previous equation can be re-formulated by using that the radius of the
# semicircle is :math:`R = 2\sqrt{\rho}`:
#
#   .. math:: \mu_{sc}(dx) = \frac{2}{\pi R^2} \sqrt{R^2 - x^2}\mathbf{1}_{|x| \le R} dx.
#
# Random matrices from a Gaussian ensemble are of the Wigner type. Therefore,
# distribution of their eigenvalues scaled by :math:`1 / \sqrt{n}`, where
# :math:`n` is the size of the matrix, approaches asymptotically Wigner's Law
# at rate :math:`O(n^{-1/2})`.
#
# The *probability density function* (PDF) of the Wigner's Semicircle Law can
# be studied and plotted using **scikit-rmt** with the
# **WignerSemicircleDistribution** class.

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
# Law can also be analyzed and plotted using **WignerSemicircleDistribution** class.

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

##############################################################################
# Tracy-Widom Law
# ---------------
#
# The distribution of the largest eigenvalue of a Wigner matrix converges
# asymptotically (i.e. in the limit of infinite matrix size) to the
# **Tracy-Widom law**. The Tracy-Widom distribution can be defined as
# the limit:
#
#   .. math:: F_2 (s) = \lim_{n \to \infty} \mathbb{P}\left( \sqrt{2} (\lambda_{max} - \sqrt{2 n}) n^{1/6} \le s \right).
#
# The shift :math:`\sqrt{2n}` is used to keep the distributions centered
# at :math:`0`, and the factor :math:`\sqrt{2} n^{1/6}` is used because
# the standard deviation of the distributions scales with :math:`O(n^{-1/6})`.
#
# The probability density function (PDF) and cumulative distribution function (CDF)
# of the Tracy-Widom Law can be computed and graphically represented using the
# **TracyWidomDistribution** class from **scikit-rmt**.

import numpy as np
import matplotlib.pyplot as plt
from skrmt.ensemble.law import TracyWidomDistribution


x = np.linspace(-5, 2, num=1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

for beta in [1,2,4]:
    twd = TracyWidomDistribution(beta=beta)

    y_pdf = twd.pdf(x)
    y_cdf = twd.cdf(x)

    ax1.plot(x, y_pdf, label=f"$\\beta$ = {beta}")
    ax2.plot(x, y_cdf, label=f"$\\beta$ = {beta}")

ax1.legend()
ax1.set_xlabel("x", fontweight="bold")
ax1.set_ylabel("density", fontweight="bold")
ax1.set_title("Probability density function")

ax2.legend()
ax2.set_xlabel("x", fontweight="bold")
ax2.set_ylabel("distribution", fontweight="bold")
ax2.set_title("Cumulative distribution function")

fig.suptitle("Tracy Widom Law", fontweight="bold")
plt.show()

##############################################################################
# Marchenko-Pastur Law
# --------------------
#
# Another heavily studied type of random matrix are Wishart random matrices.
# The spectrum of the Wishart Ensemble is characterized by the Marchenko-Pastur Law.
# The **Marchenko-Pastur Law** describes the asymptotic behavior of the spectrum
# of a Wishart matrix. Consider the :math:`p \times p` Wishart matrix
# :math:`\mathbf{M} = \sum_{i=1}^{n}\mathbf{x}_i \mathbf{x}_i^\top`, where
# :math:`\mathbf{x}_i \sim N_p(\mathbf{0}, \sigma^2 \mathbf{I}_p)`. In the 
# limit :math:`p,n \to \infty` with :math:`\lambda = p/n \in (0, 1]` fixed,
# the (discrete) distribution of the eigenvalues of :math:`\mathbf{M}`
#
#   .. math:: F^{\mathbf{M}}(x) := \frac{1}{p} \#\{1 \leq i \leq p : \lambda_i(\mathbf{M}) \leq x\},
#
# converges weakly with probability 1 to 
#
#   .. math:: F^{\mathbf{M}}(x)  \underset{\substack{n,p \to \infty \\ p/n \to \lambda}}{\longrightarrow} \int_{-\infty}^{x}f_{\lambda}(t)dt \quad \forall x \in \mathbb{R},
#
# is the Marchenko-Pastur probability density function.
#
# A similar result is obtained if :math:`\lambda > 1`. In this case, the limiting
# distribution has an additional mass probability point in the origin of
# size :math:`1 - \frac{1}{\lambda}`.
#
# Below we show how we can use the class **MarchenkoPasturDistribution** from **scikit-rmt**
# to compute, study and illustrate the PDF of the Marchenko-Pastur Law.

import numpy as np
import matplotlib.pyplot as plt
from skrmt.ensemble.law import MarchenkoPasturDistribution


x1 = np.linspace(0, 4, num=1000)
x2 = np.linspace(0, 5, num=2000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

for ratio in [0.2, 0.4, 0.6, 1.0, 1.4]:
    mpl = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)

    y1 = mpl.pdf(x1)
    y2 = mpl.pdf(x2)

    ax1.plot(x1, y1, label=f"$\lambda$ = {ratio} ")
    ax2.plot(x2, y2, label=f"$\lambda$ = {ratio} ")

ax1.legend()
ax1.set_ylim(0, 1.4)
ax1.set_xlabel("x", fontweight="bold")
ax1.set_ylabel("density", fontweight="bold")

ax2.legend()
ax2.set_ylim(0, 1.4)
ax2.set_xlim(0, 1)
ax2.set_xlabel("x", fontweight="bold")
ax2.set_ylabel("density", fontweight="bold")

fig.suptitle("Marchenko-Pastur probability density function (PDF)", fontweight="bold")
plt.show()

##############################################################################
# Manova spectrum distribution
# ----------------------------
#
# The empirical density of eigenvalues of Manova Ensemble random matrix converges
# almost surely to
#
#   .. math:: f_{M}(x) = (a+b)\frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{2 \pi x (1-x)} I_{[\lambda_- , \lambda_+]} (x),
#
# where :math:`a` and :math:`b` are the matrix parameters, and
#
#   .. math:: \lambda_{\pm} = \left( \sqrt{\frac{a}{a+b}\left( 1 - \frac{1}{a+b} \right)} \pm \sqrt{\frac{1}{a+b}\left(1 - \frac{a}{a+b} \right)} \right)^2, \quad \lambda_{\pm} \in (0,1).
#
# The support of :math:`f_M` is the compact interval :math:`(0,1)`.
#
# The class ManovaSpectrumDistribution can be used to analyze the PDF and CDF of the
# spectrum of the Manova random matrices, as exemplified below.

import numpy as np
import matplotlib.pyplot as plt
from skrmt.ensemble.law import ManovaSpectrumDistribution

plt.rcParams['figure.dpi'] = 100

x1 = np.linspace(0, 1, num=1000)
x2 = np.linspace(0, 1, num=1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

for a in [1.0, 1.2, 1.4, 1.6]:
    for b in [2.0]:
        msd = ManovaSpectrumDistribution(beta=1, a=a, b=b)

        y1 = msd.pdf(x1)
        y2 = msd.pdf(x2)

        ax1.plot(x1, y1, label=f"$a$ = {a}, $b$ = {b}")
        ax2.plot(x2, y2, label=f"$a$ = {a}, $b$ = {b}")

ax1.legend()
ax1.set_xlabel("x", fontweight="bold")
ax1.set_ylabel("density", fontweight="bold")

ax2.legend()
ax2.set_ylim(0, 4)
ax2.set_xlabel("x", fontweight="bold")
ax2.set_ylabel("density", fontweight="bold")

fig.suptitle("Manova spectrum probability density function (PDF)", fontweight="bold")
plt.show()
