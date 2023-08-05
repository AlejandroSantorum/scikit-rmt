"""
Eigenvalue Independence
=======================

In this tutorial, we highlight the fact that eigenvalues of a random matrix
sample are *not* independent. However, random numbers sampled using the PDF
of a spectral law (e.g. Wigner's Semicircle law) are actually drawn independently.

"""

# Author: Alejandro Santorum Varela
# License: BSD 3-Clause

##############################################################################
# Independent random samples vs. eigenvalues of a random matrix sample
# --------------------------------------------------------------------
# 
# Using **scikit-rmt** to simulate the behaviour of the ensembles, it is possible
# to illustrate how the eigenvalues of a random matrix sample are not independent,
# since they are draw from the same matrix sample. However, random variates sampled
# using the PDF of a spectral law (for example, Wigner's Semicircle law) are drawn
# independently.
#
# We can observe this phenomenon by plotting the **histogram of a random matrix**
# alongside the PDF of the corresponding spectral law. This is easily done by using
# the function ``plot_spectral_hist_and_law`` in ``skrmt.ensemble.utils``:

from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.utils import plot_spectral_hist_and_law

goe = GaussianEnsemble(beta=1, n=3000, use_tridiagonal=True)
plot_spectral_hist_and_law(ensemble=goe, bins=60)

##############################################################################
# In the previous example, the histogram of the spectrum of a sample from
# the Gaussian Ensemble was plotted next to the PDF of the Wigner's
# Semicircle law.
# 
# It can be observed that the eigenvalues of a single sample of a random matrix 
# are *not* independent since the fluctuations of the histogram compared to
# Wigner's Semicircle PDF are really small. In contrast, if we compare independent
# random samples from the Wigner Semicircle law and the actual PDF we observe
# higher fluctuations for the *same sample size* (in this example, 3000).

from skrmt.ensemble.spectral_law import WignerSemicircleDistribution

wsd = WignerSemicircleDistribution(beta=1)
wsd.plot_empirical_pdf(
    sample_size=3000,
    bins=60,
    density=True,
    plot_law_pdf=True
)

##############################################################################
# Similarly, this can be seen for other ensembles. For example, with the
# Wishart Ensemble.

from skrmt.ensemble.wishart_ensemble import WishartEnsemble
from skrmt.ensemble.utils import plot_spectral_hist_and_law

wre = WishartEnsemble(beta=1, p=1000, n=3000, use_tridiagonal=True)
plot_spectral_hist_and_law(ensemble=wre, bins=60)

##############################################################################
# The fluctuations with respect the PDF of the Marchenko-Pastur law are larger
# if we directly draw independent random samples from that distribution:

from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution

mpd = MarchenkoPasturDistribution(beta=1, ratio=3)
mpd.plot_empirical_pdf(
    sample_size=1000,
    bins=60,
    density=True,
    plot_law_pdf=True
)
