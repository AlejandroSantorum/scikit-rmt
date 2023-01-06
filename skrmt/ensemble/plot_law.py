"""Plot Law module

This module contains several functions that simulate various
random matrix laws, including Wigner's Semicircle Law,
Marchenko-Pastur Law and Tracy-Widom Law.

"""

import math
import numpy as np
import matplotlib.pyplot as plt

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble
from .manova_ensemble import ManovaEnsemble
from .tracy_widom_approximator import TW_Approximator


def __get_bins_centers_and_contour(bins):
    """Calculates the centers and contour of the given bins.

    Computes the centers of the given bins. Also, the smallest and the largest bin
    delimitiers are included to define the countour of the representation interval.

    Args:
        bins (list): list of numbers (floats) that specify each bin delimiter.

    Returns:
        list of numbers (floats) consisting in the list of bin centers and contour.
    
    """
    centers = [bins[0]] # Adding initial contour
    l = len(bins)
    for i in range(l-1):
        centers.append((bins[i]+bins[i+1])/2) # Adding centers
    centers.append(bins[-1]) # Adding final contour
    return centers


def __relu_func(vals):
    """Element-wise maximum between the value and zero.

    Args:
        vals (ndarray): list of numbers to compute its element-wise maximum.
    
    Returns:
        array_like consisting in the element-wise maximum vector of the given values.
    """
    return np.maximum(vals, np.zeros_like(vals))


def theory_wigner_law(val, beta):
    """Computes the theoretical Wigner's semicircle law on a given point.

    Args:
        val (float): point whose evaluation is required.
        beta (int): integer representing type of matrix entries. beta=1 if real
            entries are used (GOE), beta=2 if they are complex (GUE) or beta=4
            if they are quaternions (GSE). 
    
    Returns:
        number (float) which is image of the given value evaluated on Wigner's
        semicircle law.
    """
    radius = 2*math.sqrt(beta)
    if abs(val) >= radius:
        return 0
    return 2*math.sqrt(radius**2 - val**2)/(math.pi*radius**2)


def wigner_semicircular_law(ensemble='goe', n_size=1000, bins=100, interval=None,
                            density=False, limit_pdf=False, savefig_path=None):
    """Calculates and plots Wigner's Semicircle Law using Gaussian Ensemble.

    Calculates and plots Wigner's Semicircle Law using Gaussian Ensemble random matrices.
    Gaussian (Hermite) ensemble has improved routines (using tridiagonal forms and Sturm
    sequences) to avoid calculating the eigenvalues, so the histogram
    is built using certain techniques to boost efficiency.

    Args:
        ensemble ('goe', 'gue' or 'gse', default='goe'): ensemble to draw the
            random matrices to study Wigner's Law.
        n_size (int, default=1000): random matrix size n times n.
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        limit_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
            If set to False, just the empirical histogram is shown. This parameter is only
            considered when the argument 'density' is set also to True.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    # pylint: disable=too-many-arguments
    if n_size<1:
        raise ValueError("matrix size must be positive")

    if ensemble == 'goe':
        beta = 1
        if interval is None:
            interval = (-2,2)
    elif ensemble == 'gue':
        beta = 2
        if interval is None:
            interval = (-3,3)
    elif ensemble == 'gse':
        beta = 4
        if interval is None:
            interval = (-4,4)
    else:
        raise ValueError("ensemble not supported")
    
    ens = GaussianEnsemble(beta=beta, n=n_size, use_tridiagonal=True)

    # Wigner eigenvalue normalization constant
    norm_const = 1/np.sqrt(n_size/2)

    observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                     density=density, norm_const=norm_const)
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    # Plotting theoretical graphic
    if limit_pdf and density:
        centers = __get_bins_centers_and_contour(bins)
        expected_frec = [theory_wigner_law(cent, beta) for cent in centers]
        plt.plot(centers, expected_frec, color='red', linewidth=2)

    plt.title("Eigenvalue density histogram", fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("density")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()



def theory_marchenko_pastur(vals, ratio, lambda_minus, lambda_plus, beta):
    """Computes the theoretical Wigner's semicircle law on a given point or points.

    Args:
        vals (ndarray): numpy array of numbers whose evaluation is required.
        ratio (float): ratio between the matrix size. ratio is equal to p/n,
            where p is the number of rows and n is the number of columns of
            the matrix that generates a Wishart matrix. 'p' is also known as
            the degrees of freedom and 'n' as the sample size.
        lambda_minus (float): lower limit of the Marchenko-Pastur distribution.
        lambda_plus (float): upper limit of the Marchenko-Pastur distribution.
        beta (int): integer representing type of matrix entries. beta=1 if real
            entries are used (WRE), beta=2 if they are complex (WCE) or beta=4
            if they are quaternions (WQE). Beta is considered as the variance
            of the matrix entries.
    
    Returns:
        array_like (ndarray) which is the image of the given value (or values)
        evaluated on Marchenko-Pastur Law.
    """
    var = beta
    return np.sqrt(__relu_func(lambda_plus-vals)*__relu_func(vals-lambda_minus)) \
          / (2*np.pi*ratio*var*vals)


def marchenko_pastur_law(ensemble='wre', p_size=3000, n_size=10000, bins=100, interval=None,
                         density=False, limit_pdf=False, savefig_path=None):
    """Computes and plots Marchenko-Pastur Law using Wishart Ensemble random matrices.

    Calculates and plots Marchenko-Pastur Law using Wishart Ensemble random matrices.
    Wishart (Laguerre) ensemble has improved routines (using tridiagonal forms and Sturm
    sequences) to avoid calculating the eigenvalues, so the histogram
    is built using certain techniques to boost efficiency. This optimization is only used
    when the ratio p_size/n_size is less or equal than 1.

    Args:
        ensemble ('wre', 'wce' or 'wqe', default='wre'): ensemble to draw the
            random matrices to study Marchenko-Pastur Law.
        p_size (int, default=3000): number of rows of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        n_size (int, default=10000): number of columns of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        limit_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
            If set to False, just the empirical histogram is shown. This parameter is only
            considered when the argument 'density' is set also to True.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    # pylint: disable=too-many-arguments
    if n_size<1 or p_size<1:
        raise ValueError("matrix size must be positive")
    
    if interval and interval[0] == 0:
        print("Warning: setting the beginning of the interval to zero may generate numerical errors.")
        print(f"Setting interval to (-0.01, {interval[1]})")
        interval = (-0.01, interval[1])

    try:
        beta = ["wre", "wce", None, "wqe"].index(ensemble) + 1
    except ValueError:
        raise ValueError("ensemble not supported: "+str(ensemble))
    # calculating constants depending on matrix sizes
    ratio = p_size/n_size
    lambda_plus = beta*(1 + np.sqrt(ratio))**2
    lambda_minus = beta*(1 - np.sqrt(ratio))**2
    use_tridiag = (ratio <= 1)

    # computing interval according to the matrix size ratio and support
    if interval is None:
        if ratio <= 1:
            interval = (lambda_minus, lambda_plus)
        else:
            interval = (min(-0.05, lambda_minus), lambda_plus)
    
    ens = WishartEnsemble(beta=beta, p=p_size, n=n_size, use_tridiagonal=use_tridiag)

    # Wigner eigenvalue normalization constant
    norm_const = 1/n_size

    observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                     density=density, norm_const=norm_const)
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    # Plotting theoretical graphic
    if limit_pdf and density:
        centers = np.array(__get_bins_centers_and_contour(bins))
        expected_frec = theory_marchenko_pastur(centers, ratio, lambda_minus,
                                                lambda_plus, beta)
        plt.plot(centers, expected_frec, color='red', linewidth=2)

    plt.title("Eigenvalue density histogram", fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("density")
    if ratio > 1:
        if limit_pdf and density:
            ylim_vals = expected_frec
        else:
            ylim_vals = observed
        try:
            plt.ylim(0, np.max(ylim_vals)+0.25*np.max(ylim_vals))
        except ValueError:
            second_highest_val = np.partition(ylim_vals.flatten(), -2)[-2]
            plt.ylim(0, second_highest_val+0.25*second_highest_val)

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()



def tracy_widom_law(ensemble='goe', n_size=100, times=1000, bins=100, interval=None,
                    density=False, limit_pdf=False, savefig_path=None):
    """Calculates and plots Tracy-Widom Law using Gaussian Ensemble.

    Calculates and plots Tracy-Widom Law using Gaussian Ensemble random matrices.
    Because we need to obtain the largest eigenvalue of each sampled random matrix,
    we need to sample a certain amount them. For each random matrix sammpled, its
    largest eigenvalue is calcualted in order to simulate its density.

    Args:
        ensemble ('goe', 'gue' or 'gse', default='goe'): ensemble to draw the
            random matrices to study Wigner's Law.
        n_size (int, default=100): random matrix size n times n.
        times (int, default=1000): number of times to sample a random matrix.
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        limit_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
            If set to False, just the empirical histogram is shown. This parameter is only
            considered when the argument 'density' is set also to True.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    # pylint: disable=too-many-arguments
    if n_size<1 or times<1:
        raise ValueError("matrix size or number of repetitions must be positive")
    
    try:
        beta = ["goe", "gue", None, "gse"].index(ensemble) + 1
    except ValueError:
        raise ValueError("ensemble not supported: "+str(ensemble))

    ens = GaussianEnsemble(beta=beta, n=n_size, use_tridiagonal=False)

    eigvals = np.asarray([])
    for _ in range(times):
        vals = ens.eigvals()
        eigvals = np.append(eigvals, vals.max())
        ens.sample()

    # Tracy-Widom eigenvalue distr. normalization constants
    eigval_scale = 1
    size_scale = 1
    if ensemble == 'gue':
        eigval_scale = 1/np.sqrt(2)
    if ensemble == 'gse':
        eigval_scale = size_scale = 1/np.sqrt(2)
        n_size *= 2
    eigvals = size_scale*(n_size**(1/6))*(eigval_scale*eigvals - (2*np.sqrt(n_size)))

    if interval is None:
        xmin=eigvals.min()
        xmax=eigvals.max()
        interval=(xmin, xmax)

    # using numpy to obtain histogram in the given interval and no. of bins
    observed, bins = np.histogram(eigvals, bins=bins, range=interval, density=density)
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    # Plotting theoretical graphic
    if limit_pdf and density:
        centers = __get_bins_centers_and_contour(bins)
        tw_approx = TW_Approximator(beta=beta)
        expected_frec = tw_approx.pdf(centers)
        plt.plot(centers, expected_frec, color='red', linewidth=2)

    plt.title("Eigenvalue density histogram", fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("density")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()



def theory_manova_spectrum_distr(vals, a, b, lambda_minus, lambda_plus):
    """Computes the theoretical Manova spectrum limiting distribution law
    on a given point or points.

    Args:
        vals (ndarray): numpy array of numbers whose evaluation is required.
        a (float): parameter of the theoretical analytical function.
        b (float): parameter of the theoretical analytical function.
        lambda_minus (float): lower limit of the Manova spectrum distribution.
        lambda_plus (float): upper limit of the Manova spectrum distribution.
        beta (int): integer representing type of matrix entries. beta=1 if real
            entries are used (MRE), beta=2 if they are complex (MCE) or beta=4
            if they are quaternions (MQE). Beta is considered as the variance
            of the matrix entries.
    
    Returns:
        array_like (ndarray) which is the image of the given value (or values)
        evaluated on the Manova spectrum limiting distribution.
    """
    return (a+b) * np.sqrt(__relu_func(lambda_plus-vals)*__relu_func(vals-lambda_minus)) \
        / (2*np.pi*vals*(1-vals))


def manova_spectrum_distr(ensemble='mre', m_size=1000, n1_size=3000, n2_size=3000,
                          bins=100, interval=None, density=False, limit_pdf=False,
                          savefig_path=None):
    """Computes and plots Manova spectrum limiting and analytical distribution.

    Calculates and plots Manova spectrum limiting Law using Manova Ensemble random matrices.

    Args:
        ensemble ('mre', 'mce' or 'mqe', default='mre'): Manova Ensemble to draw the
            random matrices to study limiting distribution.
        m_size (int, default=1000): number of rows of the two Wishart Ensemble matrices that
            generates the matrix of the corresponding ensemble.
        n1_size (int, default=3000): number of columns of the first Wishart Ensemble matrix
            that generates the matrix of the corresponding ensemble.
        n2_size (int, default=3000): number of columns of the second Wishart Ensemble matrix
            that generates the matrix of the corresponding ensemble.    
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        limit_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
            If set to False, just the empirical histogram is shown. This parameter is only
            considered when the argument 'density' is set also to True.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Laszlo, L. and Farrel, B.
            "Local Eigenvalue Density for General MANOVA Matrices".
            Journal of Statistical Physics. 152.6 (2013): 1003-1032.
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    if m_size<1 or n1_size<1 or n2_size<1:
        raise ValueError("matrix size must be positive")
    
    try:
        beta = ["mre", "mce", None, "mqe"].index(ensemble) + 1
    except ValueError:
        raise ValueError("ensemble not supported: "+str(ensemble))
    
    ens = ManovaEnsemble(beta=beta, m=m_size, n1=n1_size, n2=n2_size)

    a = n1_size/m_size
    b = n2_size/m_size
    if a <= 1 or b <= 1:
        print("Warning: sample size ('n1_size' or 'n2_size') too small compared \
               to degrees of freedom ('m_size'). It may cause numerical instability.")
    lambda_term1 = np.sqrt((a/(a+b)) * (1 - (1/(a+b))))
    lambda_term2 = np.sqrt((1/(a+b)) * (1 - (a/(a+b))))
    lambda_minus = (lambda_term1 - lambda_term2)**2
    lambda_plus = (lambda_term1 + lambda_term2)**2

    if interval is None:
        interval = [lambda_minus, lambda_plus]
        if a <= 1:
            interval[0] = min(-0.05, lambda_minus)
        if b <= 1:
            interval[1] = max(lambda_plus, 1.05)
        interval = tuple(interval)

    #norm_const = m_size
    observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                     density=density, avoid_img=True) #norm_const = norm_const
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    # Plotting theoretical graphic
    if limit_pdf and density:
        centers = np.array(__get_bins_centers_and_contour(bins))
        expected_frec = theory_manova_spectrum_distr(centers, a, b, 
                                                    lambda_minus, lambda_plus)
        plt.plot(centers, expected_frec, color='red', linewidth=2)

    plt.title("Eigenvalue density histogram", fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("density")
    if a <= 1 or b <= 1:
        if limit_pdf and density:
            ylim_vals = expected_frec
        else:
            ylim_vals = observed
        try:
            plt.ylim(0, np.max(ylim_vals)+0.25*np.max(ylim_vals))
        except ValueError:
            second_highest_val = np.partition(ylim_vals.flatten(), -2)[-2]
            plt.ylim(0, second_highest_val+0.25*second_highest_val)

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()
