"""Estimator Module

This module contains the implementation of several covariance matrix
estimators given n observations of size p.
"""

import numpy as np


def sample_estimator(data_mtx, shrink=None):
    """Estimates sample covariance matrix.

    Estimates sample covariance matrix given data matrix of size nxp.

    Args:
        data_mtx: data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.
        shrink: number of degrees of freedom to substract.

    Returns:
        numpy array representing sample covariance matrix.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
        Numpy API documentation. numpy.cov
            https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    """
    n_size = data_mtx.shape[0]

    if shrink is None:
        # demean data matrix
        data_mtx = data_mtx - data_mtx.mean(axis=0)
        # subtract one degree of freedom
        shrink=1
    # effective sample size
    n_eff=n_size-shrink
    # get sample covariance estimator
    sigma_tilde = np.matmul(data_mtx.T, data_mtx)/n_eff
    return sigma_tilde



def fsopt_estimator(data_mtx, sigma):
    """Estimates FSOpt estimator.

    Estimates finite-sample optimal estimator (FSOpt estimator), also
    written as S^*. It replaces eigenvalues of sample covariance matrix
    for a new term derived from original population covariance matrix.
    This estimator is not observable in the reality, only using Monte
    Carlo simulations.

    Args:
        data_mtx: data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.
        sigma: population covariance matrix.

    Returns:
        numpy array representing sample covariance matrix.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
    """
    #n, p = data_mtx.shape

    sample = sample_estimator(data_mtx)

    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    d_star = np.array([np.matmul(np.matmul(vec, sigma), vec.T) for vec in eigvects.T])

    # compute finite-sample optimal (FSOpt) nonlinear shrinkage estimator
    sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(d_star)), eigvects.T)
    return sigma_tilde



def linear_shrinkage_estimator(data_mtx, shrink=None):
    """Estimates linear shrinkage estimator.

    When the dimension p is greater than the number of observations n,
    sample covariance matrix S is not even invertible. When the ratio p/n
    is smaller than one but not negligible, then S is invertible but it
    is not well-conditoned, i.e., inverting it amplifies estimation error.
    To get a structured well-conditioned estimator is to force the
    condition that variances are equal and covariances zero.
    Linear shrinkage estimator is a weighted average of this structured
    estimator and the sample covariance matrix.

    Args:
        data_mtx: data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.
        shrink: shrinkage value. If it is not provided, this routine
            calculates the shrinkage value selected by Ledoit and Wolf
            (see reference).

    Returns:
        numpy array representing sample covariance matrix.

    References:
        Ledoit, O. and Wolf, M.
            "A well-conditioned estimator for large-dimensional covariance matrices".
            Journal of Multivariate Analysis. 88 (2004): 365-411
    """
    n_size, p_size = data_mtx.shape

    # demean data matrix
    data_mtx = data_mtx - data_mtx.mean(axis=0)
    # compute sample covariance matrix
    sample = np.matmul(data_mtx.T, data_mtx)/n_size # WATCH OUT: it is not the 'effective size'

    # compute prior
    meanvar = np.mean(np.diag(sample)) # = trace/p = sum(eigvals)/p
    prior = meanvar * np.identity(p_size)

    # use specified shrinkage value
    if shrink:
        shrinkage = shrink
    # compute shrinkage parameters
    else:
        aux_mtx = data_mtx**2
        phi_mat = np.matmul(aux_mtx.T, aux_mtx)/n_size - sample**2
        phi = np.sum(phi_mat)
        # np norm by default calculates frobenius norm for matrices and L2-norm for vects
        gamma = np.linalg.norm(sample-prior)**2
        # compute shrinkage constant
        kappa= phi/gamma
        shrinkage=max(0, min(1, kappa/n_size))

    # compute shrinkage estimator
    sigma_tilde = shrinkage*prior + (1-shrinkage)*sample
    return sigma_tilde



def analytical_shrinkage_estimator(data_mtx, shrink=None):
    """Estimates analytical shrinkage estimator.
    This estimator combines the best qualities of three different estimators:
    the speed of linear shrinkage, the accuracy of the well-known QuEST function
    and the transparency of the routine NERCOME. This estimator achieves this
    goal through nonparametric kernel estimation of the limiting spectral
    density of the sample eigenvalues and its Hilbert transform.

    Args:
        data_mtx (numpy array): data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.
        shrink (integer): number of degrees of freedom to substract.

    Returns:
        numpy array representing sample covariance matrix.
    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
    """
    n_size, p_size = data_mtx.shape

    if shrink is None:
        # demean data matrix
        data_mtx = data_mtx - data_mtx.mean(axis=0)
        # subtract one degree of freedom
        shrink=1
    # effective sample size
    n_size=n_size-shrink

    # get sample eigenvalues and eigenvectors, and sort them in ascending order
    sample = np.matmul(data_mtx.T, data_mtx)/n_size
    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    # compute analytical nonlinear shrinkage kernel formula
    #eigvals = eigvals[max(0,p-n):p]
    repmat_eigs = np.tile(eigvals, (min(p_size,n_size), 1)).T
    h_list = n_size**(-1/3) * repmat_eigs.T

    eigs_div = np.divide((repmat_eigs-repmat_eigs.T), h_list)

    f_tilde=(3/4/np.sqrt(5))*np.mean(np.divide(np.maximum(1-eigs_div**2/5, 0), h_list), axis=1)

    hilbert_temp = (-3/10/np.pi)*eigs_div + \
                    (3/4/np.sqrt(5)/np.pi)*(1-eigs_div**2/5)*\
                        np.log(abs((np.sqrt(5)-eigs_div)/(np.sqrt(5)+eigs_div)))
    hilbert_temp[abs(eigs_div)==np.sqrt(5)] = (-3/10/np.pi) * eigs_div[abs(eigs_div)==np.sqrt(5)]
    hilbert = np.mean(np.divide(hilbert_temp, h_list), axis=1)

    # if p <= n: (we could improve it to support p>n case)
    d_tilde = np.divide(eigvals,
                        (np.pi*(p_size/n_size)*eigvals*f_tilde)**2 + \
                            (1-(p_size/n_size)-np.pi*(p_size/n_size)*eigvals*hilbert)**2
                       )

    # compute analytical nonlinear shrinkage estimator (sigma_tilde)
    return np.matmul(np.matmul(eigvects, np.diag(d_tilde)), eigvects.T)



def empirical_bayesian_estimator(data_mtx):
    """Estimates empirical bayesian estimator.

    The empirical bayesian estimator is a linear combination of sample
    covariance matrix and the identity matrix. This estimator was
    introduced by Haff in 1980, and he suggested that this estimator
    should be used when the criterion is the mean squared error.

    Args:
        data_mtx: data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.

    Returns:
        numpy array representing sample covariance matrix.

    References:
        Haff, L.R.
            "Empirical Bayes estimation of the multivariate normal covariance matrix".
            Annals of Statistics. 8 (1980): 586-597
        Ledoit, O. and Wolf, M.
            "A well-conditioned estimator for large-dimensional covariance matrices".
            Journal of Multivariate Analysis. 88 (2004): 365-411
    """
    n_size, p_size = data_mtx.shape

    sample = sample_estimator(data_mtx)

    # we consider n >= p
    det = np.linalg.det(sample)
    m_eb = det**(1/p_size)

    prior = m_eb * np.identity(p_size)
    sigma_tilde = ((p_size*n_size - 2*n_size -2)/(p_size*n_size**2))*prior + \
                    n_size/(n_size+1)*sample
    return sigma_tilde



def minimax_estimator(data_mtx):
    """Estimates minimax estimator.

    Minimax estimator has the lowest worst-case error. The minimax
    criterion is sometimes criticized as overly pessimistic, since
    it looks at the worst case only. This estimator preserves sample
    eigenvectors and replaces sample eigenvalues by (n*lambda_i)/(n+p+1-2i),
    where lambda_i (i=1,...,p) are the sample eigenvalues sorted in
    descending order.

    Args:
        data_mtx: data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.

    Returns:
        numpy array representing sample covariance matrix.

    References:
        Haff, L.R.
            "Estimation of a covariance matrix under Stein’s loss".
            Annals of Statistics. 13.4 (1985): 1581-1591
        Ledoit, O. and Wolf, M.
            "A well-conditioned estimator for large-dimensional covariance matrices".
            Journal of Multivariate Analysis. 88 (2004): 365-411
    """
    n_size, p_size = data_mtx.shape

    # get sample eigenvalues and eigenvectors, and sort them in descending order
    sample = sample_estimator(data_mtx)
    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals) # ascending order
    order = order[::-1] # descending order (reversing it)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    # calculating new eigenvalues
    new_eigvals = [n_size*val/(n_size+p_size+1-2*i) for (i, val) in enumerate(eigvals)]

    # compute minimax estimator by replacing eigenvalues
    sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(new_eigvals)), eigvects.T)
    return sigma_tilde
