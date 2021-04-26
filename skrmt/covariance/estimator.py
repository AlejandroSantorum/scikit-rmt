"""Estimator Module

This module contains the implementation of several covariance matrix
estimators given n observations of size p.
"""

import numpy as np


def sample_estimator(X, k=None):
    """Estimates sample covariance matrix.

    Estimates sample covariance matrix given data matrix X of size nxp.

    Args:
        X: data matrix containing n observations of size p, i.e.,
            X is a n times p matrix.
        k: number of degrees of freedom to substract.
    
    Returns:
        numpy array representing sample covariance matrix.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
        Numpy API documentation. numpy.cov
            https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    """
    n, p = X.shape

    if k is None:
        # demean data matrix
        X = X - X.mean(axis=0)
        # subtract one degree of freedom 
        k=1
    # effective sample size
    n=n-k
    # get sample covariance estimator
    sigma_tilde = np.matmul(X.T, X)/n
    return sigma_tilde



def FSOpt_estimator(X, Sigma):
    """Estimates FSOpt estimator.

    Estimates finite-sample optimal estimator (FSOpt estimator), also
    written as S^*. It replaces eigenvalues of sample covariance matrix
    for a new term derived from original population covariance matrix.
    This estimator is not observable in the reality, only using Monte
    Carlo simulations.

    Args:
        X: data matrix containing n observations of size p, i.e.,
            X is a n times p matrix.
        Sigma: population covariance matrix.
    
    Returns:
        numpy array representing sample covariance matrix.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
    """
    n, p = X.shape

    sample = sample_estimator(X)

    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    d_star = np.array([np.matmul(np.matmul(vec, Sigma), vec.T) for vec in eigvects.T])

    # compute finite-sample optimal (FSOpt) nonlinear shrinkage estimator
    sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(d_star)), eigvects.T)
    return sigma_tilde



def linear_shrinkage_estimator(X, shrink=None):
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
        X: data matrix containing n observations of size p, i.e.,
            X is a n times p matrix.
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
    n, p = X.shape

    # demean data matrix
    X = X - X.mean(axis=0)
    # compute sample covariance matrix
    sample = np.matmul(X.T, X)/n # WATCH OUT: it is not the 'effective size'

    # compute prior
    meanvar = np.mean(np.diag(sample)) # = trace/p = sum(eigvals)/p
    prior = meanvar * np.identity(p)
    
    # use specified shrinkage value
    if shrink:
        shrinkage = shrink;
    # compute shrinkage parameters
    else:
        Y = X**2
        phiMat = np.matmul(Y.T, Y)/n - sample**2
        phi = np.sum(phiMat)
        # np norm by default calculates frobenius norm for matrices and L2-norm for vects
        gamma = np.linalg.norm(sample-prior)**2;
        # compute shrinkage constant
        kappa= phi/gamma
        shrinkage=max(0, min(1, kappa/n))

    # compute shrinkage estimator
    sigma_tilde = shrinkage*prior + (1-shrinkage)*sample
    return sigma_tilde



def analytical_shrinkage_estimator(X, k=None):
    """Estimates analytical shrinkage estimator.

    This estimator combines the best qualities of three different estimators:
    the speed of linear shrinkage, the accuracy of the well-known QuEST function
    and the transparency of the routine NERCOME. This estimator achieves this
    goal through nonparametric kernel estimation of the limiting spectral 
    density of the sample eigenvalues and its Hilbert transform.

    Args:
        X: data matrix containing n observations of size p, i.e.,
            X is a n times p matrix.
        k: number of degrees of freedom to substract.
    
    Returns:
        numpy array representing sample covariance matrix.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
    """
    n, p = X.shape

    if k is None:
        # demean data matrix
        X = X - X.mean(axis=0)
        # subtract one degree of freedom 
        k=1
    # effective sample size
    n=n-k

    # get sample eigenvalues and eigenvectors, and sort them in ascending order
    sample = np.matmul(X.T, X)/n
    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    # compute analytical nonlinear shrinkage kernel formula
    #eigvals = eigvals[max(0,p-n):p]
    L = np.tile(eigvals, (min(p,n), 1)).T
    h=n**(-1/3)
    H=h*L.T

    x = np.divide((L-L.T), H)

    f_tilde=(3/4/np.sqrt(5))*np.mean(np.divide(np.maximum(1-x**2/5, 0), H), axis=1)

    hilbert_temp = (-3/10/np.pi)*x + (3/4/np.sqrt(5)/np.pi)*(1-x**2/5)*np.log(abs((np.sqrt(5)-x)/(np.sqrt(5)+x)))
    hilbert_temp[abs(x)==np.sqrt(5)] = (-3/10/np.pi) * x[abs(x)==np.sqrt(5)]
    hilbert = np.mean(np.divide(hilbert_temp, H), axis=1)

    # if p <= n: (we could improve it to support p>n case)
    denom = (np.pi*(p/n)*eigvals*f_tilde)**2 + (1-(p/n)-np.pi*(p/n)*eigvals*hilbert)**2
    d_tilde = np.divide(eigvals, denom)

    # compute analytical nonlinear shrinkage estimator
    sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(d_tilde)), eigvects.T)
    return sigma_tilde



def empirical_bayesian_estimator(X):
    """Estimates empirical bayesian estimator.

    The empirical bayesian estimator is a linear combination of sample
    covariance matrix and the identity matrix. This estimator was
    introduced by Haff in 1980, and he suggested that this estimator
    should be used when the criterion is the mean squared error.

    Args:
        X: data matrix containing n observations of size p, i.e.,
            X is a n times p matrix.
    
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
    n, p = X.shape

    sample = sample_estimator(X)

    # we consider n >= p
    d = np.linalg.det(sample)
    m_eb = d**(1/p)
    
    prior = m_eb * np.identity(p)
    sigma_tilde = ((p*n - 2*n -2)/(p*n**2))*prior + n/(n+1)*sample
    return sigma_tilde



def minimax_estimator(X):
    """Estimates minimax estimator.

    Minimax estimator has the lowest worst-case error. The minimax
    criterion is sometimes criticized as overly pessimistic, since
    it looks at the worst case only. This estimator preserves sample
    eigenvectors and replaces sample eigenvalues by (n*lambda_i)/(n+p+1-2i),
    where lambda_i (i=1,...,p) are the sample eigenvalues sorted in
    descending order.

    Args:
        X: data matrix containing n observations of size p, i.e.,
            X is a n times p matrix.
    
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
    n, p = X.shape

    # get sample eigenvalues and eigenvectors, and sort them in descending order
    sample = sample_estimator(X)
    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals) # ascending order
    order = order[::-1] # descending order (reversing it)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    # calculating new eigenvalues
    new_eigvals = [n*val/(n+p+1-2*i) for (i, val) in enumerate(eigvals)]

    # compute minimax estimator by replacing eigenvalues
    sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(new_eigvals)), eigvects.T)
    return sigma_tilde




