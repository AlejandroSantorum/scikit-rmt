
import numpy as np


def sample_estimator(X, k=None):
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
    n, p = X.shape

    sample = sample_estimator(X)

    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    d_star = np.array([np.matmul(np.matmul(vec, Sigma), vec.T) for vec in eigvects.T])

    # compute finite-sample optimal (FSOpt) nonlinear shrinkage estimator
    sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(d_star)), eigvects.T)
    sigma_tilde



def linear_shrinkage_estimator(X, shrink=None):
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
    n, p = X.shape

    sample = sample_estimator(X)

    # we consider n >= p
    d = np.linalg.det(sample)
    m_eb = d**(1/p)
    
    prior = m_eb * np.identity(p)
    sigma_tilde = ((p*n - 2*n -2)/(p*n**2))*prior + n/(n+1)*sample
    return sigma_tilde






