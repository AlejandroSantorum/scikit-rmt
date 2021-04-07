
import numpy as np



def tridiag_eigval_neg(T):
    n = len(T) # number of rows = number of columns
    R = np.zeros(n)

    R[0] = T[n-1][n-1] # R[0] = a_1 by default
    for i in range(1, n): # O(n)
        R[i] = T[n-i-1][n-i-1] - ((T[n-i-1][n-i]**2)/R[i-1])
    
    # number of negative values in R sequence
    return (R<0).sum()



def tridiag_eigval_hist(T, bins=100, interval=(-2,2), norm=False):

    if not isinstance(interval, tuple):
            raise ValueError("interval argument must be a tuple")

    if isinstance(bins, int):
        # calculating bin delimiters
        K = np.linspace(interval[0], interval[1], num=bins+1) # O(m)
    elif isinstance(bins, (tuple, list)):
        # bin delimiters directly introduced
        K = np.array(bins)
    else:
        raise ValueError("bins argument must be a tuple, list or integer")

    n = int(len(T)) # number of rows = number of columns
    # + 2 because of (-inf, interval[0]) and (interval[1], inf) intervals
    histogram = np.zeros(len(K))
    
    # Building matrix A in linear time
    diag = np.arange(n)
    diag_1 = diag[:-1] # = np.arange(n-1)
    A = np.zeros((n,n))
    A[diag_1, diag_1+1] = T[diag_1, diag_1+1]
    A[diag_1+1, diag_1] = T[diag_1+1, diag_1]
    
    prev = 0
    for i in range(bins+1): # O(m)
        A[diag, diag] = T[diag, diag] - K[i]
        current = tridiag_eigval_neg(A) # O(n)
        histogram[i] = current - prev
        prev = current
    
    if norm:
        # dont need (-inf, interval[0]) interval
        histogram[1:] = histogram[1:] / (sum(histogram[1:]) * np.diff(K))
    
    # dont want (-inf, interval[0]) interval
    return histogram[1:], K