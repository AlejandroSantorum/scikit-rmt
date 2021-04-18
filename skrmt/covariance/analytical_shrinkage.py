import numpy as np
from numpy.matlib import repmat

class AnalyticalShrinkage:

    
    def estimate(self, X, k=None):
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
        eigvals = eigvals[max(0,p-n):p]
        L = repmat(eigvals, min(p,n), 1).T
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
        sigma_tilde = np.matmul(np.matmul(eigvects, np.diag(d_tilde).T), eigvects.T)
        return sigma_tilde





def test1():
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    analsh = AnalyticalShrinkage()
    sigma_tilde = analsh.estimate(X)

    print(sigma_tilde)






        
if __name__ == "__main__":
    '''
    import pandas as pd

    df = pd.read_csv('wdbc.data')

    print("Values shape", df.values[:,:-1].shape)

    X = np.array(df.values[:,:-1])
    print(X.shape)
    Y = np.array(df.values[:,-1])

    analsh = AnalyticalShrinkage()
    analsh.estimate_vec(X)
    '''

    test1()






