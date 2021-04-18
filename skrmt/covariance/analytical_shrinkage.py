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
        eigvects = eigvects[order]

        # choose the global bandwidth
        h = n**(-1/3)

        # estimate the spectral density with the Epanechnikov kernel
        #### rep_eigvals = np.repeat(eigvals, p)
        #### H = h*rep_eigvals

        # specify the locally adaptive bandwidth
        H = h * eigvals

        # estimate the spectral density with the Epanechnikov kernel
        f_tilde = np.zeros(p)
        for i in range(p):
            for j in range(p):
                second_factor = 1 - 0.2*((eigvals[i] - eigvals[j])/H[j])**2
                f_tilde[i] += 3/(4*np.sqrt(5)*H[j]) * max(second_factor, 0)
            f_tilde[i] /= p

        # estimate the Hilbert transform
        Hilbert = np.zeros(p)
        for i in range(p):
            for j in range(p):
                sum1 = -3*(eigvals[i]-eigvals[j])/(10*np.pi*H[j]**2)
                factor1 = 3/(4*np.sqrt(5)*np.pi*H[j])*(1 - 0.2*((eigvals[i]-eigvals[j])/H[j])**2)
                factor2 = np.log(np.abs((np.sqrt(5)*H[j]-eigvals[i]+eigvals[j])/(np.sqrt(5)*H[j]+eigvals[i]-eigvals[j])))
                Hilbert[i] += (sum1 + factor1*factor2)
            Hilbert[i] /= p
        
        print(Hilbert[-1])

        # compute the asymptotically optimal nonlinear shrinkage
        d_tilde = np.zeros(p)
        for i in range(p):
            den = (np.pi*(p/n)*eigvals[i]*f_tilde[i])**2 + (1-(p/n)-np.pi*(p/n)*eigvals[i]*Hilbert[i])**2
            d_tilde[i] = eigvals[i]/den
        
        S_tilde = np.zeros((p,p))
        for i in range(p):
            S_tilde += d_tilde[i] * np.matmul(eigvects[:,i].T, eigvects[:,i].T)
        
        return S_tilde

    
    def estimate_vec(self, X, k=None):
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
    sigma_tilde = analsh.estimate_vec(X)

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






