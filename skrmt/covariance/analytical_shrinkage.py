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
        #eigvals = eigvals[max(0,p-n):p]
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



def my_test1():
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    analsh = AnalyticalShrinkage()
    sigma_tilde = analsh.estimate(X)

    sol = np.array([[0.551752555904758, -0.176171945854672, -0.127673277267705],
                    [-0.176171945854672, 0.317339265808824, -0.236795281588382],
                    [-0.127673277267705, -0.236795281588382, 0.474063174598454]])

    print("======== My solution ========")
    print(sigma_tilde)
    print("======== Official solution ========")
    print(sol)



def my_test2():
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])

    analsh = AnalyticalShrinkage()
    sigma_tilde = analsh.estimate(X)

    sol = np.array([[3.84630121492766, -1.3927903545004, -1.42093779975966],
                    [-1.3927903545004, 7.91982043683479, -1.3927903545004],
                    [-1.42093779975966, -1.3927903545004, 3.84630121492766]])
    
    print("======== My solution ========")
    print(sigma_tilde)
    print("======== Official solution ========")
    print(sol)


        
if __name__ == "__main__":
    import sys
    if sys.argv[1] == '1':
        my_test1()
    elif sys.argv[1] == '2':
        my_test2()
    else:
        my_test2()






