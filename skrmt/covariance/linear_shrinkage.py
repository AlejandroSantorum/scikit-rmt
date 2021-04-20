import numpy as np

class LinearShrinkage:

    def estimate(self, X, shrink=None):
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
        self.sigma_tilde = shrinkage*prior + (1-shrinkage)*sample
        return self.sigma_tilde



def my_test1():
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    linsh = LinearShrinkage()
    sigma_tilde = linsh.estimate(X)

    sol = np.array([[0.571596585433017, -0.119074762494837, -0.142889714993804],
                    [-0.119074762494837, 0.285817155445408, -0.089306071871128],
                    [-0.142889714993804, -0.089306071871128, 0.387030703566020]])

    print("======== My solution ========")
    print(sigma_tilde)
    print("======== Official solution ========")
    print(sol)


def my_test2():
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])

    linsh = LinearShrinkage()
    sigma_tilde = linsh.estimate(X)

    sol = np.array([[4.157119427352015, -0.175394418448355, -0.114387664205449],
                    [-0.175394418448355, 4.734650034184860, -0.175394418448355],
                    [-0.114387664205449, -0.175394418448355, 4.157119427352015]])
    
    print("======== My solution ========")
    print(sigma_tilde)
    print("======== Official solution ========")
    print(sol)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        my_test2()
    elif sys.argv[1] == '1':
        my_test1()
    elif sys.argv[1] == '2':
        my_test2()
    else:
        my_test2()
    

