from scipy.spatial.distance import cdist
import numpy as np

class Laplacian:
    def __init__(self, gamma=1.):
        self.gamma = gamma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        dists = cdist(X , Y , 'minkowski', p=1)
        K = np.exp(-self.gamma*(dists))
        return K
    
class RBF:
    def __init__(self, gamma=1.):
        self.gamma = gamma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        dists = cdist(X , Y , metric="sqeuclidean")
        K = np.exp(-self.gamma*(dists))
        return K
    
class Linear:
    def __init__(self):
        pass
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        K = np.inner(X, Y)
        return K ## Matrix of shape NxM
    
class chi2:
    def __init__(self, gamma):
        self.gamma = gamma
            
    def kernel(self, X, Y):
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        n_features = X.shape[1]
        K = np.zeros((n_samples_X, n_samples_Y), dtype=X.dtype)
        
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                res = 0
                for k in range(n_features):
                    denom = (X[i, k] - Y[j, k])
                    nom = (X[i, k] + Y[j, k])
                    if nom != 0:
                        res  += denom * denom / nom
                K[i, j] = -res
        K *= self.gamma
        return np.exp(K, K)