import numpy as np


class PCA:

    def __init__(self, n_components):

        self.n_components = n_components
        self.e_values_ = None
        self.e_values_ratio_ = None

    def fit(self, X, scale=False):
        n_samples, n_features = X.shape

        if scale:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        print('computing pca with SVD')
        u, s, v = np.linalg.svd(X)

        _e_vectors = u[:, :self.n_components]
        _e_vectors *= s[:self.n_components]

        self._left_vectors = v[:self.n_components]
        e_values_ = (s ** 2) / n_samples
        self.e_values_ = e_values_[:self.n_components]

        total_var = float(e_values_.sum())
        self.e_values_ratio_ = [float(e) / total_var for e in self.e_values_]

        return _e_vectors

    def transform(self, X, scale=False):
        if scale:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        _e_vectors = np.dot(X, self._left_vectors.T)
        return _e_vectors
