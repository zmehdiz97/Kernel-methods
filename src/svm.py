import numpy as np
import cvxopt
import typing
from .kernels import *

class SVM:
    """Class implementing a Support Vector Machine: instead of minimising the primal function
        L_P(w, b, lambda_mat) = 1/2 ||w||^2 - sum_i{lambda_i[(w * x + b) - 1]},
    the dual function
        L_D(lambda_mat) = sum_i{lambda_i} - 1/2 sum_i{sum_j{lambda_i lambda_j y_i y_j K(x_i, x_j)}}
    is maximised.
    Attributes:
        kernel --- type of the kernel ['linear'/'rbf'/'poly'/'sigmoid']
        kernel_fn --- kernel function
        gamma --- parameter of the kernel function
        lambdas --- lagrangian multipliers
        sv_X --- support vectors related to X
        sv_y --- support vectors related to y
        w --- matrix of hyperplane parameters
        b --- hyperplane bias
        C --- non-negative float regulating the trade-off between the amount of misclassified samples and
        the size of the margin (its 'softness' decreases as C increases); if it is set to 'None',
        hard margin is employed (no tolerance towards misclassified samples)
        is_fit --- boolean variable indicating whether the SVM is fit or not"""

    def __init__(self,
                 kernel: str = 'linear',
                 gamma: typing.Optional[float] = 1.,
                 deg: int = 3,
                 r: float = 0.,
                 C: float = 1.):
        """Initializes the SVM object by setting the kernel function, its parameters and the soft margin;
        moreover, it sets to None the matrices of lagrangian multipliers and support vectors.
            :param kernel: string representing the kernel type ('linear'/'rbf'/'poly'/'sigmoid'); by default it is
            set to 'linear'
            :param gamma: optional floating point representing the gamma parameters of the kernel;
            by default it is 'None', since it will be computed automatically during fit
            :param deg: optional integer representing the degree of the 'poly' kernel function
            :param r: optional floating point representing the r parameter of 'poly' and 'sigmoid' kernel functions
            :param C: non-negative float regulating the trade-off between the amount of misclassified samples and
            the size of the margin"""
        # Lagrangian multipliers, hyper-parameters and support vectors are initially set to None
        self.lambdas = None
        self.sv_X = None
        self.sv_y = None
        self.w = None
        self.b = None

        # If gamma is None, it will be computed during fit process
        self.gamma = gamma

        # Assign the right kernel
        self.kernel = kernel
        if kernel == 'linear':
            self.kernel_fn = Linear().kernel

        elif kernel == 'rbf':
            self.kernel_fn = RBF(gamma).kernel
        elif kernel in ['Laplacian', 'laplacian']:
            self.kernel_fn = Laplacian(gamma).kernel
        elif kernel == 'chi2':
            self.kernel_fn = chi2(gamma).kernel
        else:
            raise NotImplementedError(f"{self.kernel} not implemented")
        #if kernel == 'linear':
        #    self.kernel_fn = lambda x_i, x_j: np.dot(x_i, x_j)
        #elif kernel == 'rbf':
        #    self.kernel_fn = lambda x_i, x_j: np.exp(-self.gamma * np.dot(x_i - x_j, x_i - x_j))
        #elif kernel == 'poly':
        #    self.kernel_fn = lambda x_i, x_j: (self.gamma * np.dot(x_i, x_j) + r) ** deg
        #elif kernel == 'sigmoid':
        #    self.kernel_fn = lambda x_i, x_j: np.tanh(np.dot(x_i, x_j) + r)

        # Soft margin
        self.C = C

        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray):

        n_samples, n_features = X.shape

        # max{L_D(Lambda)} can be rewritten as
        #   min{1/2 Lambda^T H Lambda - 1^T Lambda}
        #       s.t. -lambda_i <= 0
        #       s.t. lambda_i <= c
        #       s.t. y^t Lambda = 0
        # where H[i, j] = y_i y_j K(x_i, x_j)
        # This form is conform to the signature of the quadratic solver provided by CVXOPT library:
        #   min{1/2 x^T P x + q^T x}
        #       s.t. G x <= h
        #       s.t. A x = b
        
        #K = np.zeros(shape=(n_samples, n_samples))
        #for i, j in itertools.product(range(n_samples), range(n_samples)):
        #    K[i, j] = self.kernel_fn(X[i], X[j])
        K = self.kernel_fn(X,X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        # Compute G and h matrix according to the type of margin used
        if self.C:
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples),
                                         np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples),
                                         np.ones(n_samples) * self.C)))
        else:
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        A = cvxopt.matrix(y.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        # Set CVXOPT options
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 200

        # Compute the solution using the quadratic solver
        try:
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            print('Impossible to fit, try to change kernel parameters; CVXOPT raised Value Error: {0:s}'.format(e))
            return
        # Extract Lagrange multipliers
        lambdas = np.ravel(sol['x'])
        # Find indices of the support vectors, which have non-zero Lagrange multipliers, and save the support vectors
        # as instance attributes
        is_sv = lambdas > 1e-5
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]
        self.lambdas = lambdas[is_sv]
        # Compute b as 1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lambdas * self.sv_y * K[sv_index[i], is_sv])
        self.b /= len(self.lambdas)
        # Compute w only if the kernel is linear
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for i in range(len(self.lambdas)):
                self.w += self.lambdas[i] * self.sv_X[i] * self.sv_y[i]
        else:
            self.w = None
        self.is_fit = True

    def project(self,
                X: np.ndarray,
                i: typing.Optional[int] = None,
                j: typing.Optional[int] = None):

        # If the kernel is linear and 'w' is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            fx = self.kernel_fn(X, self.sv_X)
            fx = fx * (self.sv_y * self.lambdas)
            d = np.sum(fx, axis=1)
            return d + self.b

    def predict(self, X: np.ndarray) -> int:
        # To predict the point label, only the sign of f(x) is considered
        return np.sign(self.project(X))