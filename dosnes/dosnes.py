import numpy as np

from scipy.spatial.distance import squareform, pdist
from scipy.special import xlogy

from . import sinkhorn_knopp as skp

class DOSNES():
    """
    Implementation of the Doubly Stochastic Neighbor Embedding on Spheres algorithm published by Yao Lu in Sep. 2016
    https://arxiv.org/abs/1609.01977

    This algorithm is used to embbed datas on a 3-Dimensional Sphere. It's a translation in "sklearn" of the repo
    https://github.com/yaolubrain/DOSNES but in Python

    Thanks to btaba for the implementation of sinkhorn_knopp algorithm: https://github.com/btaba/sinkhorn_knopp
    """

    def __init__(self,
                 metric = "sqeuclidean",
                 max_iter_skp = 1e3,
                 epsilon_skp = 1e-3,
                 momentum=0.5,
                 final_momentum = 0.99,
                 mom_switch_iter = 250,
                 max_iter = 1000,
                 learning_rate = 500,
                 min_gain = 0.01,
                 verbose = 0,
                 random_state = None,
                 verbose_freq = 10
                ):

        """Apply Sinkhorn–Knopp Algorithm to have a Doubly Stochastic Matrix.

        Parameters
        ----------
        metric : The distance metric to use. Must be :
                     ‘braycurtis’,
                     ‘canberra’,
                     ‘chebyshev’,
                     ‘cityblock’,
                     ‘correlation’,
                     ‘cosine’,
                     ‘dice’,
                     ‘euclidean’,
                     ‘hamming’,
                     ‘jaccard’,
                     ‘kulsinski’,
                     ‘mahalanobis’,
                     ‘matching’,
                     ‘minkowski’,
                     ‘rogerstanimoto’,
                     ‘russellrao’,
                     ‘seuclidean’,
                     ‘sokalmichener’,
                     ‘sokalsneath’,
                     ‘sqeuclidean’,
                     ‘yule’,
                     'precomputed'.
                If you enter pre-computed, a square matrix of (n_samples, n_samples) must be given as X.

        max_iter_skp : Number of maximum iteration before stopping Sinkhorn–Knopp Algorithm

        epsilon_skp :  Stopping Criterion for the Sinkhorn–Knopp Algorithm

        momentum : weight used during iteration process to update position

        final_momentum : weight used during iteration process to update position after mom_switch_iter

        mom_switch_iter : trigger to switch from momentum to final_momentum

        max_iter : number of iterations to optimise embedding

        learning_rate : Learning rate used to update position of points in the embedded space

        min_gain : value used to clip negative gains (must be > 0)

        verbose : Display cost during training

        verbose_freq :  Number of iterations between every print of the cost function

        random_state : Random State used to generate the initial y matrix
        """

        self.metric = metric
        self.max_iter_skp = max_iter_skp
        self.epsilon_skp = epsilon_skp
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.mom_switch_iter = mom_switch_iter
        self.min_gain = min_gain
        self.verbose = verbose
        self.random_state = random_state
        self.verbose_freq = verbose_freq
        np.random.seed(random_state)

    def _fit(self, X):
        """Apply Sinkhorn–Knopp Algorithm to have a Doubly Stochastic Matrix.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
            Pairwise Distance Matrix

        Returns
        -------
        P : array, shape (n_samples, 3)
            Non normalized sperical embedded in a 3-dimensional space
        """

        no_dims = 3
        n = X.shape[0]
        momentum = self.momentum
        MACHINE_EPSILON = np.finfo(np.double).eps
        min_cost = 1e10

        if self.metric == "precomputed":
            pass
        else:
            if self.verbose == 1:
                print("Computing distances")
            X = squareform(pdist(X, metric=self.metric))

        if self.verbose == 1:
            print("Set doubly Stochastic")
        P = self._set_doubly_stochastic(X)

        del X

        if self.verbose == 1:
            print("Start Embedding")
        P[np.diag_indices_from(P)] = 0.

        P = (P + P.T) / 2
        P = np.maximum( P / P.sum(), MACHINE_EPSILON)

        const = np.sum(xlogy(P, P))

        ydata = 1e-4 * np.random.random(size=(n, no_dims))

        y_incs = np.zeros(shape = ydata.shape)
        gains = np.ones(shape = ydata.shape)

        for iter in range(self.max_iter):
            sum_ydata = np.sum(ydata ** 2, axis=1)

            num = 1. / (1 + sum_ydata + sum_ydata[np.newaxis].T + -2 * np.dot(ydata, ydata.T))

            num[np.diag_indices_from(num)] = 0.

            Q = np.maximum(num / num.sum(), MACHINE_EPSILON)

            L = (P - Q) * num

            t = np.diag( L.sum(axis=0) ) - L
            y_grads = 4 * np.dot(t, ydata)

            inc = (np.sign(y_grads) != np.sign(y_incs))
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8

            gains = np.clip(gains, a_min=self.min_gain, a_max=np.inf)

            y_incs = momentum * y_incs - self.learning_rate * gains * y_grads
            ydata += y_incs

            ydata -= ydata.mean(axis=0)

            rad = np.sqrt(np.sum(ydata ** 2, axis=1))
            r_mean = np.mean(rad)
            ydata *= (r_mean / rad).reshape(-1, 1)

            # cost = const - np.sum(xlogy(P, Q))
            # if cost < min_cost:
            #     min_cost = cost
            #     self.embedding = ydata / np.sqrt(np.sum(ydata ** 2, axis=1)).reshape(-1, 1)

            if iter == self.mom_switch_iter:
                momentum = self.final_momentum

            if iter % self.verbose_freq == 0 and self.verbose == 1:
                cost = const - np.sum(xlogy(P, Q))
                print("Iteration {} : error is {}".format(iter, cost))

        self.embedding = ydata / np.sqrt(np.sum(ydata ** 2, axis=1)).reshape(-1, 1)


    def _set_doubly_stochastic(self, D):
        """Apply Sinkhorn–Knopp Algorithm to have a Doubly Stochastic Matrix.

        Parameters
        ----------
        D : array, shape (n_samples, n_samples)
            Pairwise Distance Matrix

        Returns
        -------
        P : array, shape (n_samples, n_samples)
            Doubly Stochastic Pairwise Distance Matrix
        """
        P = np.exp(-D**2/2)
        sk = skp.SinkhornKnopp(max_iter=self.max_iter_skp, epsilon=self.epsilon_skp)
        P = sk.fit(P)
        return P


    def fit_transform(self, X):
        """Fit X into an sperical embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, 3)
            Embedding of the training data in 3-dimensional space.
        """
        self._fit(X)
        return self.embedding


    def fit(self, X):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        """
        self.fit_transform(X)
        return self