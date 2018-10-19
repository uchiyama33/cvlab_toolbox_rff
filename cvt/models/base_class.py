"""
Subspace Method Interface
"""

# Authors: Junki Ishikawa

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np

from cvt.utils import subspace_bases
from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values


class SMBase(BaseEstimator, ClassifierMixin):
    """
    Base class of Subspace Method
    """

    def __init__(self, n_subdims, normalize=False):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1
        """
        self.n_subdims = n_subdims
        self.normalize = normalize
        self.le = LabelEncoder()
        self.dic = None
        self.labels = None
        self.n_classes = None

    def get_params(self):
        return {'n_subdims': self.n_subdims, 'normalize': self.normalize}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _prepare(self, X):
        """
        preprocessing data matricies X.
        normalize and transpose

        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
        """
        # normalize each vectors
        if self.normalize:
            X = [_normalize(_X) for _X in X]

        # transpose to make feature vectors as column vectors
        # this makes it easy to implement refering to formula
        X = [_X.T for _X in X]

        return X

    def fit(self, X, y):
        """
        Fit the model using the given data and parameters

        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
            Training vectors. n_classes is count of classes.
            n_samples is number of vectors of samples, this is variable across each classes.
            n_dims is number of dimentions of vectors.

        y: integer array, (n_classes)
            Class labels of training vectors. 
        """

        # preprocessing data matricies
        # ! X[i] will transposed for conventional
        X = self._prepare(X)

        # converted labels
        y = self.le.fit_transform(y)
        self.labels = y

        # number of classes
        self.n_classes = self.le.classes_.size

        self._fit(X, y)

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """
        dic = [subspace_bases(_X, self.n_subdims) for _X in X]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        self.dic = dic

    def predict(self, X):
        """
        Predict each classes

        Parameters:
        -----------
        X: 2d-array, (n_samples, n_dims)
            Matrix of input vectors.

        Returns:
        --------
        pred: array-like, shape: (n_samples)
            Prediction array

        """

        # preprocessing data matricies
        X = self._prepare([X])[0]

        pred = self._predict(X)
        return self.le.inverse_transform(pred)

    def _predict(self, X):
        """
        Parameters
        ----------
        X: arrays, (n_dims, n_samples)
        """
        raise NotImplementedError('_predict is not implemented')


class KernelSMBase(SMBase):
    """
    Base class of Kernel Subspace Method
    """

    def __init__(self, n_subdims, normalize=False, sigma=None):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1

        sigma : int or str, optional (default=None)
            a parameter of rbf kernel. if sigma is None, sqrt(n_dims / 2) will be used.
        """
        super(KernelSMBase, self).__init__(n_subdims, normalize)
        self.sigma = sigma

    def get_params(self, deep=True):
        return {
            'n_subdims': self.n_subdims,
            'sigma': self.sigma,
        }

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """
        coeff = []
        for _X in X:
            K = rbf_kernel(_X, _X, self.sigma)
            _coeff, _ = dual_vectors(K, self.n_subdims)
            coeff.append(_coeff)

        self.dic = list(zip(X, coeff))


class ConstrainedSMBase(SMBase):
    """
    Base class of Constrained Subspace Method
    """

    def __init__(self, n_subdims, n_gds_dims, normalize=False):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        n_gds_dims : int
            The dimension of Generalized Difference Subspace.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1.
        """
        super(ConstrainedSMBase, self).__init__(n_subdims, normalize)
        self.n_gds_dims = n_gds_dims
        self.gds = None

    def get_params(self, deep=True):
        return {
            'n_subdims': self.n_subdims,
            'n_gds_dims': self.n_gds_dims,
        }

    def _gds_projection(self, bases):
        """
        GDS projection.
        Projected bases will be normalized and orthogonalized.

        Parameters
        ----------
        bases: arrays, (n_dims, n_subdims)

        Returns
        -------
        bases: arrays, (n_gds_dims, n_subdims)
        """

        # bases_proj, (n_gds_dims, n_subdims)
        bases_proj = np.matmul(self.gds.T, bases)
        qr = np.vectorize(np.linalg.qr, signature='(n,m)->(n,m),(m,m)')
        bases, _ = qr(bases_proj)
        return bases

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """

        dic = [subspace_bases(_X, self.n_subdims) for _X in X]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        # all_bases, (n_dims, n_classes * n_subdims)
        all_bases = np.hstack(dic)
        # gds, (n_dims, n_gds_dims)
        self.gds = subspace_bases(all_bases, self.n_gds_dims, higher=False)

        dic = self._gds_projection(dic)
        self.dic = dic


class MSMInterface(object):
    """
    Prediction interface of Mutual Subspace Method
    """

    def predict(self, X):
        """
        Predict each classes

        Parameters:
        -----------
        X: list of 2d-arrays, (n_vector_sets, n_samples, n_dims)
            List of input vector sets.

        Returns:
        --------
        pred: array, (n_vector_sets)
            Prediction array

        """

        # preprocessing data matricies
        X = self._prepare(X)

        pred = []
        for _X in X:
            # gramians, (n_classes, n_subdims, n_subdims)
            gramians = self._get_gramians(_X)

            # i_th singular value of grammian of subspace bases is
            # square root of cosine of i_th cannonical angles
            # average of square of them is caonnonical angle between subspaces
            c = [mean_square_singular_values(g) for g in gramians]
            pred.append(self.labels[np.argmax(c)])
        pred = np.array(pred)
        return self.le.inverse_transform(pred)

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)

        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """
        raise NotImplementedError('_get_gramians is not implemented')
