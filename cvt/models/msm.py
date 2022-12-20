"""
Mutual Subspace Method
"""

# Authors: Junki Ishikawa

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..utils import subspace_bases
from .base_class import MSMInterface, SMBase


class MutualSubspaceMethod(MSMInterface, SMBase):
    """
    Mutual Subspace Method
    """

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

        # bases, (n_dims, n_subdims)
        bases = subspace_bases(X, self.test_n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, : self.n_subdims]
        gramians = np.dot(dic.transpose(0, 2, 1), bases)

        return gramians


class MSMrff(MSMInterface, SMBase):
    """
    Mutual Subspace Method
    """

    def __init__(
        self,
        n_subdims,
        m_rand_samples,
        normalize=False,
        sigma=None,
        faster_mode=False,
        test_n_subdims=None,
        n_approx=1,
    ):
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
        self.faster_mode = faster_mode
        self.le = LabelEncoder()
        self.dic = None
        self.labels = None
        self.n_classes = None
        self._test_n_subdims = test_n_subdims
        self.params = ()

        self.m = m_rand_samples
        self.n_approx = n_approx
        self.sigma = sigma

    def z(self, X, w, b, m):
        return np.sqrt(2 / m) * np.cos((w @ X).T + b).T

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """

        n_dims = X[0].shape[0]
        if self.sigma is None:
            self.sigma = np.sqrt(n_dims / 2)

        w = []
        b = []
        for n in range(self.n_approx):
            w.append(np.random.randn(self.m, n_dims) / self.sigma)
            b.append(np.random.rand(self.m) * 2 * np.pi)
        self.w = np.stack(w)
        self.b = np.stack(b)

        newX = []
        for _X in X:
            _newX = []
            for n in range(self.n_approx):
                _newX.append(self.z(_X, self.w[n], self.b[n], self.m))
            _newX = np.stack(_newX, axis=0).mean(axis=0)
            newX.append(_newX)

        dic = [subspace_bases(_X, self.n_subdims) for _X in newX]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        self.dic = dic

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

        # bases, (n_dims, n_subdims)
        newX = []
        for n in range(self.n_approx):
            newX.append(self.z(X, self.w[n], self.b[n], self.m))
        newX = np.stack(newX, axis=0).mean(axis=0)

        bases = subspace_bases(newX, self.test_n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, : self.n_subdims]
        gramians = np.dot(dic.transpose(0, 2, 1), bases)

        return gramians
