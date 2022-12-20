"""
Constrained Mutual Subspace Method
"""

# Authors: Junki Ishikawa

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..utils import mean_square_singular_values, subspace_bases
from .base_class import ConstrainedSMBase, MSMInterface


class ConstrainedMSM(MSMInterface, ConstrainedSMBase):
    """
    Constrained Mutual Subspace Method
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
        bases = subspace_bases(X, self.n_subdims)
        # bases, (n_gds_dims, n_subdims)
        bases = self._gds_projection(bases)

        # gramians, (n_classes, n_subdims, n_subdims)
        gramians = np.dot(self.dic.transpose(0, 2, 1), bases)

        return gramians


class CMSMrff(MSMInterface, ConstrainedSMBase):
    """
    Mutual Subspace Method
    """

    def __init__(
        self,
        n_subdims,
        n_gds_dims,
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
        self.n_gds_dims = n_gds_dims
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
        newX = np.array(newX)

        dic = [subspace_bases(_X, self.n_subdims) for _X in newX]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        # all_bases, (n_dims, n_classes * n_subdims)
        all_bases = np.hstack(dic)

        # n_gds_dims
        if 0.0 < self.n_gds_dims <= 1.0:
            n_gds_dims = int(all_bases.shape[1] * self.n_gds_dims)
        else:
            n_gds_dims = self.n_gds_dims

        # gds, (n_dims, n_gds_dims)
        self.gds = subspace_bases(all_bases, n_gds_dims, higher=False)

        dic = self._gds_projection(dic)
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
        newX = []
        for n in range(self.n_approx):
            newX.append(self.z(X, self.w[n], self.b[n], self.m))
        newX = np.stack(newX, axis=0).mean(axis=0)

        # bases, (n_dims, n_subdims)
        bases = subspace_bases(newX, self.n_subdims)
        # bases, (n_gds_dims, n_subdims)
        bases = self._gds_projection(bases)

        # gramians, (n_classes, n_subdims, n_subdims)
        gramians = np.dot(self.dic.transpose(0, 2, 1), bases)

        return gramians
