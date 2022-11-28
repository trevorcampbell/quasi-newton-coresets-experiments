import numpy as np
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS

class IHT(SparseNNLS):

    def __init__(self, A, b):
        super().__init__(A, b)
        self.A = A
        self.b = np.atleast_2d(b).T

    #overrides SparseNNLS build
    def build(self, size):
        if self.A.size == 0:
            self.log.warning('there are no data, returning.')
            return
        self.a_iht_ii(size)

    def a_iht_ii(self, K, tol=1e-5, max_iter_num=300):
        (M, N) = self.A.shape
        if len(self.b.shape) != 2:
            raise ValueError('b should have shape (M, 1)')

        # Initialize to zero vector
        w_cur = np.zeros([N, 1])
        y_cur = np.zeros([N, 1])

        A_w_cur = np.zeros([M, 1])
        Y_i = []

        # auxiliary variables
        complementary_Yi = np.ones([N, 1])
        i = 1

        while i <= max_iter_num:
            w_prev = w_cur
            if i == 1:
                res = self.b
                der = (self.A.T).dot(res)  # compute gradient
            else:
                res = self.b - A_w_cur - tau * A_diff
                der = (self.A.T).dot(res)  # compute gradient

            A_w_prev = A_w_cur
            complementary_Yi[Y_i] = 0
            ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
            complementary_Yi[Y_i] = 1
            S_i = Y_i + np.squeeze(ind_der[0:K]).tolist()  # identify active subspace
            ider = der[S_i]
            Pder = self.A[:, S_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
            b = y_cur + mu_bar * der  # gradient descent
            w_cur, X_i = self._l2_projection(b, K)

            A_w_cur = self.A[:, X_i].dot(w_cur[X_i])
            res = self.b - A_w_cur
            der = (self.A.T).dot(res)  # compute gradient
            ider = der[X_i]
            Pder = self.A[:, X_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
            w_cur[X_i] = w_cur[X_i] + mu_bar * ider  # debias
            w_cur, _ = self._l2_projection(w_cur, K, already_K_sparse=True, K_sparse_supp=X_i)

            A_w_cur = self.A[:, X_i].dot(w_cur[X_i])
            res = self.b - A_w_cur

            if i == 1:
                A_diff = A_w_cur
            else:
                A_diff = A_w_cur - A_w_prev

            temp = A_diff.T.dot(A_diff)
            if temp > 0:
                tau = res.T.dot(A_diff) / temp
            else:
                tau = res.T.dot(A_diff) / 1e-6

            y_cur = w_cur + tau * (w_cur - w_prev)
            Y_i = np.nonzero(y_cur)[0].tolist()

            # stop criterion
            if (i > 1) and (np.linalg.norm(w_cur - w_prev) < tol * np.linalg.norm(w_cur)):
                break
            i = i + 1

        # finished
        self.w = w_cur
        #supp = np.nonzero(w_cur)[0].tolist()  # support of the output solution
        #return w, supp

    def _l2_projection(self, x, K, already_K_sparse=False, K_sparse_supp=None):
        """
        project x to the K-sparsity constrained and non-negative region;
        the projection is optimal in l2 distance.
        :param x: numpy.ndarray of shape (N, 1)
        :param K: int (sparsity constraint), positive
        :param L: float, positive
        :param already_K_sparse: bool. If the input x has been already K-sparse, put 'True' to for a faster projection
        :param K_sparse_supp: list. If the input x has been already K-sparse, put its support here
        :return: x: numpy.ndarray of shape (N, 1). A new vector that is the projected x
                 selected_support: list of integer indexes (the support of the x).
        """
        N = x.shape[0]
        if already_K_sparse:
            x_projected = x.copy()
            x_projected[x_projected < 0] = 0
            return x_projected, K_sparse_supp
        else:
            index_x = np.flip(np.argsort(np.squeeze(x)))
            index_x = np.squeeze(index_x).tolist()
            x_projected = np.zeros([N, 1])
            selected_support = index_x[0:K]
            x_projected[selected_support] = x[index_x[0:K]]  # projection
            x_projected[x_projected < 0] = 0  # truncate negative entries
            return x_projected, selected_support
