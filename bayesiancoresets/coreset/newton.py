import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt, an_opt
from .coreset import Coreset
from scipy.linalg import solve_triangular
import time

class QuasiNewtonCoreset(Coreset):
    def __init__(self, data, projector, n_subsample_opt=None, opt_itrs=20, augment_sample=False, **kw):
        self.data = data
        self.cts = []
        self.ct_idcs = []
        self.projector = projector
        self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
        self.opt_itrs = opt_itrs
        self.augment_sample = augment_sample
        super().__init__(**kw)

    def reset(self):
        self.cts = []
        self.ct_idcs = []
        super().reset()

    def _build(self, size):
        # # reset data points
        # self.reset()
        # uniformly subset data points
        self._select(size)
        if self.augment_sample is True:
            # Augment the samples with any missed points
            self._augment()
        # optimize the weights
        self._optimize()

    def _get_projection(self, n_subsample, w, p, return_sum=False):
        # update the projector
        self.projector.update(w, p)

        # construct a tangent space
        if n_subsample is None:
            sub_idcs = None
            vecs = self.projector.project(self.data, return_sum=return_sum)
            sum_scaling = 1.
        else:
            sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
            vecs = self.projector.project(self.data[sub_idcs], return_sum=return_sum)
            sum_scaling = self.data.shape[0] / n_subsample

        if self.pts.size > 0:
            corevecs = self.projector.project(self.pts)
        else:
            corevecs = np.zeros((0, vecs.shape[1]))

        return vecs, sum_scaling, sub_idcs, corevecs

    def _select(self, size):
        for i in range(size):
            f = np.random.randint(self.data.shape[0])
            if f in self.ct_idcs:
                self.cts[self.ct_idcs.index(f)] += 1
            else:
                self.ct_idcs.append(f)
                self.cts.append(1)
        self.wts = self.data.shape[0] * np.array(self.cts) / np.array(self.cts).sum()
        self.idcs = np.array(self.ct_idcs)
        self.pts = self.data[self.idcs]

    def _augment(self):
        # Take a sample of coreset log-likelihoods from the current coreset posterior:
        _, _, _, corevecs = self._get_projection(self.n_subsample_opt, self.wts, self.pts, return_sum=True)

        # Define basis matrix and orthogonalise columns
        A = corevecs.T
        U, _ = np.linalg.qr(A)
        rel_comp_norms = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            # For each datapoint in the dataset, sample the log-likelihood
            test_vec = self.projector.project(self.data[i,:], return_sum=False)[0,:]
            # Project onto the space spanned by the (orthogonalised) coreset log-likelihoods
            test_vec_proj = U.dot(test_vec.dot(U))
            rel_comp_norms[i] = np.sqrt(((test_vec - test_vec_proj)**2).sum())/np.sqrt(((test_vec)**2).sum())
            print("Data point: {}, relative norm: {}".format(i,rel_comp_norms[i]))
            if rel_comp_norms[i] > 0.9:
                print("Adding data point: {} to coreset".format(i))
                self.ct_idcs.append(i)
                self.cts.append(1)

        print("New coreset size: {}".format(np.array(self.cts).sum()))
        # Recalculate wts, idcs, pts
        self.wts = self.data.shape[0] * np.array(self.cts) / np.array(self.cts).sum()
        self.idcs = np.array(self.ct_idcs)
        self.pts = self.data[self.idcs]

    def _optimize(self):

        def grad_norm_variance(w):
            # Tune the number of samples needed to reduce the noise
            # of the stochastic Newton step below a certain threshold
            t0 = time.perf_counter()
            vecs_sum, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,return_sum=True)
            # vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,
            #                                                                          return_sum=False)
            resid = sum_scaling*vecs_sum - w.dot(corevecs)
            # resid = sum_scaling * vecs.sum(axis=0) - w.dot(corevecs)

            grd_samples = corevecs*resid

            grd_norms = np.sqrt(np.sum(grd_samples ** 2, axis=0))
            grd_norm_variance = np.var(grd_norms)/corevecs.shape[1]

            t_sample = time.perf_counter() - t0
            print("Time taken: {} seconds".format(t_sample))
            return grd_norm_variance
        
        def search_direction(w, tau=0.01):
            t0 = time.perf_counter()
            vecs_sum, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,return_sum=True)
            # vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,
            #                                                                          return_sum=False)
            resid = sum_scaling*vecs_sum - w.dot(corevecs)
            # resid = sum_scaling * vecs.sum(axis=0) - w.dot(corevecs)

            corevecs_cov = corevecs.dot(corevecs.T) / corevecs.shape[1]
            # add regularization term to hessian
            np.fill_diagonal(corevecs_cov, corevecs_cov.diagonal() + tau)
            grd = (corevecs.dot(resid) / corevecs.shape[1])
            print("gradient norm: {}".format(np.sqrt(((grd)**2).sum())))
            # output gradient of weights at idcs
            search_direction = np.linalg.solve(corevecs_cov, grd)
            t_sample = time.perf_counter() - t0
            print("Time taken: {} seconds".format(t_sample))
            return search_direction

        def grd(w):
            vecs_sum, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,return_sum=True)
            # vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,
            #                                                                    return_sum=False)
            resid = sum_scaling*vecs_sum - w.dot(corevecs)
            # resid = sum_scaling * vecs.sum(axis=0) - w.dot(corevecs)
            grd = (corevecs.dot(resid) / corevecs.shape[1])
            return -grd

        x0 = self.wts
        self.wts = an_opt(x0, grd, search_direction, grad_norm_variance, opt_itrs=self.opt_itrs)
        # use uniform weights if sum of weights is negative
        if self.wts.sum() <= 0:
            self.wts = self.data.shape[0] * np.ones(self.pts.shape[0]) / self.pts.shape[0]
