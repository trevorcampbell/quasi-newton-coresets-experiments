import numpy as np
from .util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError

class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]))

    def project(self, pts, grad=False, return_sum=False):
        # if return_sum:
        #     lls_sum = 0
        #     for i in range(pts.shape[0]):
        #         lls_current = self.loglikelihood(pts[i, :], self.samples)
        #         lls_current -= lls_current.mean(axis=1)[:, np.newaxis]
        #         lls_sum += lls_current
        #     return lls_sum.ravel()
        if return_sum:
            # Calculate sum in batches
            num_batches = 1000
            lls_sum = 0
            pts_batches = np.array_split(pts,num_batches)
            for i in range(num_batches):
                lls_current = self.loglikelihood(pts_batches[i], self.samples)
                lls_current -= lls_current.mean(axis=1)[:, np.newaxis]
                lls_sum += lls_current.sum(axis=0)
            return lls_sum.ravel()
        else:
            lls = self.loglikelihood(pts, self.samples)
            lls -= lls.mean(axis=1)[:,np.newaxis]
            if grad:
                if self.grad_loglikelihood is None:
                    raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
                glls = self.grad_loglikelihood(pts, self.samples)
                glls -= glls.mean(axis=2)[:, :, np.newaxis]
                return lls, glls
            else:
                return lls

    def update(self, wts, pts):
        self.samples = self.sampler(self.projection_dimension, wts, pts)
