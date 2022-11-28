import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset

class HilbertCoreset(Coreset):
  def __init__(self, data, projector, n_subsample=None, snnls=GIGA, **kw):
    self.data = data
    self.projector = projector
    self.snnls_class = snnls
    self.snnls = None
    super().__init__(**kw)

  def reset(self):
    if self.snnls is not None:
        self.snnls.reset()
    super().reset()

  def _build_projector(self, size):
    cts = []
    ct_idcs = []
    for i in range(size):
        f = np.random.randint(self.data.shape[0])
        if f in ct_idcs:
            cts[ct_idcs.index(f)] += 1
        else:
            ct_idcs.append(f)
            cts.append(1)
    wts = self.data.shape[0] * np.array(cts) / np.array(cts).sum()
    idcs = np.array(ct_idcs)
    self.projector.update(wts, self.data[idcs,:])

  def _build(self, size):

    # build a projector using a uniformly random coreset
    self._build_projector(size)

    # project the data log likelihoods
    vecs = self.projector.project(self.data)

    # construct the snnls object 
    self.snnls = self.snnls_class(vecs.T, vecs.sum(axis=0))

    # build the coreset
    self.snnls.build(size)

    # extract the results from the snnls object
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = np.where(w>0)[0]
    self.pts = self.data[self.idcs]

  def _optimize(self):
    self.snnls.optimize()
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def error(self):
    return self.snnls.error()
