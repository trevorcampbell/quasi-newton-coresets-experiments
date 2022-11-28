import numpy as np

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  Y[Y == 0] = -1
  # #standardize the covariates; last col is intercept, so no stdization there
  # m = X[:, :-1].mean(axis=0)
  # V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  # X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Z = Y[:, np.newaxis]*X
  data.close()
  return X, Y, Z, None

def load_delays_data(dnm):
  d = np.load(dnm)
  d_filtered = d[(d[:, 7] == 0) & (d[:, 8] == 0), :]

  X = np.hstack((d_filtered[:, 0, np.newaxis], d_filtered[:, 6, np.newaxis], d_filtered[:, 9:17], d_filtered[:, 18:]))
  if np.isnan(X).any():
    raise ValueError('NaNs in data')
  Y = (d_filtered[:, 3]>0).astype(int)
  Y[Y == 0] = -1
  if np.isnan(Y).any():
    raise ValueError('NaNs in data')
  # Transform predictors in X
  # sin transform for day of week
  X[:, 0] = np.sin(2 * np.pi * X[:, 0] / 7)
  # log transform for distance
  X[:, 2] = np.log(X[:, 2])
  # subtract historical averages for temps
  X[:, 3:6] = X[:, 3:6] - X[:, 6:9]
  # drop historial average columns
  X = np.hstack((X[:, :6], X[:, 9:]))
  # Remove visibility column - can be same as intercept when subsampling
  X = np.hstack((X[:, :-2], X[:, -1, np.newaxis]))
  # Center other columns
  m = X[:, 7:].mean(axis=0)
  X[:, 7:] -= m

  # Add intercept term
  X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

  Z = Y[:, np.newaxis] * X
  return X, Y, Z, None

def gen_synthetic(n, d_subspace, d_complement):

  X = np.zeros((n, d_subspace+d_complement+1))
  X[:, :d_subspace] = np.random.randn(n, d_subspace)
  X[:, -1] = 1
  th = 3*np.random.randn(X.shape[1])
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y = (np.random.rand(n) <= ps).astype(int)
  y[y==0] = -1
  Z = y[:, np.newaxis]*X
  return X, y, Z, None

def log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = -np.log1p(np.exp(m[idcs]))
  m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
  return m

def log_prior(th, sig0):
  th = np.atleast_2d(th)
  sigma = np.atleast_2d(np.repeat(sig0, th.shape[1]))
  ll = -np.log(np.pi*(sigma + th**2/sigma))
  return ll.sum(axis=1)

def log_joint(z, th, wts, sig0):
    return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0) + log_prior(th, sig0)

def grad_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  m[np.logical_not(idcs)] = 1.
  return m[:, :, np.newaxis]*z[:, np.newaxis, :]

def grad_z_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  m[np.logical_not(idcs)] = 1.
  return m[:, :, np.newaxis]*th[np.newaxis, :, :]

def grad_th_log_prior(th, sig0):
  th = np.atleast_2d(th)
  sigma = np.atleast_2d(np.repeat(sig0, th.shape[1]))
  return -2*th/(sigma**2 + th**2)

def grad_th_log_joint(z, th, wts, sig0):
  return grad_th_log_prior(th, sig0) + (wts[:, np.newaxis, np.newaxis]*grad_th_log_likelihood(z, th)).sum(axis=0)

def hess_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))**2
  m[np.logical_not(idcs)] = 0.
  return -m[:, :, np.newaxis, np.newaxis]*z[:, np.newaxis, :, np.newaxis]*z[:, np.newaxis, np.newaxis, :]

def hess_th_log_prior(th, sig0):
  th = np.atleast_2d(th)
  sigma = np.atleast_2d(np.repeat(sig0, th.shape[1]))
  h = -2 * (sigma ** 2 - th ** 2) / ((sigma ** 2 + th ** 2) ** 2)
  h = np.diag(h[0,:])
  return np.tile(h, (th.shape[0], 1, 1))

def hess_th_log_joint(z, th, wts, sig0):
  return hess_th_log_prior(th, sig0) + (wts[:, np.newaxis, np.newaxis, np.newaxis]*hess_th_log_likelihood(z, th)).sum(axis=0)

def diag_hess_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))**2
  m[np.logical_not(idcs)] = 0.
  return -m[:, :, np.newaxis]*z[:, np.newaxis, :]**2

def diag_hess_th_log_prior(th, sig0):
  th = np.atleast_2d(th)
  sigma = np.atleast_2d(np.repeat(sig0, th.shape[1]))
  h = -2 * (sigma ** 2 - th ** 2) / (sigma ** 2 + th ** 2) ** 2
  dh = h[0, :]
  return np.tile(dh, (th.shape[0], 1))

def diag_hess_th_log_joint(z, th, wts, sig0):
  return diag_hess_th_log_prior(th, sig0) + (wts[:, np.newaxis, np.newaxis]*diag_hess_th_log_likelihood(z, th)).sum(axis=0)

def MH_proposal(th):
  th = np.atleast_2d(th)
  th_new = th + 0.1*np.random.randn(1,th.shape[1])
  return th_new

def log_MH_transition_ratio(th,th_new):
  # Symmetric proposal so log transition ratio is 0
  return 0

stan_code = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0,upper=1> y[n]; // outputs
  matrix[n,d] x; // inputs
  vector<lower=0>[n] w;  // weights
  real sig0; // prior scale
}
parameters {
  vector[d] theta; // auxiliary parameter
}
model {
  for(i in 1:d){
    theta[i] ~ cauchy(0, sig0);
  }
  for(i in 1:n){
    target +=  bernoulli_logit_lupmf(y[i] | x[i]*theta) * w[i];
  }
}
"""
