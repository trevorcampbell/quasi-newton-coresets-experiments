import numpy as np

np.random.seed(1)
N = 1000000
D = 100
mu = 10*np.random.randn(D)
sig = 10.
np.save("synth_gauss_large.npy", mu + sig*np.random.randn(N, D))
