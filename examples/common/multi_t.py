import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
import scipy.stats as st

def fit_t(X, max_itr = 10000):
    # initialize
    nu_inits = np.logspace(0., 3., 10)
    
    nll_best = np.inf
    mu_best = np.zeros(X.shape[1])
    Sig_best = np.eye(X.shape[1])
    nu_best = 10.

    # iterate
    for i in range(nu_inits.shape[0]):
        z = np.ones(X.shape[0])
        D = X.shape[1]
        N = X.shape[0]
        mu = X.mean(axis=0)
        Sig = ((X - mu).T).dot(X-mu) / N    
        nu = nu_inits[i]
        prev_nll = np.inf
        cur_nll = np.inf
        itr = 0
        while True:
            itr += 1
            # update mu, Sig
            mu = (z[:,np.newaxis]*X).sum(axis=0) / z.sum()
            Sig = (z*((X - mu).T)).dot(X-mu) / N

            # update nu
            delta = ((X-mu)*((np.linalg.solve(Sig, (X - mu).T)).T)).sum(axis=1)
            z = (nu+D)/(nu+delta)
            ell = np.log(z) + digamma((nu+D)/2.) - np.log((nu+D)/2.)
            res = minimize(fun = lambda x : N*gammaln(x/2.) - N*x/2.*np.log(x/2.) - x/2.* (ell-z).sum(), x0 = 100., tol=1e-12, bounds=[(0.01, None)])
            nu = res.x

            # get the log likelihood of the data under the current model
            cur_model = st.multivariate_t(mu, Sig, df=nu)
            prev_nll = cur_nll
            cur_nll = -cur_model.logpdf(X).sum()
            if (prev_nll - cur_nll)/(cur_nll) < 1e-3 or itr > max_itr:
                break
            
        if cur_nll < nll_best:
            mu_best[:] = mu.copy()
            Sig_best[:,:] = Sig.copy()
            nu_best = nu
            nll_best = cur_nll
     
    return mu_best, Sig_best, nu_best


# a quick test case to verify
if __name__ == '__main__':
    t = st.multivariate_t(np.zeros(2), np.eye(2), df=3)
    X = t.rvs(1000)
    mu, Sig, nu = fit_t(X)
    print(mu)
    print(Sig)
    print(nu)
                      
            
    

