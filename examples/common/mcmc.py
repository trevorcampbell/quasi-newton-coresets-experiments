import stan
import time
import hashlib
import logging
import numpy as np

def sample(sampler_data, N_samples, model_name, model_code, seed,
        chains=1, verbose = False):

    # suppress the large amount of stan output and newlines
    logging.getLogger('httpstan').setLevel('WARNING')
    logging.getLogger('aiohttp').setLevel('WARNING')
    logging.getLogger('asyncio').setLevel('WARNING')

    # pystan 3 caches models itself, so no need to do that any more
    if verbose: print('STAN: building/loading model ' + model_name)
    sm = stan.build(model_code, data=sampler_data, random_seed=seed)

    if verbose: print('STAN: generating ' + str(N_samples) + ' samples from ' + model_name)
    t0 = time.perf_counter()
    #call sampling with N_samples actual iterations, and some number of burn iterations
    fit = sm.sample(num_samples=N_samples, num_chains=chains)

    t_sample = time.perf_counter() - t0
    t_per_iter = t_sample/(2.*N_samples) #denominator *2 since stan doubles the number of samples for tuning
    if verbose: print('STAN: total time: ' + str(t_sample) + ' seconds')
    if verbose: print('STAN: time per iteration: ' + str(t_per_iter) + ' seconds')
    return fit, t_sample, t_per_iter


def sample_naive(sampler_data, N_samples, model_name,
                 MH_proposal, log_MH_transition_ratio, log_joint,
                 seed, verbose = False):

    if verbose: print('Naive MCMC: generating ' + str(N_samples) + ' samples from ' + model_name)

    np.random.seed(seed)

    t0 = time.perf_counter()
    #Perform MCMC sampling with N_samples actual iterations, and some number of burn iterations
    samples = np.zeros((N_samples, sampler_data['d']))
    th_current = sampler_data['th_init']
    for i in range(2*N_samples):
        if np.mod(i, 100) == 0:
            print(i)
        th_prop = MH_proposal(th_current)

        log_accept = (log_joint(sampler_data['z'], th_prop, sampler_data['w'])
                      - log_joint(sampler_data['z'], th_current, sampler_data['w'])
                      + log_MH_transition_ratio(th_current,th_prop))

        if np.log(np.random.uniform()) < log_accept:
            th_current = th_prop
        if i>=N_samples:
            samples[i-N_samples,:] = th_current


    t_sample = time.perf_counter() - t0
    t_per_iter = t_sample/(2.*N_samples) #denominator *2 since we double the number of samples for tuning
    if verbose: print('Naive MCMC: total time: ' + str(t_sample) + ' seconds')
    if verbose: print('Naive MCMC: time per iteration: ' + str(t_per_iter) + ' seconds')
    return samples, t_sample, t_per_iter