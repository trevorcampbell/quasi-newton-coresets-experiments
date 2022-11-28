from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.stats import cauchy
import time
import sys, os
import argparse
import cProfile, pstats, io
from pstats import SortKey
# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import mcmc
import laplace
import results
import plotting
import stein
from model_gaussian import KL
from bayesiancoresets.snnls import IHT



def plot(arguments):
    # load the dataset of results that matches these input arguments
    df = results.load_matching(arguments, match = ['model', 'dataset', 'samples_inference', 'opt_itrs', 'step_sched'])
    # call the generic plot function
    plotting.plot(arguments, df)


def run(arguments):
    # suffix for printouts
    log_suffix = '(coreset size: ' + str(arguments.coreset_size) + ', numdata: ' + str(arguments.data_num) + ', nbases: ' + str(arguments.n_bases_per_scale) + ', alg: ' + arguments.alg + ', trial: ' + str(arguments.trial)+')'

    #######################################
    #######################################
    ############# Setup ###################
    #######################################
    #######################################

    # check if result already exists for this run, and if so, quit
    if results.check_exists(arguments):
        print('Results already exist for arguments ' + str(arguments))
        print('Quitting.')
        quit()

    np.random.seed(10)
    bc.util.set_verbosity(arguments.verbosity)

    #######################################
    #######################################
    ############# Load Dataset ############
    #######################################
    #######################################

    #load data and compute true posterior
    #each row of x is [lat, lon, price]
    print('Loading data')

    data = np.load('../data/prices2018.npy')
    print('dataset size : ', data.shape)

    print('Subsampling down to '+str(arguments.data_num) + ' points')
    idcs = np.arange(data.shape[0])
    np.random.shuffle(idcs)
    data = data[idcs[:arguments.data_num], :]



    #log transform the prices
    data[:, 2] = np.log10(data[:, 2])

    #get empirical mean/std
    datastd = data[:,2].std()
    datamn = data[:,2].mean()

    #bases of increasing size; the last one is effectively a constant
    basis_unique_scales = np.array([.2, .4, .8, 1.2, 1.6, 2., 100])
    basis_unique_counts = np.hstack((arguments.n_bases_per_scale*np.ones(6, dtype=np.int64), 1))

    #the dimension of the scaling vector for the above bases
    d = basis_unique_counts.sum()
    print('Basis dimension: ' + str(d))

    #generate basis functions by uniformly randomly picking locations in the dataset
    print('Trial ' + str(arguments.trial))
    print('Creating bases')
    basis_scales = np.array([])
    basis_locs = np.zeros((0,2))
    for i in range(basis_unique_scales.shape[0]):
      basis_scales = np.hstack((basis_scales, basis_unique_scales[i]*np.ones(basis_unique_counts[i])))
      idcs = np.random.choice(np.arange(data.shape[0]), replace=False, size=basis_unique_counts[i])
      basis_locs = np.vstack((basis_locs, data[idcs, :2]))

    print('Converting bases and observations into X/Y matrices')
    #convert basis functions + observed data locations into a big X matrix
    X = np.zeros((data.shape[0], basis_scales.shape[0]))
    for i in range(basis_scales.shape[0]):
      X[:, i] = np.exp( -((data[:, :2] - basis_locs[i, :])**2).sum(axis=1) / (2*basis_scales[i]**2) )
    Y = data[:, 2]
    Z = np.hstack((X, Y[:,np.newaxis]))
    # test_bases = np.sort(np.array([np.sort(Z[:, i])[::-1][int(arguments.data_num/100)] for i in range(Z.shape[1])]))
    test_bases = np.sort(np.array([np.sort(Z[:, i])[::-1][200] for i in range(Z.shape[1])]))
    print(test_bases[0])

    np.random.seed(arguments.trial)

    #######################################
    #######################################
    ############ Define Model #############
    #######################################
    #######################################
    # import the model specification
    import model_linreg as model

    #model params
    mu0 = datamn
    sig0 = np.sqrt(datastd**2+datamn**2)
    sig = datastd

    ####################################################################
    ####################################################################
    ############ Construct weighted posterior sampler ##################
    ####################################################################
    ####################################################################

    print('Creating weighted sampler ' + log_suffix)
    def sample_w(n, wts, pts, get_timing=False):
        t0 = time.perf_counter()
        if pts.shape[0] > 0:
            samples = model.weighted_post_sampler(n, pts, wts, sig, mu0, sig0)
        else:
            samples = model.weighted_post_sampler(n, np.zeros((1, Z.shape[1])), np.zeros(1), sig, mu0, sig0)
        t_tot = time.perf_counter()-t0
        if get_timing:
            return samples, t_tot, t_tot/n
        else:
            return samples

    print('Creating naive MCMC sampler ' + log_suffix)

    def sample_w_naive(n, wts, pts, get_timing=False):
        if pts.shape[0] > 0:
            sampler_data = {'z': pts, 'w': wts, 'd': X.shape[1], 'n': pts.shape[0],
                            'th_init': mu0 * np.ones(X.shape[1])}
        else:
            sampler_data = {'z': np.zeros((1, X.shape[1])), 'w': np.zeros(1), 'd': X.shape[1],
                            'th_init': mu0 * np.ones(X.shape[1]), 'n': 1}
        # Naive MCMC sampling using Metropolis-Hastings
        samples, t_mcmc, t_mcmc_per_itr = mcmc.sample_naive(sampler_data, n, 'linear_regression', model.MH_proposal,
                                                            model.log_MH_transition_ratio,
                                                            lambda z, th, wts: model.log_joint(z, th, wts, sig, mu0,
                                                                                               sig0),
                                                            arguments.trial)
        if get_timing:
            return samples, t_mcmc, t_mcmc_per_itr
        else:
            return samples

    if arguments.alg == 'GIGA-LAP' or arguments.alg == 'IHT-LAP':
        t0 = time.perf_counter()
        pi_hat_lap = laplace.LaplaceApprox(lambda th : model.log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0)[0],
                        lambda th : model.grad_log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0)[0,:],
                                        np.zeros(X.shape[1]),
                        hess_log_joint = lambda th : model.hess_log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0))
        pi_hat_lap.build(1)
        t_lap = time.perf_counter() - t0

        mu_lap = pi_hat_lap.th
        LSig_lap = pi_hat_lap.LSig
    else:
        mu_lap = np.zeros(Z.shape[1])
        LSig_lap = np.identity(Z.shape[1])

    def sample_w_lap(n, wts, pts, get_timing=False):
        t0 = time.perf_counter()
        samples = mu_lap + np.random.randn(n, X.shape[1]).dot(LSig_lap.T)
        t_total = time.perf_counter() - t0
        t_per = t_total / n
        if get_timing:
            return samples, t_total, t_per
        else:
            return samples

    #######################################
    #######################################
    ###### Get samples on the full data ###
    #######################################
    #######################################

    print('Checking for cached full samples ' + log_suffix)
    cache_filename = 'full_cache/full_samples_' + str(arguments.data_num) + '_' + str(arguments.n_bases_per_scale) + '.npz'
    if os.path.exists(cache_filename):
        print('Cache exists, loading')
        tmp__ = np.load(cache_filename)
        full_samples = tmp__['samples']
        t_full_per_itr = float(tmp__['t'])
    else:
        print('Cache doesn\'t exist, running sampler')
        full_samples, t_full, t_full_per_itr = sample_w(arguments.samples_inference, np.ones(Z.shape[0]), Z, get_timing=True)
        if not os.path.exists('full_cache'):
            os.mkdir('full_cache')
        np.savez(cache_filename, samples=full_samples, t=t_full_per_itr, allow_pickle=True)

    #######################################
    #######################################
    ## Step 4: Construct Coreset
    #######################################
    #######################################

    print('Creating coreset construction objects ' + log_suffix)
    # create coreset construction objects
    projector = bc.BlackBoxProjector(sample_w, arguments.proj_dim, lambda x, th : model.log_likelihood(x, th, sig), None)
    lap_projector = bc.BlackBoxProjector(sample_w_lap, arguments.proj_dim, lambda x, th : model.log_likelihood(x, th, sig), None)
    unif = bc.UniformSamplingCoreset(Z)
    giga = bc.HilbertCoreset(Z, projector)
    giga_lap = bc.HilbertCoreset(Z, lap_projector)
    sparsevi = bc.SparseVICoreset(Z, projector, n_subsample_select=1000, n_subsample_opt=1000,
                                  opt_itrs=arguments.opt_itrs, step_sched=eval(arguments.step_sched))
    newton = bc.QuasiNewtonCoreset(Z, projector, opt_itrs=arguments.newton_opt_itrs,augment_sample=False)
    lapl = laplace.LaplaceApprox(lambda th : model.log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0)[0],
				    lambda th : model.grad_log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0)[0,:],
                                    np.zeros(X.shape[1]),
				    hess_log_joint = lambda th : model.hess_log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0))
    iht = bc.HilbertCoreset(Z, projector, snnls=IHT)
    iht_lap = bc.HilbertCoreset(Z, lap_projector, snnls=IHT)

    algs = {'SVI' : sparsevi,
            'QNC' : newton,
            'LAP' : lapl,
            'GIGA': giga,
            'UNIF': unif,
            'IHT' : iht,
            'GIGA-LAP': giga_lap,
            'IHT-LAP': iht_lap,
            'FULL': None,
            'NAIVE': None}
    alg = algs[arguments.alg] if arguments.alg not in ['FULL','NAIVE'] else None

    if arguments.alg == 'FULL':
        # cache full mcmc samples per trial (no need to rerun for different coreset sizes)
        t_build = 0.
        if not os.path.exists('full_cache'):
            os.mkdir('full_cache')
        print('Checking for cached comparison full samples ' + log_suffix)
        cache_filename = f'full_cache/full_samples_{arguments.trial}.npz'
        if os.path.exists(cache_filename):
            print('Cache exists, loading')
            tmp__ = np.load(cache_filename)
            approx_samples = tmp__['samples']
            t_approx_per_sample = float(tmp__['t'])
        else:
            print('Cache doesn\'t exist, running sampler')
            approx_samples, t_full, t_approx_per_sample = sample_w(arguments.samples_inference, np.ones(Z.shape[0]), Z, get_timing=True)
            np.savez(cache_filename, samples=approx_samples, t=t_approx_per_sample, allow_pickle=True)
    elif arguments.alg == 'NAIVE':
        # cache naive mcmc samples per trial (no need to rerun for different coreset sizes)
        t_build = 0.
        if not os.path.exists('naive_cache'):
            os.mkdir('naive_cache')
        print('Checking for cached comparison naive samples ' + log_suffix)
        cache_filename = f'naive_cache/naive_samples_{arguments.trial}.npz'
        if os.path.exists(cache_filename):
            print('Cache exists, loading')
            tmp__ = np.load(cache_filename)
            approx_samples = tmp__['samples']
            t_approx_per_sample = float(tmp__['t'])
        else:
            print('Cache doesn\'t exist, running sampler')
            approx_samples, t_full, t_approx_per_sample = sample_w_naive(arguments.samples_inference, np.ones(Z.shape[0]), Z, get_timing=True)
            np.savez(cache_filename, samples=approx_samples, t=t_approx_per_sample, allow_pickle=True)
    elif arguments.alg == 'LAP':
        # cache laplace approximation mean/covar (no need to rerun for different coreset sizes)
        if not os.path.exists('lap_cache'):
            os.mkdir('lap_cache')
        print('Checking for cached laplace samples ' + log_suffix)
        cache_filename = f'lap_cache/lap_samples_{arguments.trial}.npz'
        if os.path.exists(cache_filename):
            print('Cache exists, loading')
            tmp__ = np.load(cache_filename)
            approx_samples = tmp__['samples']
            t_approx_per_sample = float(tmp__['t'])
            t_build = float(tmp__['t_b'])
        else:
            print('Cache doesn\'t exist, running laplace')
            print('Building ' + log_suffix)
            t0 = time.perf_counter()
            alg.build(arguments.coreset_size)
            t_build = time.perf_counter() - t0
            print('Sampling ' + log_suffix)
            approx_samples, t_approx_sampling, t_approx_per_sample = alg.sample(arguments.samples_inference, get_timing=True)
            np.savez(cache_filename, samples=approx_samples, t=t_approx_per_sample, t_b=t_build, allow_pickle=True)
    else:
        # coreset algorithms need to run for each coreset size, no caching
        print('Building ' + log_suffix)
        t0 = time.perf_counter()
        alg.build(arguments.coreset_size)
        t_build = time.perf_counter() - t0
        if arguments.alg == 'GIGA-LAP' or arguments.alg == 'IHT-LAP':
            t_build += t_lap
        print('Sampling ' + log_suffix)
        wts, pts, idcs = alg.get()
        approx_samples, t_approx_sampling, t_approx_per_sample = sample_w(arguments.samples_inference, wts, pts, get_timing=True)

    print('Evaluation ' + log_suffix)
    # get full/approx posterior mean/covariance
    mu_approx = approx_samples.mean(axis=0)
    Sig_approx = np.cov(approx_samples, rowvar=False)
    logsig_approx = np.log(np.diag(Sig_approx))
    LSig_approx = np.linalg.cholesky(Sig_approx)
    LSigInv_approx = solve_triangular(LSig_approx, np.eye(LSig_approx.shape[0]), lower=True, overwrite_b=True, check_finite=False)
    mu_full = full_samples.mean(axis=0)
    Sig_full = np.cov(full_samples, rowvar=False)
    logsig_full = np.log(np.diag(Sig_full))
    LSig_full = np.linalg.cholesky(Sig_full)
    LSigInv_full = solve_triangular(LSig_full, np.eye(LSig_full.shape[0]), lower=True, overwrite_b=True, check_finite=False)
    # compute the relative 2 norm error for mean and covariance
    mu_err = np.sqrt(((mu_full - mu_approx) ** 2).sum()) / np.sqrt((mu_full ** 2).sum())
    logsig_diag_err = np.linalg.norm(logsig_full - logsig_approx) / np.linalg.norm(logsig_full)
    cwise_mu_err = np.mean(np.fabs((mu_full - mu_approx) / mu_full))
    cwise_logsig_diag_err = np.mean(np.fabs((logsig_full - logsig_approx) / logsig_full))
    Sig_err = np.linalg.norm(Sig_approx - Sig_full, ord=2)/np.linalg.norm(Sig_full, ord=2)
    # compute gaussian reverse/forward KL
    rklw = KL(mu_approx, Sig_approx, mu_full, LSigInv_full.T.dot(LSigInv_full))
    fklw = KL(mu_full, Sig_full, mu_approx, LSigInv_approx.T.dot(LSigInv_approx))
    # # compute mmd discrepancies
    # gauss_mmd = stein.gauss_mmd(approx_samples, full_samples)
    # imq_mmd = stein.imq_mmd(approx_samples, full_samples)
    # # compute stein discrepancies
    # scores_approx = model.grad_log_joint(Z, approx_samples, np.ones(Z.shape[0]), sig, mu0, sig0)
    # gauss_stein = stein.gauss_stein(approx_samples, scores_approx)
    # imq_stein = stein.imq_stein(approx_samples, scores_approx)


    print('Saving ' + log_suffix)
    # results.save(arguments, t_build=t_build, t_per_sample=t_approx_per_sample, t_full_per_sample=t_full_per_itr,
    #              rklw=rklw, fklw=fklw, mu_err=mu_err, Sig_err=Sig_err)
    results.save(arguments, t_build=t_build, t_per_sample=t_approx_per_sample, t_full_per_sample=t_full_per_itr,
                 rklw=rklw, fklw=fklw, mu_err=mu_err, cwise_mu_err=cwise_mu_err,
                 logsig_diag_err=logsig_diag_err, cwise_logsig_diag_err=cwise_logsig_diag_err,
                 Sig_err=Sig_err
                 )
    print('')
    print('')

############################
############################
## Parse arguments
############################
############################

parser = argparse.ArgumentParser(description="Runs Hilbert coreset construction on a model and dataset")
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)

parser.add_argument('--data_num', type=int, default='100000', help='Dataset subsample to use')
parser.add_argument('--n_bases_per_scale', type=int, default=50, help="The number of Radial Basis Functions per scale")#TODO: verify help message
parser.add_argument('--alg', type=str, default='GIGA-LAP',
                    choices=['SVI', 'QNC', 'GIGA', 'UNIF', 'LAP','IHT','FULL','NAIVE', 'GIGA-LAP', 'IHT-LAP'],
                    help="The algorithm to use for solving sparse non-negative least squares")  # TODO: find way to make this help message autoupdate with new methods
parser.add_argument("--samples_inference", type=int, default=10000,
                    help="number of MCMC samples to take for actual inference and comparison of posterior approximations (also take this many warmup steps before sampling)")
parser.add_argument("--proj_dim", type=int, default=500,
                    help="The number of samples taken when discretizing log likelihoods")
parser.add_argument('--coreset_size', type=int, default=500, help="The coreset size to evaluate")
parser.add_argument('--opt_itrs', type=str, default=100,
                    help="Number of optimization iterations (for SVI)")
parser.add_argument('--newton_opt_itrs', type=str, default=20,
                    help="Number of optimization iterations (for QNC)")
parser.add_argument('--step_sched', type=str, default="lambda i : 1./(i+1)",
                    help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")

parser.add_argument('--trial', type=int, default=60,
                    help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--results_folder', type=str, default="results/",
                    help="This script will save results in this folder")
parser.add_argument('--verbosity', type=str, default="debug", choices=['error', 'warning', 'critical', 'info', 'debug'],
                    help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type=str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type=str, help="The Y axis of the plot")
plot_subparser.add_argument('--plot_title', type=str, help="The title of the plot")
plot_subparser.add_argument('--plot_x_label', type=str, help="The X axis label of the plot")
plot_subparser.add_argument('--plot_y_label', type=str, help="The Y axis label of the plot")
plot_subparser.add_argument('--plot_x_type', type=str, choices=["linear", "log"], default="log",
                            help="Specifies the scale for the X-axis")
plot_subparser.add_argument('--plot_y_type', type=str, choices=["linear", "log"], default="log",
                            help="Specifies the scale for the Y-axis.")
plot_subparser.add_argument('--plot_legend', type=str, help="Specifies the variable to create a legend for.")
plot_subparser.add_argument('--plot_height', type=int, default=850, help="Height of the plot's html canvas")
plot_subparser.add_argument('--plot_width', type=int, default=850, help="Width of the plot's html canvas")
plot_subparser.add_argument('--plot_type', type=str, choices=['line', 'scatter'], default='scatter',
                            help="Type of plot to make")
plot_subparser.add_argument('--plot_fontsize', type=str, default='32pt', help="Font size for the figure, e.g., 32pt")
plot_subparser.add_argument('--plot_toolbar', action='store_true', help="Show the Bokeh toolbar")
plot_subparser.add_argument('--groupby', type=str,
                            help='The command line argument group rows by before plotting. No groupby means plotting raw data; groupby will do percentile stats for all data with the same groupby value. E.g. --groupby Ms in a scatter plot will compute result statistics for fixed values of M, i.e., there will be one scatter point per value of M')

arguments = parser.parse_args()
arguments.func(arguments)
# run(arguments)

