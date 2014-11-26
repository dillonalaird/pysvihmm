from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from copy import deepcopy

from scipy.stats import bernoulli
from scipy.stats import norm

import hmmbatchcd as HMM
import gen_synthetic
from pybasicbayes.distributions import Gaussian, Multinomial
from test_utilities import *

try:
    from numpy.random import multivariate_normal as mvnrand
except IOError:
    from util import mvnrand


def test_global_update():
    # Global Update Test
    # Comment out the local update for this test. This test generates some data
    # from Gaussians, and then passes the same gaussians into the HMM to see
    # if the parameters stay the same after running the global updates. Fix the
    # var_q to the real distribution

    # This test doesn't seem to be working anymore. Note you should also
    # comment out the lower_bound since we're not running forward backwards

    K = 2

    # for generating the data
    tran = np.array([[0.9, 0.1], [0.1, 0.9]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.zeros(2),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=1,
                     nu_0=1)
    # set mu_0 near the actual mu_2?
    emit2 = Gaussian(mu=np.ones(2) * 10,
                     sigma=np.eye(2),
                     mu_0=np.ones(2) * 10,
                     sigma_0=np.eye(2),
                     kappa_0=1,
                     nu_0=1)
    emit = np.array([emit1, emit2])
    obs, sts, _ = gen_synthetic.generate_data(tran, emit, 100)

    # Try making the prior different from the true components
    emit_prior = Gaussian(mu=np.zeros(2),
                          sigma=np.eye(2),
                          mu_0=np.zeros(2),
                          sigma_0=np.eye(2),
                          kappa_0=1,
                          nu_0=1)
    #print emit_prior

    #prior_emit = np.array([emit1, emit2])
    #prior_emit = np.array([emit_prior, deepcopy(emit_prior)])
    prior_tran = np.array([[9, 1], [1, 9]])
    prior_init = np.ones(2)
    prior_emit = [Gaussian(mu =np.zeros(2),
                           sigma=np.eye(3),
                           mu_0=np.zeros(2),
                           sigma_0=np.eye(2),
                           kappa_0=0.01, nu_0=4) for i in xrange(2)]

    init_init = np.random.rand(K)
    init_init /= np.sum(init_init)
    init_tran = np.random.rand(K,K)
    init_tran /= np.sum(init_tran, axis=1)[:,np.newaxis]

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit,
            init_init=init_init, init_tran=init_tran)
    #hmm.var_q = gen_synthetic.generate_q(sts)

    hmm.infer()
    print "=============EMIT 0 MF FIELDS"
    print "mu_mf = ", hmm.var_emit[0].mu_mf
    print "simga_mf = ", hmm.var_emit[0].sigma_mf
    print "kappa_mf = ", hmm.var_emit[0].kappa_mf
    print "nu_mf = ", hmm.var_emit[0].nu_mf
    print "=============EMIT 1 MF FIELDS"
    print "mu_mf = ", hmm.var_emit[1].mu_mf
    print "simga_mf = ", hmm.var_emit[1].sigma_mf
    print "kappa_mf = ", hmm.var_emit[1].kappa_mf
    print "nu_mf = ", hmm.var_emit[1].nu_mf
    print "=============EMIT 0 FIELDS"
    print hmm.var_emit[0]
    print "=============EMIT 1 FIELDS"
    print hmm.var_emit[1]
    print ""
    print hmm.var_tran

def test_local_update():
    # Local Update Test
    # Comment out the global update for this test. Here we pass in the true
    # distributions (used to generate the data) as the prior emit with large
    # kappa_0 and nu_0 (this ensures that the priors are given more weight) and
    # we check to see if the variational q distribution converges to the true
    # state sequence.

    # for generating the data
    tran = np.array([[0.9, 0.1], [0.1, 0.9]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.zeros(2),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=100,
                     nu_0=100)
    # set mu_0 near the actual mu_2?
    emit2 = Gaussian(mu=np.ones(2) * 10,
                     sigma=np.eye(2),
                     mu_0=np.ones(2) * 10,
                     sigma_0=np.eye(2),
                     kappa_0=100,
                     nu_0=100)
    emit = np.array([emit1, emit2])
    obs, sts, _ = gen_synthetic.generate_data(tran, emit, 100)

    priors_emit = np.array([emit1, emit2])
    priors_tran = np.array([[9, 1], [1, 9]])
    prior_init = np.ones(2)
    hmm = HMM.VBHMM(obs, prior_init, priors_tran, priors_emit)
    hmm.infer()

    plot_MAP(hmm.var_x, obs)

def test_local_global():
    # Local and Global update test
    # This tests the local and global update on an obvious state sequence

    # for generating the data
    tran = np.array([[0.9, 0.1], [0.1, 0.9]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.zeros(2),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=100,
                     nu_0=100)
    # set mu_0 near the actual mu_2?
    emit2 = Gaussian(mu=np.ones(2) * 10,
                     sigma=np.eye(2),
                     mu_0=np.ones(2) * 10,
                     sigma_0=np.eye(2),
                     kappa_0=100,
                     nu_0=100)
    emit = np.array([emit1, emit2])
    #obs, sts, _ = gen_synthetic.generate_data(tran, emit, 10)
    nobs = 100
    sts = np.zeros(nobs)
    sts[(nobs//2):] = 1
    obs = np.empty((nobs,2))
    for i in xrange(nobs):
        obs[i,:] = emit[sts[i]].rvs()[0]

    mu_0 = np.zeros(2)
    sigma_0 = 50 * np.eye(2)
    kappa_0 = 0.01
    nu_0 = 4

    prior_emit = [Gaussian(mu_0=mu_0, sigma_0=sigma_0,
                     kappa_0=kappa_0, nu_0=nu_0) for i in xrange(len(emit))]
    prior_emit = np.array(prior_emit)
    prior_tran = np.array([[1, 1], [1, 1]])
    prior_init = np.ones(2)

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit)
    hmm.infer()

    print "=============EMIT 0 MF FIELDS"
    print "mu_mf = ", hmm.var_emit[0].mu_mf
    print "simga_mf = ", hmm.var_emit[0].sigma_mf
    print "kappa_mf = ", hmm.var_emit[0].kappa_mf
    print "nu_mf = ", hmm.var_emit[0].nu_mf
    print "=============EMIT 1 MF FIELDS"
    print "mu_mf = ", hmm.var_emit[1].mu_mf
    print "simga_mf = ", hmm.var_emit[1].sigma_mf
    print "kappa_mf = ", hmm.var_emit[1].kappa_mf
    print "nu_mf = ", hmm.var_emit[1].nu_mf
    print "=============EMIT 0 FIELDS"
    print hmm.var_emit[0]
    print "=============EMIT 1 FIELDS"
    print hmm.var_emit[1]

    print "True state sequence:"
    print sts
    print "Learned state sequence:"
    print np.argmax(hmm.var_x, axis=1).astype('float64')

    return hmm, sts, obs

def test_prior_strength():
    # Test Prior Strength
    # This tests the strength of the prior by increasing kappa_0 and nu_0. The
    # parameters should remain closer to the prior parameters for higher
    # kappa_0 and nu_0

    # GENERATE DATA
    tran = np.array([[0.9, 0.1], [0.1, 0.9]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.zeros(2),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=100,
                     nu_0=100)
    # set mu_0 near the actual mu_2?
    emit2 = Gaussian(mu=np.ones(2) * 10,
                     sigma=np.eye(2),
                     mu_0=np.ones(2) * 10,
                     sigma_0=np.eye(2),
                     kappa_0=100,
                     nu_0=100)
    emit = np.array([emit1, emit2])
    obs, sts, _ = gen_synthetic.generate_data(tran, emit, 100)

    prior_init = np.ones(2)
    prior_tran = np.ones((2,2))

    obs_mean = np.mean(obs, axis=0)
    obs_var = np.var(obs, axis=0)

    prior_emit = [Gaussian(mu_0=obs_mean.copy(),
                           sigma_0=0.75*np.diag(obs_var.copy()),
                           kappa_0=0.01, nu_0=4) for k in xrange(2)]
    prior_emit = np.array(prior_emit)

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit)
    hmm.infer()

    print "=============EMIT 0 FIELDS"
    print hmm.var_emit[0]
    print "=============EMIT 1 FIELDS"
    print hmm.var_emit[1]

    var_x = hmm.var_x.copy()
    if mistake_rate(sts, var_x) > 0.5:
        var_x = np.dstack((var_x[:,1], var_x[:,0]))[0]

    print "mistake rate = ", mistake_rate(sts, var_x)
    plot_var_q(var_x, sts)

def test_local_global_hard():
    # Tests the Local and Global Update on Difficult Data
    # This tests the local and global update on data generated from a
    # transition matrix that switches between states more often

    # GENERATE DATA
    K = 2
    D = 2
    tran = np.array([[0.3, 0.7], [0.3, 0.7]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.array([3, 3]),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=2,
                     nu_0=2)
    emit2 = Gaussian(mu=np.array([5, 8]),
                     sigma=np.eye(2) * 1.5,
                     mu_0=np.ones(2) * 5,
                     sigma_0=np.eye(2),
                     kappa_0=2,
                     nu_0=2)
    emit = np.array([emit1, emit2])

    nobs = 1001
    missfrac = 0.2
    obs, sts, mask = gen_synthetic.generate_data(tran, emit, nobs, miss=missfrac)

    prior_init = np.ones(2)
    prior_tran = np.ones((2,2))

    obs_mean = np.mean(obs, axis=0)

    mu_0 = obs_mean
    sigma_0 = 0.75*np.cov(obs.T)  # 0.75*np.diag(obs_var)
    kappa_0 = 0.01
    nu_0 = 4

    # Initialize from prior
    KK = K
    init_means = np.empty((KK,D))
    for k in xrange(KK):
        init_means[k,:] = mvnrand(mu_0, cov=sigma_0)
    prior_emit = np.array([Gaussian(mu=init_means[i,:], sigma=sigma_0,
                                    mu_0=mu_0, sigma_0=sigma_0,
                                    kappa_0=kappa_0, nu_0=nu_0)
                           for i in xrange(KK)])

    init_init = np.random.rand(K)
    init_init /= np.sum(init_init)
    init_tran = np.random.rand(K,K)
    init_tran /= np.sum(init_tran, axis=1)[:,np.newaxis]

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, mask=mask,
                    init_init=init_init, init_tran=init_tran)
    hmm.infer()

    print "=============EMIT 0 FIELDS"
    print hmm.var_emit[0]
    print "=============EMIT 1 FIELDS"
    print hmm.var_emit[1]

    print "True state sequence:"
    print sts
    lss = np.argmax(hmm.var_x, axis=1).astype('float64')
    d1 = lss - sts
    d2 = np.abs(lss-1.) - sts
    diff = d1
    diff_str = ""
    if np.abs(np.sum(np.abs(d2))) < np.abs(np.sum(np.abs(d1))):
        diff = d2
        diff_str = " (flipped learned to match)"

    print "Learned state sequence (matched):"
    print lss
    print "Differences" + diff_str
    print diff

    return hmm

if __name__ == "__main__":

    #test_global_update()
    #test_local_update()
    #test_local_global()
    #hmm, sts, obs = test_local_global()
    #test_prior_strength()
    hmm = test_local_global_hard()
