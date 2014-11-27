from __future__ import division

import numpy as np

import hmmbatchsgd as HMM
from pybasicbayes.distributions import Gaussian, Multinomial

try:
    from numpy.random import multivariate_normal as mvnrand
except IOError:
    from util import mvnrand

import gen_synthetic
from test_utilities import *


def test_local_global():
    print "============================TEST LOCAL GLOBAL============================="
    # for generating the data
    tran = np.array([[0.9, 0.1], [0.1, 0.9]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.zeros(2),
                     sigma=np.eye(2))
    # set mu_0 near the actual mu_2?
    emit2 = Gaussian(mu=np.ones(2)*5,
                     sigma=np.eye(2))
    emit = np.array([emit1, emit2])

    # here you can generate observations or just make the observations trivial
    # with half one state and half the other state
    obs, sts, _ = gen_synthetic.generate_data(tran, emit, 100)
    """
    nobs = 100
    sts = np.zeros(nobs)
    sts[(nobs//2):] = 1
    obs = np.empty((nobs,2))
    for i in xrange(nobs):
        obs[i,:] = emit[sts[i]].rvs()[0]

    """

    mu_0 = np.zeros(2)
    sigma_0 = 50*np.eye(2)
    kappa_0 = 0.01
    nu_0 = 4

    prior_emit = [Gaussian(mu_0=mu_0, sigma_0=sigma_0,
                     kappa_0=kappa_0, nu_0=nu_0) for i in xrange(len(emit))]
    prior_emit = np.array(prior_emit)
    prior_tran = np.array([[1, 1], [1, 1]])
    prior_init = np.ones(2)

    # Learning rate for gradient descent
    tau = 1.
    kappa = 0.7
    maxit = 150

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, tau=tau, 
                    kappa=kappa, maxit=maxit)
    hmm.infer()

    print "=============EMIT 0 FIELDS"
    print hmm.var_emit[0]
    print "=============EMIT 1 FIELDS"
    print hmm.var_emit[1]

    print "True state sequence:"
    print sts
    print "Learned state sequence:"
    print np.argmax(hmm.var_x, axis=1).astype('float64')
    print "Differences"
    print np.argmax(hmm.var_x, axis=1).astype('float64') - sts

    return hmm, sts, obs

def test_local_global_hard():
    print "==========================TEST LOCAL GLOBAL HARD=========================="
    # Just testing it on more difficult data

    # GENERATE DATA
    K = 2
    D = 2
    tran = np.array([[0.3, 0.7], [0.3, 0.7]])
    # use same gaussians to generate the data as we pass into the model
    emit1 = Gaussian(mu=np.array([3, 3]),
                     sigma=np.eye(2))
    emit2 = Gaussian(mu=np.array([5, 8]),
                     sigma=np.eye(2)*1.5)
    emit = np.array([emit1, emit2])

    missfrac = 0.2
    obs, sts, mask = gen_synthetic.generate_data(tran, emit, 100, miss=missfrac)

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

    tau = 1.
    kappa = 0.7
    maxit = 150

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, tau, kappa,
                    mask=mask, init_init=init_init, init_tran=init_tran,
                    maxit=maxit, verbose=True)
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

    return hmm, obs, sts

if __name__ == "__main__":

    #hmm, sts, obs = test_local_global()
    hmm, obs, sts = test_local_global_hard()


