
from __future__ import division

import sys
import time
import numpy as np

# Just for debugging
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from copy import deepcopy

from numpy import newaxis as npa
from scipy.special import digamma, gammaln

from hmmbase import VariationalHMMBase
from pybasicbayes import distributions as dist

import util

# This is for taking logs of things so we don't get -inf
eps = 1e-9

tau0 = 1.
kappa0 = 0.7


class VBHMM(VariationalHMMBase):
    """ Batch variational inference for finite hidden Markov models using
        natural gradient for global parameter updates.

        obs : observations
        x : hidden states
        init : initial distribution (only useful for multiple series)
        tran : transition matrix
        emit : emission distributions

        The user passes in the hyperparameters for the initial, transition and
        emission distribution. We then store these as hyperparameters, and make
        copies of them to use as the variational parameters, these are the
        parameters we're doing updates on.

        The user should have each unique observation indexed in the emission
        hyperparameters and have those corresponding indexes listed in the
        observations. This way the user wont have to provide a map from the
        indexes to the observations, also it's a lot easier to deal with
        indexes than observations.
    """

    @staticmethod
    def make_param_dict(prior_init, prior_tran, prior_emit, tau=tau0,
                        kappa=kappa0, mask=None):
        """ Given parameters make a dict that can be used to initialize an
            object.
        """
        return {'prior_init': prior_init, 'prior_tran': prior_tran,
                'prior_emit': prior_emit, 'mask': mask, 'tau': tau,
                'kappa': kappa}


    def __init__(self, obs, prior_init, prior_tran, prior_emit, tau=tau0,
                 kappa=kappa0, mask=None, init_init=None, init_tran=None,
                 epsilon=1e-8, maxit=100, verbose=False, sts=None):
        """ This initializes the HMMSVI object. Assume we have K states and T
            observations

            obs : T x D np array of the observations in D dimensions (Can
                  be a vector if D = 1).

            prior_init : 1 x K np array containing the prior parameters
                         for the initial distribution.  Use Dirichlet
                         hyperparameters.

            prior_tran : K x K np array containing the prior parameters
                          for the transition distributions. Use K dirichlet
                          hyperparameters (1 for each row).

            prior_emit : K x 1 np array containing the emission
                          distributions, these should be distributions from
                          pybasicbayes/distributions.py

            tau : Delay for learning rate, >= 0.

            kappa : Forgetting factor for learning rate, \in (.5,1].

            mask : 1-d bool array of length T indicating which observations are
                   missing.  1 means missing.

            init_init : 1-d array of size K. Initial initial distribution.  If
                        None, then use the mean of prior_init.

            init_tran : 2-d array of size K x K.  Initial transition matrix.
                        If None, then use the row-means of prior_tran.

            epsilon : Threshold to decide algorithm has converged.  Default
                      1e-8.

            maxit : Maximum number of iterations to run optimization.
                    Default is 100.
        """

        super(VBHMM, self).__init__(obs, prior_init, prior_tran, prior_emit,
                                    mask=mask, init_init=init_init,
                                    init_tran=init_tran, verbose=verbose,
                                    sts=sts)

        self.batch = self.obs

        self.elbo = -np.inf
        self.tau = tau
        self.kappa = kappa
        self.lrate = tau**(-kappa)  # (t + tau)^{-kappa}

        self.epsilon = epsilon
        self.maxit = maxit

        # Factor to multiply natural gradient by so that it's unbiased.  This
        # will depend on the probability of selecting the minibatch and will be
        # set when the minibatch is sampled.
        self.batchfactor = 1.

        # Have to initialize in the derived class because the sizes differ
        # whether it's a batch or stochastic algorithm.
        #self.var_x = np.random.rand(self.T, self.K)
        self.var_x = np.ones((self.T, self.K))
        self.var_x /= np.sum(self.var_x, axis=1)[:,np.newaxis]

        self.alpha_table = np.zeros((self.T, self.K))
        self.beta_table = np.zeros((self.T, self.K))
        self.c_table = np.zeros(self.T)

        self.lalpha = np.empty((self.T, self.K))
        self.lbeta = np.empty((self.T, self.K))
        self.lliks = np.empty((self.T, self.K))

        # The modified parameters used in the local update
        self.mod_init = np.zeros(self.K)
        self.mod_tran = np.zeros((self.K, self.K))

    def infer(self):
        """ Runs stochastic variational inference algorithm. This works with
            only a subset of the data.
        """

        self.obs_full = self.obs.copy()
        self.obs[self.mask,:] = np.nan

        epsilon = self.epsilon
        maxit = self.maxit

        self.elbo_vec = np.inf*np.ones(maxit)
        self.pred_logprob_mean = np.nan*np.ones(maxit)
        self.pred_logprob_std = np.nan*np.ones(maxit)

        self.iter_time = np.nan*np.ones(maxit)

        for it in xrange(maxit):
            
            start_time = time.time()

            # (t + tau)^{-kappa}
            self.lrate = (it + self.tau)**(-self.kappa)

            self.local_update()
            self.global_update()

            self.iter_time[it] = time.time() - start_time

            # Keep getting matrix not positive definite in lower_bound
            # function
            lb = self.lower_bound()

            if self.verbose:
                print "iter: %d, ELBO: %.2f" % (it, lb)
                sys.stdout.flush()

            if False:  # np.allclose(lb, self.elbo, atol=epsilon):
                break
            else:
                self.elbo = lb
                self.elbo_vec[it] = lb
                tmp = self.pred_logprob()
                if tmp is not None:
                    self.pred_logprob_mean[it] = np.mean(tmp)
                    self.pred_logprob_std[it] = np.std(tmp)

        lbidx = np.where(np.logical_not(np.isinf(self.elbo_vec)))[0]
        self.elbo_vec = self.elbo_vec[lbidx]
        self.pred_logprob_mean = self.pred_logprob_mean[lbidx]
        self.pred_logprob_std = self.pred_logprob_std[lbidx]
        self.iter_time = self.iter_time[lbidx]

        # Save Hamming distance
        if self.sts is not None:
            self.hamming, self.perm = self.hamming_dist(self.var_x, self.sts)

        self.obs = self.obs_full

    def global_update(self, batch=None):
        """ Perform global updates based on batch following the stochastic
            natural gradient.
        """

        if batch is None:
            batch = self.obs

        lrate = self.lrate
        #batchfactor = self.batchfactor

        # Perform stochastic gradient update on global params.

        # Initial state distribution
        self.var_init = self.prior_init + self.var_x[0,:]

        # Transition distribution
        # Convert to natural parameters
        nats_old = self.var_tran - 1.

        # Mean-field update
        tran_mf = self.prior_tran.copy()
        for t in xrange(1, self.T):
            tran_mf += np.outer(self.var_x[t-1,:], self.var_x[t,:])

        # Convert result to natural params
        nats_t = tran_mf - 1.

        # Perform update according to stochastic gradient
        # (Hoffman, pg. 17)
        nats_new = (1.-lrate)*nats_old + lrate*nats_t

        # Convert results back to moment params
        self.var_tran = nats_new + 1.

        # Emission distributions
        inds = np.logical_not(self.mask)
        for k in xrange(self.K):
            G = self.var_emit[k]

            # Do mean-field update for this component
            mu_mf, sigma_mf, kappa_mf, nu_mf = \
                    util.NIW_meanfield(G, batch[inds,:], self.var_x[inds,k])

            # Convert to natural parameters
            nats_t = util.NIW_mf_natural_pars(mu_mf, sigma_mf,
                                              kappa_mf, nu_mf)

            # Convert current estimates to natural parameters
            nats_old = util.NIW_mf_natural_pars(G.mu_mf, G.sigma_mf,
                                                G.kappa_mf, G.nu_mf)

            # Perform update according to stochastic gradient
            # (Hoffman, pg. 17)
            nats_new = (1.-lrate)*nats_old + lrate*nats_t

            # Convert new params into moment form and store back in G
            util.NIW_mf_moment_pars(G, *nats_new)
