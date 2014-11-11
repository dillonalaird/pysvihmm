
from __future__ import division

import sys
import time
import numpy as np

# This is just for debuggin
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from copy import deepcopy

from hmmbase import VariationalHMMBase
from pybasicbayes import distributions as dist

# This is for taking logs of things so we don't get -inf
eps = 1e-9


class VBHMM(VariationalHMMBase):
    """ Batch coordinate-descent variational inference for hidden Markov
        models.

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

    def __init__(self, obs, prior_init, prior_tran, prior_emit, mask=None,
                 init_init=None, init_tran=None, epsilon=1e-8, maxit=100,
                 verbose=False, sts=None):
        """
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

        self.epsilon = epsilon
        self.maxit = maxit

        # Need to do this in the derived class because the sizes are different
        # depending on if we're batch or stochastic.
        # var_x is a T x K matrix so the k'th column represents the k'th
        # state over all T observations. So q[t,k] would represent the weight
        # for the k'th state at time t.
        self.var_x = np.random.rand(self.T, self.K)
        self.var_x /= np.sum(self.var_x, axis=1)[:,np.newaxis]

        # These are all members so we don't have to reinitialize them every
        # time we call the forward and backward functions
        #self.alpha_table = np.zeros((self.T, self.K))
        #self.beta_table = np.zeros((self.T, self.K))
        #self.c_table = np.zeros(self.T)

        self.lalpha = np.empty((self.T, self.K))
        self.lbeta = np.empty((self.T, self.K))
        self.lliks = np.empty((self.T, self.K))

        # The modified parameters used in the local update
        self.mod_init = np.zeros(self.K)
        self.mod_tran = np.zeros((self.K, self.K))

        # Stuff for viterbi that we don't use.  It could go in hmmbase.py
        #self.a_table = np.zeros((self.T, self.K))
        #self.viterbi_table = np.zeros((self.T, self.K))

    def infer(self):
        """ Run batch variational with coordinate descent on the full data set.

            epsilon : [default 1e-8] when ELBO has converged
            maxit : [default 100] number of iterations to run

        """

        # Put nans in obs and restore afterwards
        self.obs_full = self.obs.copy()
        self.obs[self.mask,:]

        epsilon = self.epsilon
        maxit = self.maxit

        self.elbo_vec = np.inf*np.ones(maxit)
        self.pred_logprob_mean = np.nan*np.ones(maxit)
        self.pred_logprob_std = np.nan*np.ones(maxit)

        self.iter_time = np.nan*np.ones(maxit)

        for it in xrange(maxit):

            start_time = time.time()

            self.local_update()
            self.global_update()

            self.iter_time[it] = time.time() - start_time

            lb = self.lower_bound()
            if self.verbose:
                print "iter: %d, ELBO: %.2f" % (it, lb)
                sys.stdout.flush()

            if np.allclose(lb, self.elbo, atol=epsilon):
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

        # Restore original data
        self.obs = self.obs_full

    def global_update(self):
        """ This is the global update for the batch version. Here we're
            updating the hyperparameters of the variational distributions over
            the parameters of the HMM.
        """

        # Initial parameter update
        self.var_init = self.prior_init + self.var_x[0,:]

        # Transition parameter updates
        self.var_tran = self.prior_tran.copy()
        for t in xrange(1, self.T):
            self.var_tran += np.outer(self.var_x[t-1,:], self.var_x[t,:])

        # Emission parameter updates
        inds = np.logical_not(self.mask)
        for k in xrange(self.K):
            self.var_emit[k].meanfieldupdate(self.obs[inds,:], self.var_x[inds,k])
