
from __future__ import division

import abc
import types

import numpy as np
import numpy.linalg as npl

# This is just for debugging
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import cPickle as pkl
import itertools

from copy import deepcopy

from numpy import newaxis as npa
from scipy.special import digamma, gammaln

import scipy.spatial.distance as dist

import hmm_fast
import util

# This is for taking logs of things so we don't get -inf
eps = 1e-9



class VariationalHMMBase(object):
    """ Abstract base class for finite variational HMMs.  Provides the
        interface, basic structure and functions that all implementations will
        need.
    """
    __metaclass__ = abc.ABCMeta

    # Interface

    @abc.abstractmethod
    def global_update():
        pass

    @abc.abstractmethod
    def infer():
        """ Perform inference. """
        pass

    @staticmethod
    def make_param_dict(prior_init, prior_tran, prior_emit, mask=None):
        """ Given parameters make a dict that can be used to initialize an
            object.
        """
        return {'prior_init': prior_init, 'prior_tran': prior_tran,
                'prior_emit': prior_emit, 'mask': mask}

    def set_mask(self, mask):
        if mask is None:
            # All observations observed
            self.mask = np.zeros(self.obs.shape[0], dtype='bool')
        else:
            self.mask = mask.astype('bool')

    def __init__(self, obs, prior_init, prior_tran, prior_emit, mask=None,
                 init_init=None, init_tran=None, verbose=False, sts=None):
        """
            obs : T x D np array of the observations in D dimensions (Can
                  be a vector if D = 1).

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

            verbose : Default False.  Print out info while running.

            sts : 2d ndarray of length N.  True state sequence.
        """

        self.verbose = verbose

        self.sts = sts

        # Save the hyperparameters
        self.prior_init = deepcopy(prior_init).astype('float64')
        self.prior_tran = deepcopy(prior_tran).astype('float64')
        self.prior_emit = deepcopy(prior_emit)

        # Initialize global variational distributions.
        if init_init is None:
            self.var_init = prior_init / np.sum(prior_init)
        else:
            self.var_init = init_init.copy()

        if init_tran is None:
            self.var_tran = prior_tran / np.sum(prior_tran, axis=1)[:,np.newaxis]
        else:
            self.var_tran = init_tran.copy()

        # We copy the prior objects becase the mean and covariance are the
        # initial values which can be set randomly when the object is created.
        self.var_emit = deepcopy(prior_emit)

        # Save the observations
        self.obs = obs
        self.set_mask(mask)

        # Number of states
        self.K = prior_tran.shape[0]

        if obs.ndim == 1:
            self.T = obs.shape[0]
            self.D = 1
        elif obs.ndim == 2:
            self.T, self.D = obs.shape
        else:
            raise RuntimeError("obs must have 1 or 2 dimensions")

        self.elbo = -np.inf

    def set_data(self, obs, mask=None):
        self.obs = obs
        if mask is None:
            self.mask = np.zeros(self.obs.shape[0], dtype='bool')
        else:
            self.mask = mask

    def lower_bound(self):
        """ Compute variational lower-bound.  This is approximate when
            stochastic optimization is used.
        """

        elbo = 0.

        # Initial distribution (only if more than one series, so ignore for now)
        p_pi = self.prior_init
        p_pisum = np.sum(p_pi)
        q_pi = self.var_init
        q_pidg = digamma(q_pi + eps)
        q_pisum = np.sum(q_pi)
        dg_q_pisum = digamma(q_pisum + eps)

        # Energy
        pi_energy = (gammaln(p_pisum + eps) - np.sum(gammaln(p_pi + eps))
                     + np.sum((p_pi-1.)*(q_pidg - dg_q_pisum)))
        # Entropy
        pi_entropy = -(gammaln(q_pisum + eps) - np.sum(gammaln(q_pi + eps))
                       + np.sum((q_pi-1.)*(q_pidg - dg_q_pisum)))

        # Transition matrix (each row is Dirichlet so can do like above)
        p_A = self.prior_tran
        p_Asum = np.sum(p_A, axis=1)
        q_A = self.var_tran
        q_Adg = digamma(q_A + eps)
        q_Asum = np.sum(q_A, axis=1)
        dg_q_Asum = digamma(q_Asum + eps)

        A_energy = (gammaln(p_Asum + eps) - np.sum(gammaln(p_A + eps), axis=1)
                    + np.sum((p_A-1)*(q_Adg - dg_q_Asum[:,npa]), axis=1))
        A_entropy = -(gammaln(q_Asum + eps) - np.sum(gammaln(q_A + eps), axis=1)
                     + np.sum((q_A-1)*(q_Adg - dg_q_Asum[:,npa]), axis=1))
        A_energy = np.sum(A_energy)
        A_entropy = np.sum(A_entropy)

        # Emission distributions -- does both energy and entropy
        emit_vlb = 0.
        for k in xrange(self.K):
            emit_vlb += self.var_emit[k].get_vlb()

        # Data term and entropy of states
        # This amounts to the sum of the logs of the normalization terms from
        # the forwards pass (see Beal's thesis).
        # Note: We use minus here b/c c_table is the inverse of \zeta_t in Beal.
        #lZ = -np.sum(np.log(self.c_table + eps))

        # We don't need the minus anymore b/c this is 1/ctable
        lZ = np.sum(np.logaddexp.reduce(self.lalpha, axis=1))

        elbo = (pi_energy + pi_entropy + A_energy + A_entropy
                + emit_vlb + lZ)

        return elbo

    def local_update(self, obs=None, mask=None):
        """ This is the local update for the batch version. Here we're creating
            modified parameters to run the forward-backward algorithm on to
            update the variational q distribution over the hidden states.

            These are always the same, and if we really need to change them
            we'll override the function.
        """
        if obs is None:
            obs = self.obs
        if mask is None:
            mask = self.mask

        self.mod_init = digamma(self.var_init + eps) - digamma(np.sum(self.var_init) + eps)
        tran_sum = np.sum(self.var_tran, axis=1)
        self.mod_tran = digamma(self.var_tran + eps) - digamma(tran_sum[:,npa] + eps)

        # Compute likelihoods
        for k, odist in enumerate(self.var_emit):
            self.lliks[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs))

        # update forward, backward and scale coefficient tables
        self.forward_msgs()
        self.backward_msgs()

        self.var_x = self.lalpha + self.lbeta
        self.var_x -= np.max(self.var_x, axis=1)[:,npa]
        self.var_x = np.exp(self.var_x)
        self.var_x /= np.sum(self.var_x, axis=1)[:,npa]

    def FFBS(self, var_init):
        """ Forward Filter Backward Sampling to simulate state sequence.
        """
        obs = self.obs
        T = self.T
        K = self.K
        A = self.var_tran

        mod_init = digamma(var_init + eps) - digamma(np.sum(var_init) + eps)
        tran_sum = np.sum(self.var_tran, axis=1)
        mod_tran = digamma(self.var_tran + eps) - digamma(tran_sum[:,npa] + eps)

        lalpha = np.empty((T, K))
        lliks = np.empty((T, K))
        # Compute likelihoods
        for k, odist in enumerate(self.var_emit):
            lliks[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs))

        lalpha[0,:] = mod_init + lliks[0,:]

        for t in xrange(1,self.T):
            lalpha[t] = np.logaddexp.reduce(lalpha[t-1] + np.log(A+eps).T, axis=1) + lliks[t]

        z = np.empty(T, dtype=np.int_)
        lp = lalpha[T-1,:] - np.max(lalpha[T-1,:])
        p = np.exp(lp)
        p /= np.sum(p)
        z[T-1] = np.random.choice(K, p=p)
        for t in xrange(T-2, -1, -1):
            lp = lalpha[t,:] + np.log(A[:,z[t+1]]+eps)
            lp -= np.max(lp)
            z[t] = np.random.choice(K, p=p)

        return z

    def forward_msgs(self, obs=None, mask=None):
        """ Creates an alpha table (matrix) where
            alpha_table[i,j] = alpha_{i}(z_{i} = j) = P(z_{i} = j | x_{1:i}).
            This also creates the scales stored in c_table. Here we're looking
            at the probability of being in state j and time i, and having
            observed the partial observation sequence form time 1 to i.

            obs : iterable of observation indices.  If None defaults to
                    all timestamps.

            See: http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
                 for an explanation of forward-backward with scaling.

            Override this for specialized behavior.
        """

        if obs is None:
            obs = self.obs
        if mask is None:
            mask = self.mask

        ltran = self.mod_tran
        ll = self.lliks

        lalpha = self.lalpha

        lalpha[0,:] = self.mod_init + ll[0,:]

        for t in xrange(1,self.T):
            lalpha[t] = np.logaddexp.reduce(lalpha[t-1] + ltran.T, axis=1) + ll[t]

    def backward_msgs(self, obs=None, mask=None):
        """ Creates a beta table (matrix) where
            beta_table[i,j] = beta_{i}(z_{i} = j) = P(x_{i+1:T} | z_{t} = j).
            This also scales the probabilies. Here we're looking at the
            probability of observing the partial observation sequence from time
            i+1 to T given that we're in state j at time t.

            Override this for specialized behavior.
        """

        if obs is None:
            obs = self.obs
        if mask is None:
            mask = self.mask

        ltran = self.mod_tran
        ll = self.lliks

        lbeta = self.lbeta
        lbeta[self.T-1,:] = 0.

        for t in xrange(self.T-2,-1,-1):
            np.logaddexp.reduce(ltran + lbeta[t+1] + ll[t+1], axis=1,
                                out=lbeta[t])

    def pred_logprob(self):
        """ Compute vector of predictive log-probabilities of data marked as
            missing in the `mask` member.

            Returns None if no missing data.
        """
        K = self.K
        obs = self.obs
        mask = self.mask
        nmiss = np.sum(mask)
        if nmiss == 0:
            return None

        logprob = np.zeros((nmiss,K))

        for k, odist in enumerate(self.var_emit):
            logprob[:,k] = np.log(self.var_x[mask,k]+eps) + odist.expected_log_likelihood(obs[mask,:])

        return np.mean(np.logaddexp.reduce(logprob, axis=1))

    def full_local_update(self):
        self.local_update()
        return self.var_x

    def hamming_dist(self, full_var_x, true_sts):
        """ This function returns the hamming distance between the full
            variational distribution on the states, and the true state
            sequence, after matching via the munkres algorithm
            
            full_var_x: variational distribution of state sequence.  Generate
                        it with self.full_local_update().

            true_sts: true state sequence

            Returns float with hamming distance and best permutation to match
            the states.
        """

        state_sq = np.argmax(full_var_x, axis=1).astype(int) #these are learned states
        best_match = util.munkres_match(true_sts, state_sq, self.K)
        return dist.hamming(true_sts, best_match[state_sq]), best_match

    def KL_L2_gaussian(self, emit_true, permutation):
        """ This functions computes the KL divergence between the variational
            Gaussian distribution and given true Gaussian distribution (input).
            It also returns the total l2 distance between the means
        
            emit_true : iterable of true emission distributions

            permutation : best matching permutation (see above)

            Returns
            
              - KL : The KL-divergence between the estimated and true emission
                     distributions
              - distance_mus : L_2 distance between true and estimated emission
                               means
        """
        
        KL = 0                  #running sum of total KL divergence
        distance_mus = 0        #running sum of just l2 distance between means
        dim = len(self.var_emit[1].mu)
        for k, k2 in enumerate(permutation):
            k = permutation[k2]     #index in permutation corresponding to true
            diffmeans = emit_true[k].mu - self.var_emit[k2].mu
            distance_mus += npl.norm(diffmeans)     #l2 distance
            sig_emit_inv = npl.inv(emit_true[k].sigma)
            KL += .5* (np.trace( np.dot( sig_emit_inv, self.var_emit[k2].sigma ) ) + np.dot( diffmeans, np.dot(sig_emit_inv, diffmeans) ) - dim - np.log( npl.det(self.var_emit[k2].sigma) / npl.det(emit_true[k].sigma) ) )  
        return KL, distance_mus

    def A_dist(self, A_true, perm):
        """ This computes the frobenius norm of the difference between learned
            transition matrix and true transition matrix, after permuting the
            columns properly to match up.

            A_true : true transition matrix

            perm : best matching permutation of estimated states (see
                         hamming_dist)
            
        """
        A = self.var_tran / np.sum(self.var_tran, axis=1)[:,np.newaxis]
        #permute the matrix to match
        A_true = A_true[np.ix_(perm, perm)]
        return npl.norm(A_true - A)


# Add cython optimized methods
VariationalHMMBase.ffbs_fast = \
    types.MethodType(hmm_fast.FFBS, None, VariationalHMMBase)
