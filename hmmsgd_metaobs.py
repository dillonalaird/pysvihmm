from __future__ import division

import sys
import time
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy.spatial.distance as dist


# Just for debugging
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from numpy import newaxis as npa
from scipy.special import digamma, gammaln

from hmmbase import VariationalHMMBase
from pybasicbayes.distributions import Gaussian, Categorical

import util

# This is for taking logs of things so we don't get -inf
eps = 1e-9

tau0 = 1.
kappa0 = 0.7
metaobs_half0 = 1
mb_sz0 = 1

# There is a bug in python 2.7.3 that can't pickle namedtuples in a backwards
# compatible way.  So make a class instead...
#from collections import namedtuple
# A metaobservation is a tuple with a min index and a max index (inclusive).
# This means code that uses these will need to add 1 to i2 so that it's
# included in ranges and slices.
#MetaObs = namedtuple('MetaObs', ['i1', 'i2'])


class MetaObs(object):
    def __init__(self, i1, i2):
        self.i1 = i1
        self.i2 = i2

class VBHMM(VariationalHMMBase):
    """ Stochastic variational inference for finite hidden Markov models using
        natural gradient for global parameter updates.  Consecutive groups of
        nodes are sampled as a "meta-observation".

        obs : observations
        x : hidden states
        init : initial distribution (only useful for multiple series)
        tran : transition matrix
        emit : emission distributions
    """

    @staticmethod
    def make_param_dict(prior_init, prior_tran, prior_emit, tau=tau0,
                        kappa=kappa0, metaobs_half=metaobs_half0, mb_sz=mb_sz0,
                        mask=None):
        """ Given parameters make a dict that can be used to initialize an
            object.
        """
        return {'prior_init': prior_init, 'prior_tran': prior_tran,
                'prior_emit': prior_emit, 'mask': mask, 'tau': tau,
                'kappa': kappa, 'metaobs_half': metaobs_half, 'mb_sz': mb_sz}

    def set_metaobs_fun(self):
        if self.metaobs_fun_name == 'unif':
            self.metaobs_fun = self.metaobs_unif
        elif self.metaobs_fun_name == 'noverlap':
            self.metaobs_fun = self.metaobs_noverlap
        else:
            raise RuntimeError("Unknown value for metaobs_fun: %s" % (self.metaobs_fun_name,))

    def __init__(self, obs, prior_init, prior_tran,
                 prior_emit, tau=tau0, kappa=kappa0,
                 metaobs_half=metaobs_half0, mb_sz=mb_sz0, mask=None,
                 full_predprob=False, init_init=None, init_tran=None,
                 maxit=100, verbose=False, adagrad=False, metaobs_fun='unif',
                 seed=None, sts=None, fullpred_freq=10, fullpred_sched=None,
                 growBuffer=False, bufferBudget=False):
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

            metaobs_half : Metaobs will be of size 2*metaobs_half + 1, must be
                           >= 1.
            mb_sz : Number of meta-observations in a minibatch.

            mask : 1-d bool array of length T where True indicates missing
                   data.

            full_predprob : Bool indicating whether to compute the predictive
                            log-probability of the full data set at each
                            iteration.  Result will be stored in
                            pred_logprob_full_mean and pred_logprob_full_std.
                            Default is False.

            init_init : 1-d array of size K. Initial initial distribution.  If
                        None, then use the mean of prior_init.

            init_tran : 2-d array of size K x K.  Initial transition matrix.
                        If None, then use the row-means of prior_tran.

            maxit : Maximum number of iterations to run SVI for.

            verbose : Print info while running.  Default False.

            adagrad : Use adaptive gradient.  Default False

            ada_G : If adagrad = true, the weight matrix for scaling stepsize

            metaobs_fun : String of which meta-observation sampling functio to
                          use.  Default 'unif'.
                          Possible values: 'unif', 'noverlap'

            seed : Random number seed.  Default 0.

            sts : True state sequence (for debugging)

            fullpred_freq : Integer, Number of iterations between computing
                            full predictivie log-probability, default 10.

            growBuffer : Bool, Grow buffer around metaobs to lower error at
                         end points.
        """

        np.random.seed(seed)
        self.seed = seed
        
        super(VBHMM, self).__init__(obs, prior_init, prior_tran,
                                    prior_emit, mask=mask, init_init=init_init,
                                    init_tran=init_tran, verbose=verbose,
                                    sts=sts)

        self.elbo = -np.inf
        self.tau = tau
        self.kappa = kappa
        self.lrate = tau**(-kappa)  # (t + tau)^{-kappa}
        self.full_predprob = full_predprob
        self.fullpred_freq = fullpred_freq
        if fullpred_sched is not None:
            self.fullpred_sched = fullpred_sched
        else:
            # By default do every 10 iterations
            self.fullpred_sched = np.arange(0, maxit, 10)

        self.mataobs_fun_name = metaobs_fun
        if metaobs_fun == 'unif':
            self.metaobs_fun = self.metaobs_unif
            self.metaobs_fun_name = 'unif'
        elif metaobs_fun == 'noverlap':
            self.metaobs_fun = self.metaobs_noverlap
            self.metaobs_fun_name = 'noverlap'
        else:
            raise RuntimeError("Unknown value for metaobs_fun: %s" % (metaobs_fun,))

        # TODO: Initialize adagrad stuff
        # If using adagrad initialize sufficient statistics
        self.adagrad = adagrad
        if adagrad:
            self.ada_G = 1.0*np.ones(prior_tran.shape)  #initialize with 1's the weight matrix

        self.maxit = maxit

        self.growBuffer = growBuffer
        self.bufferBudget = bufferBudget

        if metaobs_half < 1:
            raise RuntimeError("metaobs (%d) must be >= 1." % (metaobs_half,))
        self.metaobs_half = metaobs_half
        self.mb_sz = mb_sz
        self.cur_mo = None

        # Factor to multiply natural gradient by so that it's unbiased.  This
        # will depend on the probability of selecting the minibatch and will be
        # set when the minibatch is sampled.
        self.batchfactor = 1.

        metaobs_sz = 2*metaobs_half + 1
        self.var_x = np.random.rand(metaobs_sz, self.K)
        #self.var_x = np.ones((metaobs_sz, self.K))
        self.var_x /= np.sum(self.var_x, axis=1)[:,np.newaxis]

        self.lalpha = np.empty((metaobs_sz, self.K))
        self.lbeta = np.empty((metaobs_sz, self.K))
        self.lliks = np.empty((metaobs_sz, self.K))

    def metaobs_unif(self, N, L, n):
        """ Sample n basic (possibly overlapping) meta-observations of length
            2L as a minibatch.  N is the length of the observation sequence.
        """

        # Region of points that we can select a meta-observation of size L.
        ll = L
        uu = N - 1 - L

        # Pick centers of meta-observation at random.
        c_vec = npr.randint(ll, uu+1, n)
        minibatch = list()

        # Construct meta-observations as named tuples.
        for c in c_vec:
            minibatch.append(MetaObs(c-L,c+L))

        return minibatch

    def metaobs_noverlap(self, N, L, n):
        """ Sample n non-overlapping meta-observations of length 2L as
            minibatch. 
            
            The function assumes that it's possible draw non-overlapping
            meta-observations.  So n should be pretty small relative to N.
        """

        # Region of points that we can select a meta-observation of size L.
        ll = L
        uu = N - 1 - L
        
        c_vec = np.inf * np.ones(n)
        minibatch = list()
        # First meta-observation is uniform random
        c = npr.randint(ll, uu+1, 1)[0]
        minibatch.append(MetaObs(c-L,c+L))

        for i in xrange(n):
            c = npr.randint(ll, uu+1, 1)[0]
            while np.any(np.abs(c_vec - c) <= L):
                c = npr.randint(ll, uu+1, 1)[0]

            c_vec[i] = c
            minibatch.append(MetaObs(c-L,c+L))

        return minibatch

    def local_lower_bound(self):
        """ Contribution of meta-observation to approximate lower bound
            
            approx. lower bound = local + global
        """
        # Data term and entropy of states
        # This amounts to the sum of the logs of the normalization terms from
        # the forwards pass (see Beal's thesis).
        # Note: We use minus here b/c c_table is the inverse of \zeta_t in Beal.
        # c_table is based on the current meta-observation, so we don't need to
        # pass it.
        #return -np.sum(np.log(self.c_table + eps))

        # We don't need the minus anymore b/c this is 1/ctable
        return np.sum(np.logaddexp.reduce(self.lalpha, axis=1))

    def global_lower_bound(self):
        """ Contribution of global parameters to approximate lower bound
        """
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

        return A_energy + A_entropy + emit_vlb

    def infer(self, adaptive=False, perIter=10, epsilon=1e-6, minHalfL=1,
              avgResidual=False, Lincrement=1, Lcutoff=1000):
        """ Runs stochastic variational inference algorithm. This works with
            only a subset of the data.

            mb_gen : Generator to sample minibatches.

            -- We should be able to determine this from the minibatches
            R : This is defined as T / |S| where T is the size of the entire
                dataset and |S| is the size of each sample.
        """
        np.random.seed(self.seed)

        growBuffer = self.growBuffer
        bufferBudget = self.bufferBudget

        #self.obs_full = self.obs.copy()
        #self.obs[self.mask,:] = np.nan

        maxit = self.maxit

        # Set the sampling function if it's None
        if self.metaobs_fun is None:
            self.set_metaobs_fun()

        # Initialize to nan so that we can detect errors and if there's no
        # missing data somewhere.
        self.elbo_vec = np.inf*np.ones(maxit)

        #self.pred_logprob_mean = np.inf*np.ones(maxit)
        #self.pred_logprob_std = np.inf*np.ones(maxit)

        #if self.full_predprob:
        #    self.pred_logprob_full_mean = np.inf*np.ones(maxit)
        #    self.pred_logprob_full_std = np.inf*np.ones(maxit)

        K = self.K
        D = self.D

        self.iter_time = np.inf * np.ones(maxit)

        mb_sz = self.mb_sz
        L = self.metaobs_half
        miniL = L

        if (L is None or adaptive) and growBuffer:
            raise RuntimeError("Cannot specify both adaptive and buffer simultaneously!")


        for it in xrange(maxit):
            start_time = time.time()

            # Update learning rate, (t + tau)^{-kappa}
            self.lrate = (it + self.tau)**(-self.kappa)

            #need to specify how often we call select L, and a max L
            if L is None or (adaptive and it % perIter == 0):
                L = self.select_L(mb_sz, epsilon=epsilon, minHalfL=minHalfL, 
                    avgResidual=avgResidual, Lincrement=Lincrement, Lcutoff= Lcutoff)

                #re-initialize to proper size:
                metaobs_sz = 2*L + 1
                self.var_x = np.random.rand(metaobs_sz, self.K)
                #self.var_x = np.ones((metaobs_sz, self.K))
                self.var_x /= np.sum(self.var_x, axis=1)[:,np.newaxis]
                self.lalpha = np.empty((metaobs_sz, self.K))
                self.lbeta = np.empty((metaobs_sz, self.K))
                self.lliks = np.empty((metaobs_sz, self.K))
                miniL = L
                #print miniL

            #L must be specified in order for this function to work:
            if growBuffer and it % perIter == 0:
                """currently, the number of indices to grow around is ALWAYS self.mb_sz
                    the buffer_budget functions resizes mb_sz to be used for minibatches
                    in the steps between
                """
                bufferL = self.select_buffer(self.mb_sz, epsilon=epsilon, halfL=L, 
                    avgResidual=avgResidual, Lincrement=Lincrement, Lcutoff=Lcutoff)
                
                #print "bufferL: %d" % bufferL
                #re-initialize to proper size:
                #bufferL = 20
                metaobs_sz = 2*bufferL + 1
                self.var_x = np.random.rand(metaobs_sz, self.K)
                #self.var_x = np.ones((metaobs_sz, self.K))
                self.var_x /= np.sum(self.var_x, axis=1)[:,np.newaxis]
                self.lalpha = np.empty((metaobs_sz, self.K))
                self.lbeta = np.empty((metaobs_sz, self.K))
                self.lliks = np.empty((metaobs_sz, self.K))
                miniL = bufferL

                #scale down the minibatch size based on the computational budget
                if bufferBudget:
                    mb_sz = self.buffer_budget(bufferL)
                #otherwise, the mb_sz is constant throughout

            #print mb_sz
            minibatch = self.metaobs_fun(self.T, miniL, mb_sz)

            lb = 0.

            A_inter = np.zeros_like(self.var_tran)

            emit_inter = [util.NIW_zero_nat_pars(self.var_emit[0])
                          for k in xrange(K)]

            for data in minibatch:

                self.cur_mo = data

                # Compuate stationary distribution of mean of distribution for
                # current transition matrix.  We'll use this to initialize the
                # first and last observation in the forwards and backwards
                # passes.
                A_mean = self.var_tran / np.sum(self.var_tran, axis=1)[:,npa]
                ew, ev = np.linalg.eig(A_mean.T)  # B/c computes right evs.
                ew_dec = np.argsort(ew)[::-1]  # Reverse to get decreasing
                # Store in mod_init so previous code works.
                # Take abs b/c not guarenteed to get positive components.
                self.var_init = np.abs(ev[:,ew_dec[0]])

                # Local update for this meta-observation, whole buffer gets 
                # updated here if growBuffer= True
                self.local_update(metaobs=data)

                # Natural gradient for this meta-observation
                if growBuffer:
                    A_i, e_i = self.intermediate_pars_buffer(data, bufferL, L)
                else:
                    A_i, e_i = self.intermediate_pars(data)
                
                A_inter += A_i

                for k in xrange(K):
                    emit_inter[k] += e_i[k]

                # Approximate lower bound contribution of this meta-observation
                lb += self.local_lower_bound()

            # Global update for this mini batch
            self.global_update(A_inter, emit_inter)

            self.iter_time[it] = time.time() - start_time

            # Approximate lower bound contribution of this mini batch
            lb += self.global_lower_bound()
            self.elbo_vec[it] = lb

            if self.verbose:
                print "iter: %d, ELBO: %.2f" % (it, lb)
                sys.stdout.flush()

            # Predictive log-probability for missing data in the
            # meta-observations that are part of this minibatch.
            #pred_logprob = None
            #for data in minibatch:
            #    tmp = self.pred_logprob(metaobs=data)
            #    if tmp is not None:
            #        if pred_logprob is None:
            #            pred_logprob = tmp
            #        else:
            #            pred_logprob = np.hstack((pred_logprob, tmp))

            #if pred_logprob is not None:
            #    self.pred_logprob_mean[it] = np.nanmean(pred_logprob)
            #    self.pred_logprob_std[it] = np.nanstd(pred_logprob)

            # Full predictive log-prob every 10 iterations.
            # Do not do this for large datasets
            if self.full_predprob and it in self.fullpred_sched:
                tmp = self.pred_logprob_full()
                self.pred_logprob_full_mean[it] = np.nanmean(tmp)
                self.pred_logprob_full_std[it] = np.nanstd(tmp)

        #self.obs = self.obs_full

        # Compute necessary stats for the whole observation sequence
        # Don't want to do this when handling large data sets
        #self.var_x_full = self.full_local_update()

        # Save Hamming distance
        #if self.sts is not None:
        #    self.hamming, self.perm = self.hamming_dist(self.var_x_full,
        #                                                self.sts)

        # So that the hmm object can be pickled
        self.metaobs_fun = None

    def local_update(self, metaobs=None):
        """ Local update that handles minibatches.  This needed to be
            reimplemented because forward_msgs and backward_msgs need to be
            specialized.
        """

        if metaobs is None:
            loff = 0
            uoff = self.T-1
        else:
            loff, uoff = metaobs.i1, metaobs.i2

        # update the modified parameter tables (don't do emissions b/c
        # pybasicbayes takes care of those).
        # Don't overwrite mod_init b/c we stored something in it
        self.mod_init = digamma(self.var_init + eps) - digamma(np.sum(self.var_init) + eps)
        tran_sum = np.sum(self.var_tran, axis=1)
        self.mod_tran = digamma(self.var_tran + eps) - digamma(tran_sum[:,npa] + eps)

        obs = self.obs
        # Compute likelihoods
        for k, odist in enumerate(self.var_emit):
            self.lliks[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs[loff:(uoff+1),:]))

        # update forward, backward and scale coefficient tables
        self.forward_msgs(metaobs=metaobs)
        self.backward_msgs(metaobs=metaobs)

        # update weights
        self.var_x = self.lalpha + self.lbeta
        self.var_x -= np.max(self.var_x, axis=1)[:,npa]
        self.var_x = np.exp(self.var_x)
        self.var_x /= np.sum(self.var_x, axis=1)[:,npa]

    def select_L(self, numIndices=1, epsilon=1e-5, minHalfL=1, avgResidual=False, Lincrement=1, Lcutoff=1000):
        #select indices randomly

        #don't sample from endpoints
        indices = npr.choice(self.T-2*minHalfL-1, size=numIndices) + minHalfL
        maxL = -1
        
        if not avgResidual:
            for ind in indices:
                q_diff = np.finfo(np.float_).max
                L = minHalfL
                q_old = self.get_marginal( self.get_local_messages(ind,minHalfL) , minHalfL ) 
            
                while True: #we can't let it grow past endpoints
                    if ind-L < 1+Lincrement or ind+L+Lincrement+1 > self.T or L > Lcutoff:
                        break
                    if q_diff < epsilon:
                        break
                    L += Lincrement
                    q_new = self.get_marginal( self.get_local_messages(ind,L), L )
                    q_diff = np.sum(np.abs(q_new - q_old)) #1-norm
                    q_old = q_new

                maxL = np.maximum(maxL, L)
        
        else: 
            for ind in indices:
                L = minHalfL
                q_old = self.get_marginal( self.get_local_messages(ind,minHalfL), minHalfL)
                count = 0
                q_running_av = 0.
                q_running_old = 0.
                while True: #we can't let it grow past endpoints
                    if ind-L < 1+Lincrement or ind+L+Lincrement+1 > self.T or L > Lcutoff:
                        break
                    count += 1
                    if count > 1:
                        if (q_running_av-q_running_old)/(count-1) < epsilon:
                            break
                    #otherwise, proceed and grow the metaobs
                    L += Lincrement
                    q_new = self.get_marginal( self.get_local_messages(ind,L), L )
                    q_running_old = q_running_av
                    q_running_av += np.sum(np.abs(q_new - q_old))
                    q_old = q_new

                maxL = np.maximum(maxL, L)

        return maxL

    def buffer_budget(self, halfL, budget = 400):
        """This divides a computational budget of total observations 
            by halflength to set number of metaobservations
            to be used per minibatch between trials of growBuffer
            Returns an integer that is to be used at minibatch size
        """
        return int(np.ceil(budget/(2*halfL+1)))

    def select_buffer(self, numIndices=1, epsilon=1e-5, halfL=10,
                      avgResidual=False, Lincrement=1, Lcutoff=1000):
        """ 
            Select width of buffered metaobservation such that the local
            messages inside the original endpoints are \epsilon approximations
            to the true messages.

            halfL : int > 0, half length of metaobservation.
            Document the rest of these

            Returns:
                - Length of buffer
        """

        #select indices randomly
        #don't sample from endpoints
        indices = npr.choice(self.T-2*halfL-1, size=numIndices) + halfL
        maxL = -1
        
        if not avgResidual:
            for ind in indices:
                q_diff_left = np.finfo(np.float_).max
                q_diff_right = np.finfo(np.float_).max

                bufferL = halfL
                q_old_left = self.get_marginal( self.get_local_messages(ind,halfL), bufferL - halfL)
                q_old_right = self.get_marginal( self.get_local_messages(ind,halfL), bufferL + halfL )

                while True: #we can't let it grow past endpoints
                    if ind-bufferL < 1+Lincrement or ind+bufferL+Lincrement+1 > self.T or bufferL > Lcutoff:
                        break
                    if q_diff_left < epsilon and q_diff_right < epsilon:
                        break
                    bufferL += Lincrement
                    var_new = self.get_local_messages(ind,bufferL)
                    q_new_left = self.get_marginal( var_new , bufferL - halfL)
                    q_new_right = self.get_marginal( var_new , bufferL + halfL)
                    
                    q_diff_left = np.sum(np.abs(q_new_left - q_old_left))
                    q_diff_right = np.sum(np.abs(q_new_right - q_old_right))

                    q_old_left  = q_new_left
                    q_old_right = q_new_right

                maxL = np.maximum(maxL, bufferL)
        
        else: 
            for ind in indices:
                bufferL = halfL

                q_old_left = self.get_marginal( self.get_local_messages(ind,halfL), bufferL - halfL)
                q_old_right = self.get_marginal( self.get_local_messages(ind,halfL), bufferL + halfL )

                count = 0
                q_running_av_left = 0.
                q_running_av_right = 0.
                q_running_old_left = 0.
                q_running_old_right = 0.

                while True: #we can't let it grow past endpoints
                    if ind-bufferL < 1+Lincrement or ind+bufferL+Lincrement+1 > self.T or bufferL > Lcutoff:
                        break
                    count += 1
                    if count > 1:
                        if (q_running_av_left-q_running_old_left)/(count-1) < epsilon and (q_running_av_right-q_running_old_right)/(count-1) < epsilon:
                            break
                    #otherwise, proceed and grow the metaobs
                    bufferL += Lincrement

                    q_new_left = self.get_marginal( var_new , bufferL - halfL)
                    q_new_right = self.get_marginal( var_new , bufferL + halfL)

                    q_running_old_left = q_running_av_left
                    q_running_old_right = q_running_av_right

                    q_running_av_left += np.sum(np.abs(q_new_left - q_old_left))
                    q_running_av_right += np.sum(np.abs(q_new_right - q_old_right))
                    q_old_left = q_new_left
                    q_old_right = q_new_right

                maxL = np.maximum(maxL, bufferL)

        return maxL

    def get_local_messages(self, ind, halflength):
        """ Computes local update but returns the variational distribution over
            the middle index.

            ind : int > 0, center of metaobservations.
            halflength : int > 0, half the width of the metaobservation, total
                         width will by 2*halflength + 1.
        """

        # update the modified parameter tables (don't do emissions b/c
        # pybasicbayes takes care of those).
        # Don't overwrite mod_init b/c we stored something in it
        mod_init = digamma(self.var_init + eps) - digamma(np.sum(self.var_init) + eps)
        tran_sum = np.sum(self.var_tran, axis=1)
        mod_tran = digamma(self.var_tran + eps) - digamma(tran_sum[:,npa] + eps)

        obs = self.obs
        loff= ind-halflength
        uoff= ind+halflength

        # Compute likelihoods
        lliks = np.empty( (2*halflength+1, self.K ) )
        for k, odist in enumerate(self.var_emit):
            lliks[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs[loff:(uoff+1),:]))

        # update forward, backward and scale coefficient tables
        a = self.get_forward(MetaObs(loff,uoff), lliks, mod_tran, mod_init)
        b = self.get_backward(MetaObs(loff,uoff), lliks, mod_tran)

        # update weights
        var_x = a + b
        var_x -= np.max(var_x, axis=1)[:,npa]
        var_x = np.exp(var_x)
        var_x /= np.sum(var_x, axis=1)[:,npa]

        #returns a vector of probabilities of length K, over the index 
        #return np.squeeze(var_x[halflength,:])
        return var_x

    def get_marginal(self, var_over_x, index, ):
        """ returns a vector of probabilities of length K, over 
            the index, to be used on
            the value returned by get_local_messages (a variational 
            distribution over a whole metaobs)
        """
        return np.squeeze( var_over_x[index,:] )


    def get_forward(self, metaobs, lliks, mod_tran, mod_init):
        """ Creates an alpha table (matrix) where
            alpha_table[i,j] = alpha_{i}(z_{i} = j) = P(z_{i} = j | x_{1:i}).
            This also creates the scales stored in c_table. Here we're looking
            at the probability of being in state j and time i, and having
            observed the partial observation sequence form time 1 to i.

            metaobs : Optional metaobservation to specify a consecutive subset
                      of the data.

            See: http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
                 for an explanation of forward-backward with scaling.
        """

        if metaobs is None:
            loff = 0
            uoff = self.T-1
        else:
            loff, uoff = metaobs.i1, metaobs.i2

        ltran = mod_tran
        ll = lliks

        lalpha = np.empty( (uoff-loff+1, self.K) )

        lalpha[0,:] = mod_init + ll[0,:]

        for t in xrange(loff+1,uoff+1):
            lalpha[t-loff] = np.logaddexp.reduce(lalpha[t-loff-1] + ltran.T, axis=1) + ll[t-loff]
        return lalpha

    def get_backward(self, metaobs, lliks, mod_tran):
        """ Creates a beta table (matrix) where
            beta_table[i,j] = beta_{i}(z_{i} = j) = P(x_{i+1:T} | z_{t} = j).
            This also scales the probabilies. Here we're looking at the
            probability of observing the partial observation sequence from time
            i+1 to T given that we're in state j at time t.

            metaobs : Optional metaobservation to specify a consecutive subset
                      of the data.

            Override this for specialized behavior.
        """

        if metaobs is None:
            loff = 0
            uoff = self.T-1
        else:
            loff, uoff = metaobs.i1, metaobs.i2

        ltran = mod_tran
        ll = lliks

        lbeta = np.empty((uoff-loff+1,self.K))
        lbeta[-1,:] = 0.

        for t in reversed(xrange(loff, uoff)):
            np.logaddexp.reduce(ltran + lbeta[t-loff+1,:] + ll[t-loff+1], axis=1,
                                out=lbeta[t-loff,:])

        return lbeta



    def forward_msgs(self, metaobs=None):
        """ Creates an alpha table (matrix) where
            alpha_table[i,j] = alpha_{i}(z_{i} = j) = P(z_{i} = j | x_{1:i}).
            This also creates the scales stored in c_table. Here we're looking
            at the probability of being in state j and time i, and having
            observed the partial observation sequence form time 1 to i.

            metaobs : Optional metaobservation to specify a consecutive subset
                      of the data.

            See: http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
                 for an explanation of forward-backward with scaling.
        """

        if metaobs is None:
            loff = 0
            uoff = self.T-1
        else:
            loff, uoff = metaobs.i1, metaobs.i2

        ltran = self.mod_tran
        ll = self.lliks

        lalpha = self.lalpha

        lalpha[0,:] = self.mod_init + ll[0,:]

        for t in xrange(loff+1,uoff+1):
            lalpha[t-loff] = np.logaddexp.reduce(lalpha[t-loff-1] + ltran.T, axis=1) + ll[t-loff]

    def forward_msgs_real_data(self, lalpha_init=None):
        ltran = self.mod_tran

        T = self.T
        K = self.K
        obs = self.obs

        lalpha = np.empty((T, K))
        ll = np.empty((T, K))

        for k, odist in enumerate(self.var_emit):
            ll[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs))

        if lalpha_init is None:
            lalpha[0,:] = self.mod_init + ll[0,:]
        else:
            lalpha[0,:] = lalpha_init

        for t in xrange(1,self.T):
            lalpha[t] = np.logaddexp.reduce(lalpha[t-1] + ltran.T, axis=1) + ll[t]

        return lalpha

    def backward_msgs(self, metaobs=None):
        """ Creates a beta table (matrix) where
            beta_table[i,j] = beta_{i}(z_{i} = j) = P(x_{i+1:T} | z_{t} = j).
            This also scales the probabilies. Here we're looking at the
            probability of observing the partial observation sequence from time
            i+1 to T given that we're in state j at time t.

            metaobs : Optional metaobservation to specify a consecutive subset
                      of the data.

            Override this for specialized behavior.
        """

        if metaobs is None:
            loff = 0
            uoff = self.T-1
        else:
            loff, uoff = metaobs.i1, metaobs.i2

        ltran = self.mod_tran
        ll = self.lliks

        lbeta = self.lbeta
        lbeta[-1,:] = 0.

        for t in reversed(xrange(loff, uoff)):
            np.logaddexp.reduce(ltran + lbeta[t-loff+1,:] + ll[t-loff+1], axis=1,
                                out=lbeta[t-loff,:])

    def intermediate_pars(self, metaobs=None):
        """ Compute natural gradient of global parameters according to the
            current meta-observation.

            metaobs : Optional metaobservation to specify a consecutive subset
                      of the data.
        """
        if metaobs is None:
            loff = 0
            uoff = self.T
        else:
            loff, uoff = metaobs.i1, metaobs.i2

        obs = self.obs
        mask = self.mask

        # Transition distributions

        # Mean-field update
        tran_mf = self.prior_tran.copy()
        for t in xrange(loff, uoff+1):
            tran_mf += np.outer(self.var_x[t-loff-1,:], self.var_x[t-loff,:])

        # Convert result to natural params -- this is the direction to follow
        A_inter = tran_mf - 1.

        # Emission distributions
        inds = np.logical_not(mask[loff:(uoff+1)])
        emit_inter = list()

        if type(self.var_emit[0]) is Gaussian:
            for k in xrange(self.K):
                G = self.var_emit[k]

                # Do mean-field update for this component
                # Slicing obs makes a copy.
                weights = self.var_x[inds,k]

                # The indexing is weird:  First we grab the subset of observations
                # we care about, and then we only grab those that aren't missing.
                #mu_mf, sigma_mf, kappa_mf, nu_mf = \
                #        util.NIW_meanfield(G, obs[loff:(uoff+1),:][inds,:], weights)
                #emit_inter.append(util.NIW_mf_natural_pars(mu_mf, sigma_mf,
                #                                           kappa_mf, nu_mf))
                
                # These are natural params already, so no need to convert
                sstats = util.NIW_suffstats(G, obs[loff:(uoff+1),:][inds,:], weights)
                emit_inter.append(sstats)

                # Convert to natural parameters
        elif type(self.var_emit[0]) is Categorical:
            for k in xrange(self.K):
                G = self.var_emit[k]

                w = self.var_x[inds,k]
                # The indexing is weird:  First we grab the subset of observations
                # we care about, and then we only grab those that aren't missing.
                data = obs[loff:(uoff+1)][inds]

                # data is actually ignored in the following function, so we
                # need to mask w to only
                C = G.num_parameters()
                dN = data.shape[0]
                z = np.zeros((dN, C))
                z[[[np.arange(data.shape[0])], [data]]] = 1
                w *= z
                alpha_mf = G._posterior_hypparams(*G._get_weighted_statistics(data,w))

                # Transform to natural parameters
                emit_inter.append(alpha_mf - 1.)

        return A_inter, emit_inter



    def intermediate_pars_buffer(self, metaobs, bufferL, L):
        """ Compute natural gradient of global parameters according to the
            current meta-observation.

            metaobs : Optional metaobservation to specify a consecutive subset
                      of the data.

            Takes in different variational distribution rather than self.var_x
        """

        if metaobs is None:
            loff = 0
            uoff = self.T
        else:
            loff, uoff = metaobs.i1+bufferL-L, metaobs.i2-bufferL+L

        var_x = self.var_x[bufferL-L:bufferL+L+1, :] 

        obs = self.obs
        mask = self.mask

        # Transition distributions

        # Mean-field update
        tran_mf = self.prior_tran.copy()
        for t in xrange(loff, uoff+1):
            tran_mf += np.outer(var_x[t-loff-1,:], var_x[t-loff,:])

        # Convert result to natural params -- this is the direction to follow
        A_inter = tran_mf - 1.

        # Emission distributions
        inds = np.logical_not(mask[loff:(uoff+1)])
        emit_inter = list()

        if type(self.var_emit[0]) is Gaussian:
            for k in xrange(self.K):
                G = self.var_emit[k]

                # Do mean-field update for this component
                # Slicing obs makes a copy.
                weights = var_x[inds,k]

                # The indexing is weird:  First we grab the subset of observations
                # we care about, and then we only grab those that aren't missing.
                #mu_mf, sigma_mf, kappa_mf, nu_mf = \
                #        util.NIW_meanfield(G, obs[loff:(uoff+1),:][inds,:], weights)
                #emit_inter.append(util.NIW_mf_natural_pars(mu_mf, sigma_mf,
                #                                           kappa_mf, nu_mf))
                
                # These are natural params already, so no need to convert
                sstats = util.NIW_suffstats(G, obs[loff:(uoff+1),:][inds,:], weights)
                emit_inter.append(sstats)

                # Convert to natural parameters
        elif type(self.var_emit[0]) is Categorical:
            for k in xrange(self.K):
                G = self.var_emit[k]

                w = var_x[inds,k]
                # The indexing is weird:  First we grab the subset of observations
                # we care about, and then we only grab those that aren't missing.
                data = obs[loff:(uoff+1)][inds]

                # data is actually ignored in the following function, so we
                # need to mask w to only
                C = G.num_parameters()
                dN = data.shape[0]
                z = np.zeros((dN, C))
                z[[[np.arange(data.shape[0])], [data]]] = 1
                w *= z
                alpha_mf = G._posterior_hypparams(*G._get_weighted_statistics(data,w))

                # Transform to natural parameters
                emit_inter.append(alpha_mf - 1.)

        return A_inter, emit_inter

    def global_update(self, A_inter, emit_inter):
        """ Perform global updates based on batch following the stochastic
            natural gradient.

            A_inter : Intermediate parameter for transition matrix.
            emit_inter : Intermediate parameters for emission distribitions.

            The intermediate parameters are the ascent directions.
        """

        lrate = self.lrate
        L = self.metaobs_half
        S = self.mb_sz
        T = self.T

        # Perform stochastic gradient update on global params.

        # Transition distributions
        # Convert current estimate to natural params
        nats_old = self.var_tran - 1.

        # Perform update according to stochastic gradient
        # (Hoffman, pg. 17)
        bfact = (T-2*L-1) / (2.*L*S)
        A_up = bfact * A_inter
        #scale learning rate by square root of all historical gradients
        if self.adagrad:
            self.ada_G += nats_old**2
            #what if we try normalizing the matrix
            adaMatrix = self.ada_G**.25
            nats_new = (1. - 1.0/adaMatrix)*nats_old + A_up/adaMatrix   #right now we are not using lrate with ada
        else:
            nats_new = (1.-lrate)*nats_old + lrate*A_up

        # Convert results back to moment params
        self.var_tran = nats_new + 1.

        # Emission distributions
        bfact = (T-2*L-1) / ((2.*L+1.)*S)

        if type(self.var_emit[0]) is Gaussian:
            for k in xrange(self.K):
                G = self.var_emit[k]

                # Convert current estimates to natural parameters
                nats_old = util.NIW_mf_natural_pars(G.mu_mf, G.sigma_mf,
                                                    G.kappa_mf, G.nu_mf)

                # Perform update according to stochastic gradient
                # (Hoffman, pg. 17)

                # Get hyperparms in natural parameter form
                prior_hypparam = util.NIW_mf_natural_pars(G.mu_0, G.sigma_0,
                                                          G.kappa_0, G.nu_0)

                nats_new = (1.-lrate)*nats_old \
                           + lrate*(prior_hypparam + bfact*emit_inter[k])

                # Convert new params into moment form and store back in G
                util.NIW_mf_moment_pars(G, *nats_new)

        elif type(self.var_emit[0]) is Categorical:
            for k in xrange(self.K):

                G = self.var_emit[k]

                # Convert current estimates to natural parameters
                nats_old = G.alpha_mf - 1.

                # Perform update according to stochastic gradient
                nats_new = (1.-lrate)*nats_old + lrate*bfact*emit_inter[k]

                # Convert new natural params back into mean form and store in G
                G._alpha_mf = nats_new + 1.
                G.weights = G._alpha_mf / G._alpha_mf.sum()

    def pred_logprob(self, metaobs=None):
        """ Compute predictive log-probability of missing data in specified
            meta-observation.

            BEWARE: If metaobs is not the same as self.cur_mo then a local step
                    is run on metaobs and the internal local state of the object
                    (alpha_table, beta_table, var_x, etc.) will change.
            
            metaobs: Optional MetaObs namedtuple indicating which
                     meta-observation to copute for.  If not specified the
                     current meta-observation is used.
        """
        cur_mo = self.cur_mo
        if metaobs is None:
            metaobs = cur_mo

        if ((metaobs is not cur_mo) and
              (metaobs.i1 != cur_mo.i1 and metaobs.i2 != cur_mo.i2)):
            self.local_update(metaobs=metaobs)

        K = self.K
        loff, uoff = metaobs.i1, metaobs.i2
        obs = self.obs_full[loff:(uoff+1),:]
        mask = self.mask[loff:(uoff+1)]
        nmiss = np.sum(mask)
        if nmiss == 0:
            return None

        logprob = np.zeros((nmiss,K))

        for k, odist in enumerate(self.var_emit):
            logprob[:,k] = np.log(self.var_x[mask,k]+eps) + odist.expected_log_likelihood(obs[mask,:])

        return np.mean(np.logaddexp.reduce(logprob, axis=1))

    def pred_logprob_full(self):
        """ Compute predictive log-probability of all missing data.
            Requires a full forwards-backwards pass.

            We may want to do this at some point.
        """
        # Have to implement this separaetly because the sizes of alpha_table,
        # ect. aren't this big, so it would mess things up to use the other
        # function.

        full_var_x = self.full_local_update()

        K = self.K
        obs = self.obs_full
        mask = self.mask
        nmiss = np.sum(mask)
        if nmiss == 0:
            return None

        logprob = np.zeros((nmiss,K))

        for k, odist in enumerate(self.var_emit):
            logprob[:,k] = np.log(full_var_x[mask,k]+eps) + odist.expected_log_likelihood(obs[mask,:])

        return np.mean(np.logaddexp.reduce(logprob, axis=1))

    def full_local_update(self):
        """ Local update on full data set.  Reimplements member functions
            because we don't want to use the object's internal variables.

            This is only useful if we can store the whole state sequence in
            memory.
        """

        # update the modified parameter tables (don't do emissions b/c
        # pybasicbayes takes care of those).
        mod_init = digamma(self.var_init + eps) - digamma(np.sum(self.var_init) + eps)
        tran_sum = np.sum(self.var_tran, axis=1)
        mod_tran = digamma(self.var_tran + eps) - digamma(tran_sum[:,npa] + eps)

        T = self.T
        K = self.K
        obs = self.obs
        mask = self.mask

        # Mask out missing data (restored below)
        obs_full = obs.copy()
        obs[mask,:] = np.nan

        lalpha = np.empty((T, K))
        lbeta = np.empty((T, K))

        ll = np.empty((T, K))
        # Compute likelihoods
        for k, odist in enumerate(self.var_emit):
            ll[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs))

        # Forward messages
        ltran = mod_tran

        lalpha[0,:] = mod_init + ll[0,:]

        for t in xrange(1,self.T):
            lalpha[t] = np.logaddexp.reduce(lalpha[t-1] + ltran.T, axis=1) + ll[t]

        # Backward messages
        ltran = mod_tran

        lbeta[self.T-1,:] = 0.

        for t in xrange(self.T-2,-1,-1):
            np.logaddexp.reduce(ltran + lbeta[t+1] + ll[t+1], axis=1,
                                out=lbeta[t])


        # Update weights
        var_x = lalpha + lbeta
        var_x -= np.max(var_x, axis=1)[:,npa]
        var_x = np.exp(var_x)
        var_x /= np.sum(var_x, axis=1)[:,npa]

        # Restore full observations
        self.obs = obs_full

        return var_x
