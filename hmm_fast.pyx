
# Optimized functions for topicstream module

import numpy as np
cimport numpy as np

import cython
cimport cython

from scipy.special import digamma
from numpy import newaxis as npa

from libc.stdlib cimport rand, RAND_MAX


cdef extern from "float.h":
    double DBL_EPSILON
    double DBL_MAX


cdef extern from "math.h":
    double exp(double x)
    double log(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int rand_discrete(int K, double* p):
    cdef double r = rand() / <double>RAND_MAX
    cdef double rsum = 0.
    cdef int i
    for i in xrange(K):
        rsum += p[i]
        if r <= rsum:
            return i


#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def FFBS(self, np.ndarray[np.double_t, ndim=1] var_init,
               np.ndarray[np.double_t, ndim=2] lalpha_init=None):
    """ Forward Filter Backward Sampling to simulate state sequence.

        var_init : The initial variational distribution.

        lalpha_init : The forward messages. Pass this if you don't want FFBS to
            calculate forward messages and just do backward sampling.
    """
    cdef np.ndarray[np.double_t, ndim=2] obs
    cdef np.ndarray[np.double_t, ndim=2] A
    cdef np.ndarray[np.double_t, ndim=1] mod_init
    cdef np.ndarray[np.double_t, ndim=1] tran_sum
    cdef np.ndarray[np.double_t, ndim=2] mod_tran
    cdef np.ndarray[np.double_t, ndim=2] lalpha
    cdef np.ndarray[np.double_t, ndim=2] lliks
    cdef np.ndarray[np.int_t, ndim=1] z
    cdef np.ndarray[np.double_t, ndim=1] lp
    cdef np.ndarray[np.double_t, ndim=1] p
    cdef int T, K, t, k

    obs = self.obs
    T = self.T
    K = self.K
    A = self.var_tran

    # Allocate
    z = np.empty(T, dtype=np.int_)
    lp = np.empty(K)
    p = np.empty(K)

    mod_init = digamma(var_init + DBL_EPSILON) \
               - digamma(np.sum(var_init) + DBL_EPSILON)
    tran_sum = np.sum(A, axis=1)
    mod_tran = digamma(A + DBL_EPSILON) \
               - digamma(tran_sum[:,npa] + DBL_EPSILON)

    if lalpha_init == None:
        lalpha = np.empty((T, K))
        lliks = np.empty((T, K))
        # Compute likelihoods
        for k, odist in enumerate(self.var_emit):
            lliks[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs))

        for k in xrange(K):
            lalpha[0,k] = mod_init[k] + lliks[0,k]

        for t in xrange(1,T):
            lalpha[t] = np.logaddexp.reduce(lalpha[t-1]
                                            + np.log(A+DBL_EPSILON).T, axis=1) \
                                            + lliks[t]
    else:
        lalpha = lalpha_init

    cdef double lp_max = -DBL_MAX
    cdef double psum = 0.
    for k in xrange(K):
        if lalpha[T-1,k] > lp_max:
            lp_max = lalpha[T-1,k]
    for k in xrange(K):
        p[k] = exp(lalpha[T-1,k] - lp_max)
        psum += p[k]
    for k in xrange(K):
        p[k] /= psum
    z[T-1] = rand_discrete(K, <double*>p.data)

    for t in xrange(T-2, -1, -1):
        lp_max = -DBL_MAX
        psum = 0.
        for k in xrange(K):
            lp[k] = lalpha[t,k] + log(A[k,z[t+1]] + DBL_EPSILON)
            if lp[k] > lp_max:
                lp_max = lp[k]
        for k in xrange(K):
            p[k] = exp(lp[k] - lp_max)
            psum += p[k]
        for k in xrange(K):
            p[k] /= psum

        z[t] = rand_discrete(K, <double*>p.data)

    return z, lalpha
