
import numpy as np
cimport numpy as np

import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def FDRs(np.ndarray[np.int_t, ndim=1] sts_pred,
         np.ndarray[np.int8_t, ndim=1] chrom_tss,
         int K):

    cdef np.ndarray[np.double_t, ndim=1] false_pos
    cdef np.ndarray[np.double_t, ndim=1] preds
    #cdef np.ndarray[np.double_t, ndim=1] tmp
    cdef bint tss
    cdef int i
    cdef int T = chrom_tss.shape[0]

    false_pos = np.zeros(K)
    preds = np.zeros(K)

    for i in xrange(T):
        tss = chrom_tss[i]
        preds[sts_pred[i]] += 1
        if not tss:
            false_pos[sts_pred[i]] += 1

    #tmp = np.zeros(K)
    #for i in xrange(K):
    #    if preds[i] > 0:
    #        tmp[i] = false_pos[i] / preds[i]
    #    else:
    #        tmp[i] = -1

    return false_pos, preds
    #return np.where(np.logical_or(np.isinf(tmp), np.isnan(tmp)), -1., tmp)
