from __future__ import division

import numpy as np

from util import make_mask, make_mask_prediction


def generate_data(tran, emit, T, miss=0., nmasks=1):
    """ Generate synthetic data from finite HMM with continuous emissions.
        A `miss` fraction of data is marked as missing.  `nmasks` sets of
        missing data are created.

        tran : K x K discrete transition matrix.

        emit : K x 1 vector of continuous emission distributions.

        T : Number of observations to generate.

        miss : Fraction of observations that are missing.  Tries to pick evenly
               over all states.

        nmasks : Number of missing masks to generate.
    """

    # doesn't matter for now because we're only dealing with one sequence of
    # observations
    K = tran.shape[0]
    curr_st = 0
    obs = []

    sts = [0]
    obs.append(emit[0].rvs()[0])
    for i in xrange(T-1):
        # transition
        # pass in probability of 1
        curr_st = np.random.choice(K, p=tran[curr_st,:])
        sts.append(curr_st)

        # emit an observation
        point = emit[curr_st].rvs()[0]
        obs.append(point)

    obs = np.array(obs)
    sts = np.array(sts)
    masks = None

    if miss > 0.:
        masks = list()
        for i in xrange(nmasks):
            masks.append(make_mask(sts, miss))

        # Backwards compatability with old tests
        if len(masks) == 1:
            masks = masks[0]

    return obs, sts, masks


def generate_data_smoothing(tran, emit, T, miss=0., left=0, nmasks=1):
    """ Generate synthetic data from finite HMM with continuous emissions.
        A `miss` fraction of data to the right of index `left` is marked as
        missing.  `nmasks` sets of missing data are created.

        tran : K x K transition matrix (row normalized).

        emit : 1-d array-like of size K containing continuous emissions.

        T : Length of resulting observation sequence.

        miss : Fraction of observations that are missing.

        left : Index of leftmost observation that can be missing.

        nmasks : Number of missing masks to generate.
    """
    # doesn't matter for now because we're only dealing with one sequence of
    # observations
    K = tran.shape[0]
    curr_st = 0
    obs = []

    sts = [0]
    obs.append(emit[0].rvs()[0])
    for i in xrange(T-1):
        # transition
        # pass in probability of 1
        curr_st = np.random.choice(K, p=tran[curr_st,:])
        sts.append(curr_st)

        # emit an observation
        point = emit[curr_st].rvs()[0]
        obs.append(point)

    obs = np.array(obs)
    sts = np.array(sts)
    masks = None

    if miss > 0.:
        masks = list()
        for i in xrange(nmasks):
            masks.append(make_mask(sts, miss, left))

    # Backwards compatability with old tests
    if len(masks) == 1:
        masks = masks[0]

    return obs, sts, masks


def generate_data_prediction(tran, emit, T, miss=0., nmasks=1):
    """ Generate synthetic data from finite HMM with continuous emissions.
        A `miss` fraction of data at the end of the observation sequence is
        marked as missing.  `nmasks` sets of missing data are created.

        tran : K x K transition matrix (row normalized).

        emit : 1-d array-like of size K containing continuous emissions.

        T : Length of resulting observation sequence.

        miss : Fraction of observations that are missing.

        nmasks : Number of missing masks to generate.
    """
    # doesn't matter for now because we're only dealing with one sequence of
    # observations
    K = tran.shape[0]
    curr_st = 0
    obs = []

    sts = [0]
    obs.append(emit[0].rvs()[0])
    for i in xrange(T-1):
        # transition
        # pass in probability of 1
        curr_st = np.random.choice(K, p=tran[curr_st,:])
        sts.append(curr_st)

        # emit an observation
        point = emit[curr_st].rvs()[0]
        obs.append(point)

    obs = np.array(obs)
    sts = np.array(sts)
    masks = None

    if miss > 0:
        masks = list()
        for i in xrange(nmasks):
            masks.append(make_mask_prediction(sts, miss))

    if len(masks) == 1:
        masks = masks[0]

    return obs, sts, masks


def generate_data_mmap(tran, emit, T):
    """ This writes large sequences of observations to disk using mmap

        Uses numpy's memmap here isntead of mmap which is better for
        storing numpy arrays
    """

    fpo = np.memmap('obs.dat', dtype='float64', mode='w+', shape=(T, tran.shape[0]))
    fps = np.memmap('sts.dat', dtype='int32', mode='w+', shape=(T, 1))

    states = np.arange(tran.shape[0])
    curr_st = 0

    fps[0,:] = 0
    fpo[0,:] = emit[0].rvs()[0]
    for i in xrange(T - 1):
        # transition
        # pass in probability of 1
        curr_st = np.random.choice(states, p=tran[curr_st,:])
        fps[i,:] = curr_st

        # emit an observation
        fpo[i,:] = emit[curr_st].rvs()[0]

    # Delete the file pointers to flush the changes to memory. Here we're only
    # using the filenames obs.dat and sts.dat
    del fps
    del fpo


def read_data_mmap(N, T, size):
    fp = np.memmap('obs.dat', dtype='float64', mode='r', shape=(T, N))
    for i in xrange(T//size):
        yield np.array(fp[i*size:(i+1)*size,:])
