
from __future__ import division
from munkres import Munkres, print_matrix

import math
import itertools
import numpy as np
import numpy.linalg as npl
import scipy.spatial.distance as distance


def NIW_zero_nat_pars(G):
    p = len(G.mu_mf)
    return np.array([np.zeros(p), 0., np.zeros((p,p)), 0])

def NIW_nat2moment_pars(e1, e2, e3, e4):
    p = len(e1)
    mu = e1 / e2
    kappa = e2
    # This may be wrong, but  bnpy/distr/GaussWishDistr.py uses it and it's
    # actually stable.
    sigma = e3 - np.outer(mu, mu) / kappa
    #sigma = e3 - np.outer(mu, mu) * kappa
    nu = e4 - 2 - p
    return np.array([mu, sigma, kappa, nu])


def NIW_mf_natural_pars(mu, sigma, kappa, nu):
    """ Convert moment parameters of Normal inverse-Wishart distribution to
        natural parameters.
    """
    p = len(mu)
    # This may be wrong, but  bnpy/distr/GaussWishDistr.py uses it and it's
    # actually stable.
    #eta3 = sigma + np.outer(mu, mu) / kappa
    eta3 = sigma + np.outer(mu, mu) * kappa
    return np.array([kappa * mu, kappa, eta3, nu + 2 + p])


def NIW_mf_moment_pars(G, e1, e2, e3, e4):
    """ Convert natural parameters of Normal inverse-Wishart given by e1,...,e4
        to moment parameterization and update the values in G.
    """
    p = len(e1)
    mu = e1 / e2
    kappa = e2
    # This may be wrong, but  bnpy/distr/GaussWishDistr.py uses it and it's
    # actually stable.
    #sigma = e3 - np.outer(mu, mu) / kappa
    sigma = e3 - np.outer(mu, mu) * kappa
    nu = e4 - 2 - p

    G.mu_mf = mu
    G.sigma_mf = sigma
    G.kappa_mf = kappa
    G.nu_mf = nu

    # Just to match the meanfieldupdate function
    G.mu = G.mu_mf
    G.sigma = G.sigma_mf/(G.nu_mf - p - 1)  # for plotting


def NIW_meanfield(G, data, weights):
    """ Compute mean-field updates for Normal inverse-Wishart object G and
        return them.
    """
    D = len(G.mu_0)
    mu_mf, sigma_mf, kappa_mf, nu_mf = \
        G._posterior_hypparams(*G._get_weighted_statistics(data,weights,D))
    return np.array([mu_mf, sigma_mf, kappa_mf, nu_mf])


def NIW_suffstats(G, data, weights):
    """ Compute sufficient statistics for NIW object G and return them in the
        form of the natural parameters used in NIW_mf_natural_pars."
    """
    D = len(G.mu_0)
    tmp = weights[:,np.newaxis]*data
    S = data.T.dot(tmp)
    xbar = np.sum(tmp, axis=0)
    neff = weights.sum()

    return np.array([xbar, neff, S, neff])


def KL_gaussian(mu0, sig0, mu1, sig1):
    """ KL(N_0 || N_1).
    """
    D = len(mu0)
    if D != len(mu1) or D != sig0.shape[0] or D != sig1.shape[0]:
        raise RuntimeError("Means and covariances my be the same dimension.")
    if sig0.shape[0] != sig0.shape[1] or sig1.shape[0] != sig1.shape[1]:
        raise RuntimeError("Covariance matrices must be square.")

    s1inv = npl.inv(sig1)
    s0_ld = npl.slogdet(sig0)[1]
    s1_ld = npl.slogdet(sig1)[1]
    x = mu1 - mu0
    tmp = np.trace(np.dot(s1inv, sig0)) + np.dot(x.T, np.dot(s1inv, x))
    tmp += -D - s0_ld + s1_ld

    return 0.5 * tmp


def dirichlet_natural_pars(alpha):
    return alpha - 1.


def dirichlet_moment_pars(eta):
    return eta + 1.


def plot_ellipse(pos, P, edge='k', face='none'):
    # Written this way so that util compiles on the cluster
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        U, s, Vh = npl.svd(P)
        orient = math.atan2(U[1,0],U[0,0])*180/np.pi
        ellipsePlot = Ellipse(xy=pos, width=2.0*math.sqrt(s[0]),
                              height=2.0*math.sqrt(s[1]), angle=orient,
                              facecolor=face, edgecolor=edge)
        ax = plt.gca()
        ax.add_patch(ellipsePlot)
        return ellipsePlot

    except ImportError:
        pass


def plot_emissions(obs, prior_emit, var_emit):
    # Written this way so that util compiles on the cluster
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(obs[:,0], obs[:,1])
        for G in prior_emit:
            plt.scatter(*G.mu_mf, color='green')
        for G in var_emit:
            plt.scatter(*G.mu_mf, color='red')
            plot_ellipse(G.mu_mf, G.sigma, edge='r', face='none')

    except ImportError:
        pass


def mvnrand(mean, cov, size=1):
    """ Simulate samples from multivariate normal.

        mean : 1-D array-like of length D.
        cov : 2-D array-like of shape (D,D).
        size : int Number of samples to generate, result will be an N x D
                   array.
    """
    mu = np.squeeze(mean)
    D = mu.shape[0]
    C = npl.cholesky(cov)
    z = np.random.randn(size, D)

    return np.squeeze(mu + np.dot(z, C.T))


def make_mask(sts, miss=0., left=0):
    """ Indicate `miss` fraction of observations after index `left` as missing.
    """

    sts_l = sts[left:]
    K = np.unique(sts_l).shape[0]
    mask = np.zeros(len(sts), dtype='bool')
    if miss > 0.:
        for k in xrange(K):
            obs_k = np.where(sts_l == k)[0]

            # Don't hold out data for this state if too few observations
            if obs_k.shape[0] < 10:
                continue

            # Number held out should be based on length of whole series
            nobs_k = np.ceil(miss*np.sum(sts == k))
            if obs_k.shape[0] < nobs_k:
                print "Only %d observations in state %d, using fraction of that instead" % (obs_k.shape[0], k)
                nobs_k = np.ceil(miss*obs_k.shape[0])
            nobs_k = int(nobs_k)

            inds = np.random.choice(obs_k, size=nobs_k, replace=False)

            # Have to add left back into inds b/c we chopped off the left of
            # the sequence
            mask[left+inds] = True

    return mask


def make_mask_prediction(sts, miss=0.):
    """ Indicate last `miss` frac of observation sequence as missing.
    """

    nobs = len(sts)
    mask = np.zeros(nobs, dtype='bool')
    if miss == 0.:
        return mask

    nmiss = np.ceil(miss*nobs)
    mask[-nmiss:] = True

    return mask


def match_state_seq(sts_true, sts_pred, K):
    """ Matchs the set of states in sts_pred such that it minimizes the hamming
        distance between sts_pred and sts_true. We assume here that the states
        are labeled 0, ..., K - 1.

        sts_true : A numpy array of integers.

        sts_pred : A numpy array of integers.

        K : Number of states in case sts_true doesn't cover all states.
    """

    sts = np.arange(K, dtype='int')
    sts_true = sts_true.astype('int')
    sts_pred = sts_pred.astype('int')
    min_perm = None
    min_hd = np.inf
    for p in itertools.permutations(sts):
        cur_sts = np.array(p)[sts_pred]
        hd = distance.hamming(sts_true, cur_sts)
        if hd < min_hd:
            min_hd = hd
            min_perm = p

    return np.array(min_perm)


def munkres_match(sts_true, sts_pred, K):
    """ Matchs the set of states in sts_pred such that it minimizes the hamming
        distance between sts_pred and sts_true. We assume here that the states
        are labeled 0, ..., K - 1. This uses the Munkres algorithm to minimize
        the hamming distance which is must faster than match_state_seq.

        sts_true : A numpy array of integers.

        sts_pred : A numpy array of integers.

        K : Number of states in case sts_true doesn't cover all states.
    """

    sts_true = sts_true.astype('int')
    sts_pred = sts_pred.astype('int')

    cost_mat = np.zeros((K, K))

    #unq_true = np.unique(sts_true)
    #unq_pred = np.unique(sts_pred)
    #K_true = unq_true.shape[0]
    #K_pred = unq_pred.shape[0]

    #DM = np.zeros((K_pred, K_true))
    #for ei in xrange(K_pred):
    #    iei = np.where(sts_pred == unq_pred[ei])[0]
    #    for ti in xrange(K_true):
    #        n_incorr = np.sum(sts_true[iei] == unq_true[ti])
    #        DM[ei,ti] = n_incorr

    DM = np.zeros((K, K))
    for k in xrange(K):
        iei = np.where(sts_pred == k)[0]
        for l in xrange(K):
            n_incorr = np.sum(sts_true[iei] == l)
            DM[k,l] = n_incorr

    cost_mat = 1 - (DM / np.sum(DM))

    m = Munkres()
    indexes = m.compute(cost_mat)
    return np.array([x[1] for x in indexes])


if __name__ == "__main__":

    K = 6
    sts_true = np.random.choice(K, size=1000)
    rand_perm = np.random.permutation(K)
    sts_perm = rand_perm[sts_true]

    match_perm = match_state_seq(sts_true, sts_perm, K)
    match_munk = munkres_match(sts_true, sts_perm, K)

    t1 = np.all(sts_true == match_perm[sts_perm])
    print "match_state_seq works? %s" % ("True" if t1 else "False",)

    t2 = np.all(sts_true == match_munk[sts_perm])
    print "munkres_match works? %s" % ("True" if t2 else "False",)
