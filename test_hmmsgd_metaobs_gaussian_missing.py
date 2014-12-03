from __future__ import division

import numpy as np
import numpy.linalg as npl
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist


from copy import deepcopy
from itertools import groupby
from sklearn.cluster import KMeans

try:
    from numpy.random import multivariate_normal as mvnrand
except IOError:
    from util import mvnrand

import hmmsgd_metaobs as HMM
import gen_synthetic
import util
#import graphing
from pybasicbayes.distributions import Gaussian, Multinomial

from util import make_mask  # Just needed for test3


# These are the emission distributions for the following tests
K = 4
D = 2
mu_true = np.array([[5, 5], [5, -5], [-5, 5], [-5, -5]], dtype='float64')
sigma_true = [np.eye(2) for i in xrange(K)]

emit = [Gaussian(mu=mu_true[i,:], sigma=sigma_true[i],
                 mu_0=mu_true[i,:], sigma_0=sigma_true[i],
                 kappa_0=1, nu_0=4) for i in xrange(K)]

def initialize_prior_emit(obs, KK, D, mu_true, mu_0, sigma_true, sigma_0, 
                          kappa_0, nu_0, initialize='k-means'):
    N = len(obs)
    # Initialize around ground truth to test
    if initialize == 'true':
        prior_emit = np.array([Gaussian(mu=mu_true[i,:], sigma=sigma_true[i],
                                        mu_0=mu_0, sigma_0=sigma_0,
                                        kappa_0=kappa_0, nu_0=nu_0)
                                        for i in xrange(KK)])

    # Initialize means with observations
    elif initialize == 'obs':
        init_means = np.empty((KK,D))
        inds = np.arange(N)
        for k in xrange(KK):
            ii = np.random.choice(inds)
            np.delete(inds, ii)
            init_means[k,:] = obs[ii,:]
        prior_emit = np.array([Gaussian(mu=init_means[i,:], sigma=sigma_0,
                                        mu_0=mu_0, sigma_0=sigma_0,
                                        kappa_0=kappa_0, nu_0=nu_0)
                               for i in xrange(KK)])

    # Initialize from prior
    elif initialize == 'prior':
        init_means = np.empty((KK,D))
        # Is this necessary? Won't Gaussian draw a mu and sigma from mu_0 and
        # sigma_0?
        for k in xrange(KK):
            init_means[k,:] = mvnrand(mu_0, sigma_0)
        prior_emit = np.array([Gaussian(mu=init_means[i,:], sigma=sigma_0,
                                        mu_0=mu_0, sigma_0=sigma_0,
                                        kappa_0=kappa_0, nu_0=nu_0)
                               for i in xrange(KK)])

    # Initialize with k-means
    elif initialize == 'k-means':
        km = KMeans(n_clusters=KK, max_iter=500, n_init=25)
        km.fit(obs)
        init_means = km.cluster_centers_
        prior_emit = np.array([Gaussian(mu=init_means[i,:], sigma=sigma_0,
                                        mu_0=mu_0, sigma_0=sigma_0,
                                        kappa_0=kappa_0, nu_0=nu_0)
                               for i in xrange(KK)])
    else:
        raise Exception('initalize must be \'true\', \'obs\', \'prior\' or \'k-means\'')

    return prior_emit

# These are used for the following tests
def test_base(tran, emit):
    N = 1001
    missfrac = 0.2

    obs, sts, mask = gen_synthetic.generate_data(tran, emit, N, miss=missfrac)

    obs_mean = np.mean(obs, axis=0)

    mu_0 = obs_mean
    sigma_0 = 0.75*np.cov(obs.T)  # 0.75*np.diag(obs_var)
    kappa_0 = 0.01
    nu_0 = 4

    # Initialize around ground truth to test
    #KK = K
    #prior_emit = np.array([Gaussian(mu=mu_true[i,:], sigma=sigma_true[i],
    #                                mu_0=mu_0, sigma_0=sigma_0,
    #                                kappa_0=kappa_0, nu_0=nu_0)
    #                                for i in xrange(len(emit))])

    # Initialize means with observations
    #KK = K
    #init_means = np.empty((KK,D))
    #inds = np.arange(N)
    #for k in xrange(KK):
    #    ii = np.random.choice(inds)
    #    np.delete(inds, ii)
    #    init_means[k,:] = obs[ii,:]
    #prior_emit = np.array([Gaussian(mu=init_means[i,:], sigma=sigma_0,
    #                                mu_0=mu_0, sigma_0=sigma_0,
    #                                kappa_0=kappa_0, nu_0=nu_0)
    #                       for i in xrange(KK)])

    # Initialize from prior
    KK = K
    init_means = np.empty((KK,D))
    # Is this necessary? Won't Gaussian draw a mu and sigma from mu_0 and
    # sigma_0?
    for k in xrange(KK):
        init_means[k,:] = mvnrand(mu_0, sigma_0, 1)
    prior_emit = np.array([Gaussian(mu=init_means[i,:], sigma=sigma_0,
                                    mu_0=mu_0, sigma_0=sigma_0,
                                    kappa_0=kappa_0, nu_0=nu_0)
                           for i in xrange(KK)])

    prior_tran = np.ones(KK**2).reshape(KK, KK)
    prior_init = np.ones(KK)

    init_init = np.random.rand(KK)
    init_init /= np.sum(init_init)
    init_tran = np.random.rand(KK,KK)
    init_tran /= np.sum(init_tran, axis=1)[:,np.newaxis]

    print "Prior emissions:"
    for i in xrange(KK):
        print prior_emit[i]

    maxit = 25
    tau = 1
    kappa = 0.75
    metaobs_half = 1
    mb_sz = 5

    full_predprob = False

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, tau=tau,
                    kappa=kappa, metaobs_half=metaobs_half, mb_sz=mb_sz,
                    mask=mask, full_predprob=full_predprob,
                    init_init=init_init, init_tran=init_tran,
                    verbose=True, metaobs_fun='unif',
                    maxit=maxit)
    hmm.infer()

    for i in xrange(KK):
        print "=============EMIT ", i, " FIELDS"
        print hmm.var_emit[i]

    # Do a full local step to estimate state sequence b/c it's synthetic data
    full_var_x = None
    full_var_x = hmm.full_local_update()

    print "True state sequence:"
    print sts
    print "Learned state sequence:"
    state_sq = np.argmax(full_var_x, axis=1).astype('float64')
    print state_sq

    #plot_state_sq(state_sq)

    # Make plot
    plt.figure()
    plt.scatter(obs[:,0], obs[:,1])
    for G in prior_emit:
        plt.scatter(*G.mu_mf, color='green')
    for G in hmm.var_emit:
        plt.scatter(*G.mu_mf, color='red')
        util.plot_ellipse(G.mu_mf, G.sigma, edge='r', face='none')
    plt.show()

    priors = {'emit': prior_emit, 'tran': prior_tran, 'init': prior_init}
    return hmm, full_var_x, obs, sts, priors


def bridge_10state(mu_true, A_true, initialize='k-means'):
    """
    """

    K = A_true.shape[0]
    D = 2

    sigma_true = [np.eye(2) for i in xrange(K)]
    emit_true = [Gaussian(mu=mu_true[i,:], sigma=sigma_true[i],
                          mu_0=mu_true[i,:], sigma_0=sigma_true[i],
                          kappa_0=1, nu_0=4) for i in xrange(K)]

    T = 1001
    miss = 0.2
    nfolds = 1
    left = T // 4  # No missing data in first quarter of data

    obs, sts, mask = gen_synthetic.generate_data_smoothing(A_true, emit_true,
                                                           T, miss, left,
                                                           nfolds)

    obs_mean = np.mean(obs, axis=0)

    mu_0 = obs_mean
    sigma_0 = .75*np.cov(obs.T)  # 0.75*np.diag(obs_var)
    kappa_0 = .001
    nu_0 = 4
    KK = K

    prior_emit = initialize_prior_emit(obs, KK, D, mu_true, mu_0, sigma_true, sigma_0,
                                       kappa_0, nu_0, initialize)

    prior_tran = np.ones(KK**2).reshape(KK, KK)
    prior_init = np.ones(KK)

    init_init = np.random.rand(KK)
    init_init /= np.sum(init_init)
    init_tran = np.random.rand(KK,KK)
    init_tran /= np.sum(init_tran, axis=1)[:,np.newaxis]

    print "Prior emissions:"
    for i in xrange(KK):
        print prior_emit[i]

    tau = 1
    kappa = 0.4
    metaobs_half = 1
    mb_sz = 130

    full_predprob = False

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, tau, kappa,
                    metaobs_half, mb_sz, mask, full_predprob,
                    init_init=init_init, init_tran=init_tran, adagrad = False,
                    verbose=True, metaobs_fun='unif')
    hmm.infer()

    for i in xrange(KK):
        print "=============EMIT ", i, " FIELDS"
        print hmm.var_emit[i]

    # Do a full local step to estimate state sequence b/c it's synthetic data
    full_var_x = hmm.full_local_update()

    print "True state sequence:"
    print sts
    print "Learned state sequence:"
    state_sq = np.argmax(full_var_x, axis=1).astype('float64')
    print state_sq

    #plot_state_sq(state_sq)

    # Make plot
    plt.figure()
    plt.scatter(obs[:,0], obs[:,1])
    for G in prior_emit:
        plt.scatter(*G.mu_mf, color='green')
    for G in hmm.var_emit:
        plt.scatter(*G.mu_mf, color='red')
        util.plot_ellipse(G.mu_mf, G.sigma, edge='r', face='none')
    plt.show()


    ##compare to adagrad: shown in second plot
    #hmm2 = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, tau, kappa,
    #                metaobs_half, mb_sz, mask, full_predprob,
    #                init_init=init_init, init_tran=init_tran, adagrad = True,
    #                verbose=True, metaobs_fun='noverlap')
    #hmm2.infer()

    #plt.figure()
    #plt.scatter(obs[:,0], obs[:,1])
    #for G in prior_emit:
    #    plt.scatter(*G.mu_mf, color='green')
    #for G in hmm2.var_emit:
    #    plt.scatter(*G.mu_mf, color='red')
    #    util.plot_ellipse(G.mu_mf, G.sigma, edge='r', face='none')
    #plt.show()

    #graph true matrix
    #generate_trans_graph(A_true, True10state.png)
    #generate_trans_graph(hmm.var_tran, Var10state.png)

    priors = {'emit': prior_emit, 'tran': prior_tran, 'init': prior_init}
    return hmm, full_var_x, obs, sts, priors


def reversed_cycles(mu_true, A_true, initialize='prior'):
    """
    """

    K = A_true.shape[0]
    D = 2

    sigma_true = [12*np.eye(2) for i in xrange(K)]
    emit_true = [Gaussian(mu=mu_true[i,:], sigma=sigma_true[i],
                          mu_0=mu_true[i,:], sigma_0=sigma_true[i],
                          kappa_0=1, nu_0=4) for i in xrange(K)]

    T = 10001
    miss = 0.001
    nfolds = 1
    left = T // 50  # No missing data in first quarter of data

    obs, sts, mask = gen_synthetic.generate_data_smoothing(A_true, emit_true,
                                                           T, miss, left,
                                                           nfolds)

    obs_mean = np.mean(obs, axis=0)

    mu_0 = obs_mean
    sigma_0 = .75*np.cov(obs.T)  # 0.75*np.diag(obs_var)
    kappa_0 = .001
    nu_0 = 4
    KK = K

    prior_emit = initialize_prior_emit(obs, KK, D, mu_true, mu_0, sigma_true, sigma_0,
                                       kappa_0, nu_0, initialize)

    prior_tran = np.ones(KK**2).reshape(KK, KK)
    prior_init = np.ones(KK)

    init_init = np.random.rand(KK)
    init_init /= np.sum(init_init)
    init_tran = np.random.rand(KK,KK)
    init_tran /= np.sum(init_tran, axis=1)[:,np.newaxis]

    # print "Prior emissions:"
    # for i in xrange(KK):
    #     print prior_emit[i]

    tau = 1
    kappa = 0.3
    metaobs_half = 100
    mb_sz = 1

    full_predprob = False

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, tau, kappa,
                    metaobs_half, mb_sz, mask, full_predprob, seed = 40,
                    init_init=init_init, init_tran=init_tran, adagrad = False,
                    verbose=False, metaobs_fun='unif', growBuffer=False,maxit=100)
    hmm.infer()

    # for i in xrange(KK):
    #     print "=============EMIT ", i, " FIELDS"
    #     print hmm.var_emit[i]

    # Do a full local step to estimate state sequence b/c it's synthetic data
    full_var_x = hmm.full_local_update()

    print "True state sequence:"
    print sts
    print "Learned state sequence:"
    #state_sq = np.argmax(full_var_x, axis=1).astype('float64')
    #state_sq = np.argmax(full_var_x, axis=1).astype(int) #changed this to integer
    #print state_sq

    hamm, perm = hmm.hamming_dist(full_var_x, sts)
    print perm

    print "Hamming Distance: %.3f" % hamm

    KL, L_2 = hmm.KL_L2_gaussian(emit_true, perm)
    print "KL divergence to truth: %.3f" % KL
    print "Total L_2 distance between means: %.3f" % L_2

    plt.figure()
    plt.scatter(obs[:,0], obs[:,1])
    for G in prior_emit:
        plt.scatter(*G.mu_mf, color='green')
    for G in hmm.var_emit:
        plt.scatter(*G.mu_mf, color='red')
        util.plot_ellipse(G.mu_mf, G.sigma, edge='r', face='none')
    plt.show()

    #graph true matrix
    #generate_trans_graph(A_true, True10state.png)
    #generate_trans_graph(hmm.var_tran, Var10state.png)

    priors = {'emit': prior_emit, 'tran': prior_tran, 'init': prior_init}
    return hmm, full_var_x, obs, sts, priors


def plot_state_sq(sts):
    fig = plt.figure(figsize=(8,2))
    ax = fig.add_axes([0.05, 0.3, 0.9, 0.5])

    pos, = np.where(np.diff(sts) != 0)
    bounds = np.concatenate(([0], pos+1, [len(sts)]))

    uniq_sts = set(sts)
    jet = plt.get_cmap('jet')
    colors = dict(zip(uniq_sts, jet(np.linspace(0, 1, len(uniq_sts), endpoint=True))))
    sts_seq_uniq = [x[0] for x in groupby(sts)]
    color_map = [colors[st] for st in sts_seq_uniq]

    cmap = mpl.colors.ListedColormap(color_map)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                       norm=norm,
                                       boundaries=bounds,
                                       #ticks=bounds,
                                       spacing='proportional',
                                       orientation='horizontal')
    plt.show()

if __name__ == "__main__":

    #np.random.seed(8675309)
    A_true = np.zeros((10,10))
    A_true[0, [0,1,2]] = [0.495, 0.495, 0.01]
    A_true[1, [0,1,2]] = [0.495, 0.495, 0.01]
    A_true[2, [2,3]] = [0.05, 0.95]
    A_true[3, [3,4]] = [0.05, 0.95]
    A_true[4, [4,5,6]] = [0.1, 0.45, 0.45]
    A_true[5, [5,6,7]] = [0.495, 0.495, 0.01]
    A_true[6, [5,6,7]] = [0.495, 0.495, 0.01]
    A_true[7, [7,8]] = [0.05, 0.95]
    A_true[8, [8,9]] = [0.05, 0.95]
    A_true[9, [0,1,9]] = [0.45, 0.45, 0.1]

                        #   1     2   3   4     5     6    7   8
    #A_true = np.array([[ 0.5,  0.5,  0., 0.,   0.,   0.,  0., 0.],  # 1
    #                   [0.45, 0.45, 0.1, 0.,   0.,   0.,  0., 0.],  # 2
    #                   [  0.,   0.,  0., 1.,   0.,   0.,  0., 0.],  # 3
    #                   [  0.,   0.,  0., 0.,   1.,   0.,  0., 0.],  # 4
    #                   [  0.,   0.,  0., 0.,  0.5,  0.5,  0., 0.],  # 5
    #                   [  0.,   0.,  0., 0., 0.45, 0.45, 0.1, 0.],  # 6
    #                   [  0.,   0.,  0., 0.,   0.,   0.,  0., 1.],  # 7
    #                   [  1.,   0.,  0., 0.,   0.,   0.,  0., 0.]]) # 8

    #A_true = np.zeros((10,10))
    #A_true[0, [0,1,2]] = [0.4975, 0.4975, 0.005]
    #A_true[1, [0,1,2]] = [0.4975, 0.4975, 0.005]
    #A_true[2, [2,3]] = [0.95, 0.05]
    #A_true[3, [3,4]] = [0.95, 0.05]
    #A_true[4, [4,5,6]] = [0.75, 0.125, 0.125]
    #A_true[5, [5,6,7]] = [0.4975, 0.4975, 0.005]
    #A_true[6, [5,6,7]] = [0.4975, 0.4975, 0.005]
    #A_true[7, [7,8]] = [0.95, 0.05]
    #A_true[8, [8,9]] = [0.95, 0.05]
    #A_true[9, [0,1,9]] = [0.125, 0.125, 0.75]


    mu_true = np.array([[0, 20],     # 1
                        [20, 0],     # 2
                        [-30, -30],  # 3
                        [30, -30],   # 4
                        [0, 50],     # 5
                        [-20, 0],    # 6
                        [0, -20],    # 7
                        [30, 30],    # 8
                        [-30, 30],   # 9
                        [0, -50]     # 10
                        ], dtype='float64')

    #hmm, full_var_x, obs, sts, priors = bridge_10state(mu_true, A_true)

                       #   1     2    3     4     5     6    7     8
    A_true = np.array([[0.01, 0.99,   0.,   0.,   0.,   0.,  0.,   0.],  # 1
                       [  0., 0.01, 0.99,   0.,   0.,   0.,  0.,   0.],  # 2
                       [0.85,   0.,   0., 0.15,   0.,   0.,  0.,   0.],  # 3 can find the bridge to 4
                       [  0.,   0.,   0.,   0.,   1.,   0.,  0.,   0.],  # 4 #must go to other component via 5
                       [  0.,   0.,   0.,   0., 0.01,  .99,  0.,   0.],  # 5
                       [  0.,   0.,   0.,   0.,   0.,  .01, .99,   0.],  # 5
                       [  0.,   0.,   0.,   0., 0.85,   0.,  0., 0.15],  # 7 can find bridge to 8
                       [  1.,   0.,   0.,   0.,   0.,   0.,  0.,   0.]]) # 8 #must go to other component via 1
    # we may also experiment with longer cycles but the same structure: 
    # this is the simplest/minimal implementation


    # #identifiable means
    #mu_true = np.array([[-30, 30],      # 1
    #                    [-30, -30],      # 2
    #                    [-50, 0],  # 3
    #                    [-20, -10],   # 4
    #                    [30, 30],     # 5
    #                    [30, -30],     # 6
    #                    [50, 0],     # 7
    #                    [20, 10],    # 8
    #                    ], dtype='float64')


    # less identifiable means: one component is very similar to the other, but the order is reversed
    mu_true = np.array([[-50, 0],      # 1
                        [30, -30],      # 2
                        [30, 30],  # 3
                        [-100, -10],   # 4
                        [ 40, -40],     # 5
                        [-65, 0],     # 6
                        [ 40, 40],     # 7
                        [100, 10],    # 8
                        ], dtype='float64')

    hmm, full_var_x, obs, sts, priors = reversed_cycles(mu_true, A_true)

