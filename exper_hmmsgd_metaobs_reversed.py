from __future__ import division

import os
import itertools
import glob
import numpy as np
import scipy.spatial.distance as dist

from experiment import ExperimentSequential as ExpSeq

from pybasicbayes.distributions import Gaussian
from pybasicbayes.util.stats import sample_invwishart
import hmmsgd_metaobs as HMM
import util
from gen_synthetic import generate_data_smoothing

try:
    from numpy.random import multivariate_normal as mvnrand
except IOError:
    from util import mvnrand

# True parameter values
K = 8
D = 2
T = 101  # 10000
miss = 0.2

maxit = 100

left = T // 2  # No missing data in first half of data

                   #   1     2   3    4    5    6    7    8
A_true = np.array([[ .01,  .99,  0.,  0.,  0.,  0.,  0.,  0.],  # 1
                   [  0.,  .01, .99,  0.,  0.,  0.,  0.,  0.],  # 2
                   [ .85,   0.,  0., .15,  0.,  0.,  0.,  0.],  # 3 can find the bridge to 4
                   [  0.,   0.,  0.,  0.,  1.,  0.,  0.,  0.],  # 4 must go to other component via 5
                   [  0.,   0.,  0.,  0., .01, .99,  0.,  0.],  # 5
                   [  0.,   0.,  0.,  0.,  0., .01, .99,  0.],  # 5
                   [  0.,   0.,  0.,  0.,  0., .85,  0., .15],  # 7 can find bridge to 8
                   [  1.,   0.,  0.,  0.,  0.,  0.,  0., 0.]])  # 8 must go to other component via 1

mu_true = np.array([[-50, 0],    # 1
                    [30, -30],   # 2
                    [30, 30],    # 3
                    [-100, -10], # 4
                    [ 40, -40],  # 5
                    [-60, 0],    # 6
                    [ 40, 40],   # 7
                    [100, 10],   # 8
                    ], dtype='float64')


sigma_true = [np.eye(2) for i in xrange(K)]
emit_true = [Gaussian(mu=mu_true[i,:], sigma=sigma_true[i],
                      mu_0=mu_true[i,:], sigma_0=sigma_true[i],
                      kappa_0=1, nu_0=4) for i in xrange(K)]

pi_prior = np.ones(K)
A_prior = np.ones((K,K))


def run_exper(obs, par, mask):
    # Initialize object and run inference
    hmm = HMM.VBHMM(obs, mask=mask, **par)
    hmm.infer()

    resdir = 'res'

    if not os.path.exists(resdir):
        try:
            os.makedirs(resdir)
        except IOError:
            raise RuntimeError("Could not create datadir: %s" % (resdir,))
    else:
        if not os.path.isdir(resdir):
            raise RuntimeError("resdir: %s exists but is not a directory" % (resdir,))

    logprob_f = os.path.join(resdir, 'pred_logprob.txt')
    hd_f = os.path.join(resdir, 'hamming_distances.txt')

    pred_logprob = hmm.pred_logprob_full()
    with open(logprob_f, 'a') as f:
        f.write(str(hmm.metaobs_half) + ', ' + str(hmm.mb_sz) + ', ' + \
                str(pred_logprob) + '\n')
    f.close()

    sts_true = np.array(np.genfromtxt(datadir + '/' + datafn + "_states.txt").astype(int))
    var_x = hmm.full_local_update()
    sts_pred = np.argmax(var_x, axis=1).astype(int)
    match_perm = util.munkres_match(sts_true, sts_pred, K)
    print sts_true
    print sts_pred
    print match_perm[sts_pred]
    print dist.hamming(sts_true, match_perm[sts_pred])
    with open(hd_f, 'a') as f:
         f.write(str(hmm.metaobs_half) + ', ' + str(hmm.mb_sz) + ', ' + \
                str(dist.hamming(sts_true, match_perm[sts_pred])) + '\n')
    f.close()


    # Return hmm object as result, but remove reference to observations and
    # masks so we don't # store lots of copies of the data.
    hmm.obs = None
    hmm.mask = None

    return hmm


def main(name, datadir, datafn, expdir=None, nfolds=1, nrestarts=1, seed=None):
    """ Run experiment on 4 state, two group synthetic data.

        name :

        datadir : Path to directory containing data.

        datafn : Prefix name to files that data and missing masks are stored
                 in.

        expdir : Path to directory to store experiment results.  If None
                 (default), then a directory, `name`_results, is made in the
                 current directory.

        nfolds : Number of folds to generate if datafn is None.

        nrestarts : Number of random initial parameters.

        seed : Random number seed.
    """

    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate/Load data and folds (missing masks)
    # These are the emission distributions for the following tests
    if not os.path.exists(datadir):
        try:
            os.makedirs(datadir)
        except IOError:
            raise RuntimeError("Could not create datadir: %s" % (datadir,))
    else:
        if not os.path.isdir(datadir):
            raise RuntimeError("datadir: %s exists but is not a directory" % (datadir,))

    nodata = True
    if datafn is None:
        datafn = name

    dpath = os.path.join(datadir, datafn + "_data.txt")
    mpath = os.path.join(datadir, datafn + "_fold*.txt")
    try:
        X = np.loadtxt(dpath)
        nodata = False
    except IOError:
        if os.path.exists(dpath) and not os.path.isdir(dpath):
            raise RuntimeError("Could not load data: %s" % (dpath,))

    if nodata:
        print "Could not find data, generating it..."
        X, sts, masks = generate_data_smoothing(A_true, emit_true, T, miss,
                                                left, nfolds)

        dpath = os.path.join(datadir, datafn + "_data.txt")
        np.savetxt(dpath, X)

        spath = os.path.join(datadir, datafn + "_states.txt")
        np.savetxt(spath, sts)

        for i in xrange(nfolds):
            mpath = os.path.join(datadir, datafn + "_fold%d.txt" % (i,))
            np.savetxt(mpath, masks[i])
        print "Data saved to %s" % (dpath,)

    masks = glob.glob(mpath)
    if len(masks) == 0:
        masks = [None]

    # Initialize parameter possibilities

    obs_mean = np.mean(X, axis=0)
    mu_0 = obs_mean
    sigma_0 = 0.75*np.cov(X.T)
    kappa_0 = 0.01
    nu_0 = 4

    prior_init = np.ones(K)
    prior_tran = np.ones((K,K))

    mh = np.array([  1,  5, 10, 50, 100, 300])
    ms = np.array([100, 27, 14,  4,   3,   1])

    par_list = list()
    for r in xrange(nrestarts):
        init_means = np.empty((K,D))
        init_cov = list()
        for k in xrange(K):
            init_means[k,:] = mvnrand(mu_0, cov=sigma_0)
            init_cov.append(sample_invwishart(np.linalg.inv(sigma_0), nu_0))
        # We use prior b/c mu and sigma are sampled here
        prior_emit = np.array([Gaussian(mu=init_means[k,:], sigma=sigma_0,
                                       mu_0=mu_0, sigma_0=sigma_0,
                                       kappa_0=kappa_0, nu_0=nu_0)
                              for k in xrange(K)])

        init_init = np.random.rand(K)
        init_init /= np.sum(init_init)

        init_tran = np.random.rand(K,K)
        init_tran /= np.sum(init_tran, axis=1)[:,np.newaxis]

        # Make dict with initial parameters to pass to experiment.
        pd = {'init_init': init_init, 'init_tran': init_tran,
              'prior_init': prior_init, 'prior_tran': prior_tran,
              'prior_emit': prior_emit, 'tau': 1, 'kappa': 0.7,
              'metaobs_half': mh[r%nrestarts],
              'mb_sz': ms[r%nrestarts], 'maxit': maxit}
        par_list.append(pd)

    # Create ExperimentSequential and call run_exper
    dname = os.path.join(datadir, datafn + "_data.txt")
    exp = ExpSeq('exper_synth_reversed', dname, run_exper, par_list,
                 masks=masks, exper_dir=expdir)
    exp.run()


if __name__ == "__main__":
    """ This tests the reversed cycle.
    """
    name = 'exper_hmmbatchcd_synth_reversed'
    datadir = 'data'
    datafn = 'synth_reversed_T100'
    expdir = None
    nfolds = 5
    # must be multiple of length of mh and ms
    nrestarts = 1 # 300
    seed = 8675309

    main(name, datadir, datafn, expdir, nfolds, nrestarts, seed)
