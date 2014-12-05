from __future__ import diviison

import numpy as np

from pybasicbayes.distributions import Gaussian


def test_hmmbatchcd():
    """
    """

    K = 2
    D = 2
    kappa_0 = 2
    nu_0 = 2

    emit1 = Gaussian(mu=np.array([0,0]),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=kappa_0,
                     nu_0=nu_0)
    emit2 = Gaussian(mu=np.array([5,5]),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=kappa_0,
                     nu_0=nu_0)
    emit = np.array([emit1, emti2])

    N = 100
    obs = np.array([emit[int(np.round(i/N))].rvs()[0]
                    for i in xrange(N)])

