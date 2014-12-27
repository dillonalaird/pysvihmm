from __future__ import division

import importlib
import numpy as np

import cPickle as pkl

if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    argc = len(argv)

    if argc != 5:
        raise RuntimeError("usage: run_cluster_simple.py hmmfile datafile maskfile parfile outfile")
    # No extra args for this version but this is where'd they go.

    hmmf, df, mf, pf, outf = argv

    obs = np.loadtxt(df)
    mask = np.loadtxt(mf)

    with open(pf, 'r') as f:
        par = pkl.load(f)

    HMM = importlib.import_module(hmmf)

    hmm = HMM.VBHMM(obs, mask=mask, **par)
    hmm.infer()

    # Remove reference to observations and masks so we don't store lots of
    # copies of the data.
    hmm.obs = None
    hmm.mask = None

    # Write results
    with open(outf, 'w') as f:
        pkl.dump(hmm, f)
