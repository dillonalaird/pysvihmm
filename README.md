pysvihmm
========

Implementation of stochastic variational inference for Bayesian hidden 
Markov models.

Contents
--------

hmmbatchcd.py : Batch variational inference via coordinate ascent.
hmmbatchsgd.py : Batch VI via natural gradient.
hmmsvi.py : Base implementation of stochastic variational inference (SVI).
            Implementations that require significant changes to the logic
            should be based on this but broken off.
hmmsgd_metaobs.py : SVI with batches of meta-observations.  A meta-observation
    is a group of consecutive observations.  We then form minibatches from
    these.  The natural gradient for the global variables is computed for all
    observations in a meta-observation, and then those are averaged over all
    meta-observations in the minibatch.

gen_synthetic.py : Functions to generate synthetic data.
test_* : Scripts to test correctness of algorithms.
exper_* : Scripts to run experiments.

Cython Modules
--------------
Run `python setup.py build_ext --inplace` to build external Cython modules.
