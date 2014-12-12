pysvihmm
========

Implementation of stochastic variational inference for Bayesian hidden 
Markov models.

Contents
--------

### HMM Classes

`hmmbase.py` : Abstract base class for finite variational HMMs.

`hmmsvi.py` : Base implementation of stochastic variational inference (SVI).
  Implementations that require significant changes to the logic should be based
  on this but broken off.

`hmmbatchcd.py` : Batch variational inference via coordinate ascent.

`hmmbatchsgd.py` : Batch VI via natural gradient.

`hmmsgd_metaobs.py` : SVI with batches of meta-observations.  A meta-observation
  is a group of consecutive observations.  We then form minibatches from these.
  The natural gradient for the global variables is computed for all observations
  in a meta-observation, and then those are averaged over all meta-observations
  in the minibatch.

`hmm_fast.pyx` : A fast implemenation of forward filtering backward sampling.

### Test Classes

`gen_synthetic.py` : Functions to generate synthetic data.

`test_*` : Scripts to test correctness of algorithms.


### Utilities

`test_utitlities.py` : Plotting and data generation functions used in the tests.

`util.py` : Miscellaneous files for HMM Classes and Test Classes.

Cython Modules
--------------
Run `python setup.py build_ext --inplace` to build external Cython modules.
