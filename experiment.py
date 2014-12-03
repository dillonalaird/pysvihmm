from __future__ import division

import os
import importlib
import itertools
import shutil
import numpy as np

import cPickle as pkl


class ExperimentBase(object):
    """ Abstraction of an experiment to run.
    """

    def __init__(self, name, data, func, pars, masks=[None]):
        """
            name : Name of experiment to identify

            data : Path to file containing observations.  File should contain
                   an observation on each line with components separated by a
                   space so that numpy.loadtxt will read it.

            func : Function to perform the experiment given a setting of the
                   parameters.  Needs to return a picklable object containing
                   the desired results.

            pars : List of dicts to initialize algorithm.

            masks : List of paths to file containing mask indicating missing
                   data.  The file should have the same number of lines as the
                   data file.  Each line should contain 1 or 0 depending on if
                   the observation is missing or not, respectively.  The
                   resulting array will be of type bool.
        """
        self.name = name
        self.data = data
        self.func = func
        self.pars = pars
        self.mask = masks


class ExperimentSequential(ExperimentBase):
    """ Experiment that runs different parameter settings sequentially.
    """

    def __init__(self, name, data, func, pars, masks=[None], exper_dir=None):
        """
            name : Name of experiment in order to identify.
                
            data : Path to file containing observations.  File should contain
                   an observation on each line with components separated by a
                   space so that numpy.loadtxt will read it.

            func : Function to perform the experiment given a setting of
                   parameters.  The signature should be:
                   func(obs, pars, mask=None)

                   pars can include any extra parameters that are needed by
                   func.

            pars : List of dicts to initialize algorithm.  Includes paths to
                   missing data mask files.

            hmm_mod : Path to file of the hmm algorithm to run on data.  Module
                      MUST implement a class called VBHMM.

            masks : List of paths to files containing mask indicating missing
                   data.  The files should have the same number of lines as the
                   data file.  Each line should contain 1 or 0 depending on if
                   the observation is missing or not, respectively.  The
                   resulting array will be of type bool.

            exper_dir : Optional.  Path to directory to store results.
                        If None (the default), then a directory is created in
                        the current working directory use `name`.
        """
        super(ExperimentSequential, self).__init__(name, data, func, pars,
                                                   masks)

        # We load the observations and masks to save initialization time for
        # each individual experiment.
        self.obs = np.loadtxt(data)
        self.masks = [np.loadtxt(mfile) if mfile is not None else None for mfile in masks]

        if exper_dir is None:
            self.exper_dir = "./%s_results" % (name,)
        else:
            self.exper_dir = exper_dir

        if os.path.exists(self.exper_dir):
            if not os.path.isdir(self.exper_dir):
                raise RuntimeError("exper_dir: %s exists but is not a directory" % (self.exper_dir,))
        else:
            try:
                os.makedirs(self.exper_dir)
            except:
                raise RuntimeError("Could not create exper_dir: %s" % (self.exper_dir,))

    def run(self):

        masks = self.masks
        pars = self.pars
        par_prod = itertools.product(enumerate(pars), enumerate(masks))
        for (pi, par), (mi, mask) in par_prod:

            res = self.func(self.obs, par, mask)

            # Write pickled results
            exp_dir = self.exper_dir

            outfile = "%s/%s_par%s_fold%s.pkl" % (exp_dir, self.name, pi, mi)
            with open(outfile, 'w') as f:
                pkl.dump(res, f)


class ExperimentMosix(ExperimentBase):
    pass
    """ Experiment that runs different parameter settings on Mosix cluster.
        
        This just writes the necessary files to a directory that can then be
        copied to the cluster and run with `mosrun` and passing the jobfile
        that is written.
    """

    def __init__(self, name, data, script, pars, hmmstr, masks=[None], exper_dir=None,
                 extra_args=None):
        """
            name : Name of experiment in order to identify.
                
            data : Path to file containing observations.  File should contain
                   an observation on each line with components separated by a
                   space so that numpy.loadtxt will read it.

            script : Path to script that implements experiment.  The script
                     needs to implement the following interface:

                     script.py hmm datafile maskfile parfile outfile [extra optional args]

                     hmm : path to hmm module to use.

                     datafile is the path to the data file (must be moved to
                     the cluster somehow).

                     maskfile is the path to the mask file to consider (must be
                     moved to the cluster somehow).

                     parfile is the name of a pickle file containing the
                     parameters dict to initialize an hmm with (it will be
                     stored in the experiment directory).

                     outfile : path to file to write results to.

                     The extra optional args can be whatever `script` needs.
                     They are just passed as a string (see extra_args below) so
                     it is your responsibility that any files it needs to load
                     are copied to the directory.

                     IMPORTANT: All paths should be relative to the experiment
                     directory, so you'll probably need some ..'s.

            pars : List of dicts to initialize algorithm.  Includes paths to
                   missing data mask files.

            hmmstr : Type of hmm algorithm to run on data.  For example
                     'hmmbatchcd', or 'hmmsgd_metaobs'.  Module MUST
                     implement a class called VBHMM.

            masks : List of paths to files containing mask indicating missing
                   data.  The files should have the same number of lines as the
                   data file.  Each line should contain 1 or 0 depending on if
                   the observation is missing or not, respectively.  The
                   resulting array will be of type bool.

            exper_dir : Optional.  Path to directory to store results.
                        If None (the default), then a directory is created in
                        the current working directory use `name`.

            extra_args : Optional, default None.  String of extra args that
                         should be passed to `script`.  Note, these must be
                         constant args for all instantiations of the
                         experiment.
        """
        # We overload the func member to hold the path to the script
        super(ExperimentMosix, self).__init__(name, data, script, pars, masks)

        self.hmmstr = hmmstr
        self.extra_args = extra_args

        if exper_dir is None:
            self.exper_dir = "./%s" % (name,)
        else:
            self.exper_dir = exper_dir

        if os.path.exists(self.exper_dir):
            if not os.path.isdir(self.exper_dir):
                raise RuntimeError("exper_dir: %s exists but is not a directory" % (self.exper_dir,))
        else:
            try:
                os.makedirs(self.exper_dir)
            except:
                raise RuntimeError("Could not create exper_dir: %s" % (self.exper_dir,))

        res_dir = os.path.join(self.exper_dir, "results")
        if not os.path.exists(res_dir):
            try:
                os.makedirs(res_dir)
            except:
                raise RuntimeError("Could not create results directory")
        else:
            if not os.path.isdir(res_dir):
                raise RuntimeError("result_dir: %s exists but is not a directory")

    def run(self):

        # Open job file
        ex_dir = self.exper_dir
        jobfile = os.path.join(ex_dir, "jobfile.job")
        script = self.func
        hmmstr = self.hmmstr

        # Copy hmm module to the experiment directory so we don't need to mess
        # with the pyton path.
        hmmfile = hmmstr + ".py"
        try:
            shutil.copy(hmmfile, ex_dir)
        except:
            raise RuntimeError("Could not copy hmm %s to %s" % (hmmfile, ex_dir))

        with open(jobfile, 'w') as jf:

            data = self.data
            maskfiles = self.mask
            pars = self.pars
            par_prod = itertools.product(enumerate(pars), enumerate(maskfiles))

            for (pi, par), (mi, maskfile) in par_prod:

                par_name = "par_%d.pkl" % (pi,)
                par_file = os.path.join(ex_dir, par_name)
                if not os.path.exists(par_file):
                    with open(par_file, 'w') as pf:
                        pkl.dump(par, pf)

                #res_dir = os.path.join(ex_dir, "results")
                # This is relative to the experiment directory as that's where
                # it will be run.
                res_dir = "results"
                res_name = os.path.join(res_dir, "par_%d_fold_%d.pkl" % (pi, mi))

                cmd = "python %s %s %s %s %s %s" % (script, hmmstr, data, maskfile, par_name, res_name)
                if self.extra_args is not None:
                    cmd = "%s %s" % (cmd, self.extra_args)

                jf.write("%s\n" % (cmd,))
