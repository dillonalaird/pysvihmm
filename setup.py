import numpy as np

import os
import sys
import subprocess

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

args = sys.argv[1:]

ext_names = ["hmm_fast", "fdr"]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Note that shell=True should be OK because the command is constant.
    # Just in case the build directory was created by accident, delete it
    for ext_name in ext_names:
        subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
        subprocess.Popen("rm -rf %s.c" % ext_name, shell=True, executable="/bin/bash")
        subprocess.Popen("rm -rf %s.so" % ext_name, shell=True, executable="/bin/bash")

    # Now do a normal clean
    sys.argv[1] = "clean"

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

# Only build for 64-bit target
os.environ['ARCHFLAGS'] = "-arch x86_64"

# Set up extension(s) and build
# You can add as many extensions as you need
for ext_name in ext_names:
    cy_ext = Extension(ext_name,
                       ["%s.pyx" % ext_name],
                       include_dirs=[np.get_include()],
                       library_dirs=[],
                       libraries=['m'],
                       #extra_compile_args=["-g"],
                       #extra_link_args=["-g"]
                       extra_compile_args=["-O2"],
                       extra_link_args=["-O2"]
                       )
    
    # Pass all extensions created above in  list to ext_modules argument
    setup(cmdclass={'build_ext': build_ext},
          ext_modules=[cy_ext])
