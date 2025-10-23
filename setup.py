# to properly compile:
# $ python setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup (
    ext_modules = cythonize("critical_calc.pyx"),
    include_dirs=[numpy.get_include()]
)


