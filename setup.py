# to properly compile for development:
# $ python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        name="optimized_src.critical_calc",
        sources=["optimized_src/critical_calc.pyx"]
    )
]

setup (
    packages=['original_src', 'optimized_src'],

    ext_modules = cythonize(extensions, language_level=3),
    include_dirs=[numpy.get_include()]
)


