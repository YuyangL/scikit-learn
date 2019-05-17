# FIXME: BROKEN
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

system = 'unix'  # 'windows', 'unix'
fileName = '_criterion'

"""
python setup.py build_ext --inplace
"""

if system == 'unix':
    ext_modules = [Extension(fileName,
                             [fileName + '.pyx'],
                             libraries=["m"],  # Unix-like specific
                             extra_compile_args = ['-ffast-math', '-O3', '-fopenmp'],
                             extra_link_args = ['-fopenmp'])]
else:
    ext_modules = [Extension(fileName,
                             [fileName + '.pyx'],
                             extra_compile_args = ['/openmp'],
                             extra_link_args = ['/openmp'])]

setup(name = fileName,
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      include_dirs = [numpy.get_include()])
