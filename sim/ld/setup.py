from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='trainLD',
    ext_modules=cythonize('trainLD.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)