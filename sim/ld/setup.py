from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "rng",
        ["rng.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "trainLD",
        ["trainLD.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="ld_project",
    ext_modules=cythonize(
        extensions,
        language_level="3",
    ),
    zip_safe=False,
)
