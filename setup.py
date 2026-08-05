from pathlib import Path
import numpy as np
import os, subprocess
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

# Cython extensions with correct folder path (tdqc/numerics/tdqc/)
extensions = [
    Extension(
        "*",
        ["tdqc/numerics/tdqc/*.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-march=native', '-fopenmp', '-Wno-cpp'],
    ),
]

setup(
    name="tdqc",
    python_version=">=3.6",
    description="Deep Q-Learning for Quantum State Preparation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Héloïse Albot",
    author_email="h.albot@physik.uni-muenchen.de",
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=["pytest>=0", "pdoc>=0", "numpy>=0", "pybind11>=0"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}, annotate=True),
)