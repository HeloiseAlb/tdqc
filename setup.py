from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

"""
system_cpp_module = Pybind11Extension(
    'system_cpp',
    [str(fname) for fname in Path('/project/th-scratch/h/H.Albot/ed/tdqc_project/tdqc/numerics/deep_q_learning/system_cpp/').glob("*.cpp")], 
    include_dirs=['include'],
    extra_compile_args=['-O3']
)
"""
setup(
	name = "tdqc",
	python_version = ">=3.6",
	description = "change me",
	long_description = open("README.md").read(),
	long_description_content_type = "text/markdown",
	author = "Héloïse Albot",
	author_email = "h.albot@physik.uni-muenchen.de",
	license = 'BSD 2-clause',
	packages = find_packages(),
	install_requires = ["pytest>=0","pdoc>=0","numpy>=0","pybind11>=0"],
        #ext_modules=[system_cpp_module],
	classifiers = [
		"Developmet Status :: 1 - Planning",
		"Intended Audience :: Science/Research",
		"Operating System :: POSIX :: Linux",
		"Operating System :: MacOS :: MacOS X",
		"Programming Language :: Python :: 3",
	],
cmdclass={"build_ext": build_ext},
)
