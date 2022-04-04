from setuptools import setup, find_packages


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
	install_requires = ["pytest>=0","pdoc>=0","numpy>=0"],

	classifiers = [
		"Developmet Status :: 1 - Planning",
		"Intended Audience :: Science/Research",
		"Operating System :: POSIX :: Linux",
		"Operating System :: MacOS :: MacOS X",
		"Programming Language :: Python :: 3",
	],
)