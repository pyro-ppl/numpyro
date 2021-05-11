# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Find version
for line in open(os.path.join(PROJECT_PATH, "numpyro", "version.py")):
    if line.startswith("__version__ = "):
        version = line.strip().split()[2][1:-1]

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md:\n  {}\n".format(e))
    sys.stderr.flush()
    long_description = ""


setup(
    name="numpyro",
    version=version,
    description="Pyro PPL on NumPy",
    packages=find_packages(include=["numpyro", "numpyro.*"]),
    url="https://github.com/pyro-ppl/numpyro",
    author="Uber AI Labs",
    install_requires=[
        "jax>=0.2.11",
        "jaxlib>=0.1.62",
        "tqdm",
    ],
    extras_require={
        "doc": [
            "ipython",  # sphinx needs this to render codes
            "jinja2<3.0.0",
            "nbsphinx",
            "sphinx<4.0.0",
            "sphinx_rtd_theme",
            "sphinx-gallery",
        ],
        "test": [
            "black",
            "flake8",
            "isort>=5.0",
            "pytest>=4.1",
            "pyro-api>=0.1.1",
            "scipy>=1.1",
        ],
        "dev": [
            "dm-haiku",
            "flax",
            # TODO: bump funsor version before the release
            "funsor @ git+https://github.com/pyro-ppl/funsor.git@d5574988665dd822ec64e41f2b54b9dc929959dc",
            "graphviz",
            "optax==0.0.6",
            # TODO: change this to tensorflow_probability>0.12.1 when the next version
            # of tfp is released. The current release is not compatible with jax>=0.2.12.
            "tfp-nightly",
        ],
        "examples": ["arviz", "jupyter", "matplotlib", "pandas", "seaborn"],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="probabilistic machine learning bayesian statistics",
    license="Apache License 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
