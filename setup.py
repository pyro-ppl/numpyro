# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Find version
for line in open(os.path.join(PROJECT_PATH, 'numpyro', 'version.py')):
    if line.startswith('__version__ = '):
        version = line.strip().split()[2][1:-1]

# READ README.md for long description on PyPi.
try:
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write('Failed to read README.md:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = ''


setup(
    name='numpyro',
    version=version,
    description='Pyro PPL on NumPy',
    packages=find_packages(include=['numpyro', 'numpyro.*']),
    url='https://github.com/pyro-ppl/numpyro',
    author='Uber AI Labs',
    author_email='npradhan@uber.com',
    install_requires=[
        # TODO: pin to a specific version for the release (until JAX's API becomes stable)
        'jax==0.2',
        # check min version here: https://github.com/google/jax/blob/master/jax/lib/__init__.py#L26
        'jaxlib==0.1.55',
        'tqdm',
    ],
    extras_require={
        'doc': ['sphinx', 'sphinx_rtd_theme', 'sphinx-gallery'],
        'test': [
            'flake8',
            'pytest>=4.1',
            'pyro-api>=0.1.1'
        ],
        'dev': [
            'funsor',
            'ipython',
            'isort',
            'flax',
            'dm-haiku',
            'tfp-nightly==0.12.0.dev20200930',  # TODO: change this to stable release or a specific nightly release
        ],
        'examples': ['matplotlib', 'seaborn', 'graphviz', 'arviz'],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='probabilistic machine learning bayesian statistics',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
