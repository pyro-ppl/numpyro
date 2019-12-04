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
        # TODO: pin to a specific version for the release (unless JAX's API becomes stable)
        'jax==0.1.53',
        'jaxlib==0.1.36',
        'tqdm',
    ],
    extras_require={
        'doc': ['sphinx', 'sphinx_rtd_theme', 'sphinx-gallery'],
        'test': [
            'flake8',
            'pytest>=4.1',
            'pyro-api>=0.1.1'
        ],
        'dev': ['ipython', 'isort'],
        'examples': ['matplotlib'],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='probabilistic machine learning bayesian statistics',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
