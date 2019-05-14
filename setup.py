from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

setup(
    name='numpyro',
    version='0.0.0',
    description='Pyro PPL on Numpy',
    packages=find_packages(include=['numpyro', 'numpyro.*']),
    url='https://github.com/neerajprad/numpyro',
    author='Uber AI Labs',
    author_email='npradhan@uber.com',
    install_requires=[
        'jax>=0.1.26',
        'jaxlib>=0.1.14',
        'tqdm',
    ],
    extras_require={
        'doc': ['sphinx', 'sphinx_rtd_theme'],
        'test': ['flake8', 'pytest>=4.1'],
        'dev': ['ipython'],
        'examples': ['matplotlib'],
    },
    tests_require=['flake8', 'pytest>=4.1'],
    keywords='probabilistic machine learning bayesian statistics',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
    ],
)
