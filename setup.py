from __future__ import absolute_import, division, print_function

import sys

from setuptools import find_packages, setup


# Convert README.md to rst for display at https://pypi.python.org/pypi/pyro-ppl
# When releasing on pypi, make sure pandoc is on your system:
# $ brew install pandoc          # OS X
# $ sudo apt-get install pandoc  # Ubuntu Linux
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    print(long_description)
except Exception as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = open('README.md').read()

setup(
    name='numpyro',
    version='0.1.0',
    description='Pyro PPL on Numpy',
    packages=find_packages(include=['numpyro', 'numpyro.*']),
    url='https://github.com/neerajprad/numpyro',
    author='Uber AI Labs',
    author_email='npradhan@uber.com',
    install_requires=[
        # TODO: Remove soon as JAX's API becomes stable
        'jax==0.1.35',
        'jaxlib>=0.1.14',
        'tqdm',
    ],
    extras_require={
        'doc': ['sphinx', 'sphinx_rtd_theme'],
        'test': ['flake8', 'pytest>=4.1'],
        'dev': ['ipython'],
        'examples': ['matplotlib'],
    },
    long_description=long_description,
    long_description_content_type='rst',
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
