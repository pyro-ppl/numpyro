import os
from subprocess import check_call
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), 'examples')


EXAMPLES = [
    'baseball.py --num-samples 100 --num-warmup 100 --num-chains 2',
    'covtype.py --algo HMC --num-samples 10',
    'hmm.py --num-samples 100 --num-warmup 100 --num-chains 2',
    'bnn.py --num-samples 10 --num-warmup 10 --num-data 7 --num-chains 2',
    'sparse_regression.py --num-samples 10 --num-warmup 10 --num-data 10 --num-dimensions 10',
    'gp.py --num-samples 10 --num-warmup 10 --num-chains 2',
    'minipyro.py',
    'ode.py --num-samples 100 --num-warmup 100 --num-chains 1',
    'stochastic_volatility.py --num-samples 100 --num-warmup 100',
    'ucbadmit.py --num-chains 2',
    'vae.py -n 1',
]


@pytest.mark.parametrize('example', EXAMPLES)
@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
def test_cpu(example):
    print('Running:\npython examples/{}'.format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
