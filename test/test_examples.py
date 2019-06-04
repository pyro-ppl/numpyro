import os
import sys
from subprocess import check_call

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), 'examples')


EXAMPLES = [
    'baseball.py --num-samples 100 --num-warmup 100',
    'covtype.py --algo HMC --num-samples 10',
    'hmm.py --num-samples 100 --num-warmup 100',
    'bnn.py --num-samples 10 --num-warmup 10 --num-data 7',
    'gp.py --num-samples 10 --num-warmup 10',
    'minipyro.py',
    'stochastic_volatility.py --num-samples 100 --num-warmup 100',
    'ucbadmit.py',
    'vae.py -n 1',
]


@pytest.mark.parametrize('example', EXAMPLES)
@pytest.mark.test_examples
def test_cpu(example):
    print('Running:\npython examples/{}'.format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
