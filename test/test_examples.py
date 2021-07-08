# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
from subprocess import check_call
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), "examples")

EXAMPLES = [
    "annotation.py --model mn",
    "annotation.py --model ds",
    "annotation.py --model mace",
    "annotation.py --model hds",
    "annotation.py --model id",
    "annotation.py --model lre",
    "baseball.py --num-samples 100 --num-warmup 100 --num-chains 2",
    "bnn.py --num-samples 10 --num-warmup 10 --num-data 7 --num-chains 2",
    "capture_recapture.py --num-samples 4 --num-warmup 1 -m 3",
    "capture_recapture.py --num-samples 4 --num-warmup 1 -m 5",
    "covtype.py --algo HMC --num-samples 10 --num-warmup 10",
    "gaussian_shells.py --num-samples 100",
    "gaussian_shells.py --num-samples 100 --enum",
    "gp.py --num-samples 10 --num-warmup 10 --num-chains 2",
    "hmcecs.py --subsample_size 5 --num_svi_steps 1 --num_blocks 1 "
    "--dataset mock --num_warmup 1 --num_samples 5 --num_datapoints 100",
    "hmm.py --num-samples 100 --num-warmup 100 --num-chains 2",
    "hmm_enum.py -m 1 -t 3 -d 4 --num-warmup 1 -n 4",
    "hmm_enum.py -m 2 -t 3 -d 4 --num-warmup 1 -n 4",
    "hmm_enum.py -m 3 -t 3 -d 3 --num-warmup 1 -n 4",
    "hmm_enum.py -m 3 -t 3 -d 4 --num-warmup 1 -n 4",
    "hmm_enum.py -m 4 -t 3 -d 4 --num-warmup 1 -n 4",
    "hmm_enum.py -m 6 -t 4 -d 3 --num-warmup 1 -n 4",
    "minipyro.py",
    "neutra.py --num-samples 100 --num-warmup 100",
    "ode.py --num-samples 100 --num-warmup 100 --num-chains 1",
    "prodlda.py --num-steps 10 --hidden 10 --nn-framework flax",
    "prodlda.py --num-steps 10 --hidden 10 --nn-framework haiku",
    "sparse_regression.py --num-samples 10 --num-warmup 10 --num-data 10 --num-dimensions 10",
    "stochastic_volatility.py --num-samples 100 --num-warmup 100",
    "ucbadmit.py --num-chains 2",
    "vae.py -n 1",
]


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.filterwarnings("ignore:There are not enough devices:UserWarning")
@pytest.mark.filterwarnings("ignore:Higgs is a 2.6 GB dataset:UserWarning")
def test_cpu(example):
    print("Running:\npython examples/{}".format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
