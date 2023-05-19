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
    "dais_demo.py --num-svi-steps 10 --num-samples 10 --num-warmup 10",
    pytest.param(
        "gaussian_shells.py --num-samples 100",
        marks=pytest.mark.skipif(
            "CI" in os.environ, reason="The example is flaky on CI."
        ),
    ),
    pytest.param(
        "gaussian_shells.py --num-samples 100 --enum",
        marks=pytest.mark.skipif(
            "CI" in os.environ, reason="The example is flaky on CI."
        ),
    ),
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
    "horseshoe_regression.py --num-samples 10 --num-warmup 10 --num-data 5",
    "hsgp.py --num-samples 10 --num-warmup 10 --num-chains 2",
    "minipyro.py",
    "mortality.py --num-samples 10 --num-warmup 10 --num-chains 2",
    "neutra.py --num-samples 100 --num-warmup 100",
    "ode.py --num-samples 100 --num-warmup 100 --num-chains 1",
    pytest.param(
        "prodlda.py --num-steps 10 --hidden 10 --nn-framework flax",
        marks=pytest.mark.skipif(
            "CI" in os.environ, reason="This example requires a lot of memory."
        ),
    ),
    pytest.param(
        "prodlda.py --num-steps 10 --hidden 10 --nn-framework haiku",
        marks=pytest.mark.skipif(
            "CI" in os.environ, reason="This example requires a lot of memory."
        ),
    ),
    "sparse_regression.py --num-samples 10 --num-warmup 10 --num-data 10 --num-dimensions 10",
    "ssbvm_mixture.py --num-samples 10 --num-warmup 10",
    "stein_dmm.py --max-iter 5 --subsample-size 5 --gru-dim 10",
    "stein_bnn.py --max-iter 10 --subsample-size 10",
    "stochastic_volatility.py --num-samples 100 --num-warmup 100",
    "toy_mixture_model_discrete_enumeration.py  --num-steps=1",
    "ucbadmit.py --num-chains 2",
    "vae.py -n 1",
    "ar2.py --num-samples 10 --num-warmup 10 --num-chains 2",
    "ar2.py --num-samples 10 --num-warmup 10 --num-chains 2 --num-data 10 --unroll-loop",
    "holt_winters.py --T 4 --num-samples 10 --num-warmup 10 --num-chains 2",
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
