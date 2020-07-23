# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Enumerate Hidden Markov Model
=============================

This example is ported from [1], which shows how to marginalize out
discrete model variables in Pyro.

This combines MCMC with a variable elimination algorithm, where we
use enumeration to exactly marginalize out some variables from the
joint density.

To marginalize out discrete variables ``x``:

1. Verify that the variable dependency structure in your model
    admits tractable inference, i.e. the dependency graph among
    enumerated variables should have narrow treewidth.
2. Ensure your model can handle broadcasting of the sample values
    of those variables

Note that difference from [1], which uses Python loop, here we use
:func:`~numpryo.contrib.control_flow.scan` to reduce compilation
times (only one step needs to be compiled) of the model. Under the
hood, `scan` stacks all the priors' parameters and values into
an additional time dimension. This allows us computing the joint
density in parallel. In addition, the stacked form allows us
to use the parallel-scan algorithm in [2], which reduces parallel
complexity from O(length) to O(log(length)).

Data are taken from [3]. However, the original source of the data
seems to be the Institut fuer Algorithmen
und Kognitive Systeme at Universitaet Karlsruhe.

**References:**

    1. *Pyro's Hidden Markov Model example*,
       (https://pyro.ai/examples/hmm.html)
    2. *Temporal Parallelization of Bayesian Smoothers*,
       Simo Sarkka, Angel F. Garcia-Fernandez
       (https://arxiv.org/abs/1905.13002)
    3. *Modeling Temporal Dependencies in High-Dimensional Sequences:
       Application to Polyphonic Music Generation and Transcription*,
       Boulanger-Lewandowski, N., Bengio, Y. and Vincent, P.
"""

import argparse
import logging
import os
import time

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import scan
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.examples.datasets import JSB_CHORALES, load_dataset
from numpyro.handlers import mask
from numpyro.infer import HMC, MCMC, NUTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 44 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y.
def model_1(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with mask(mask_array=include_prior):
        probs_x = numpyro.sample("probs_x",
                                 dist.Dirichlet(0.9 * jnp.eye(args.hidden_dim) + 0.1)
                                     .to_event(1))
        probs_y = numpyro.sample("probs_y",
                                 dist.Beta(0.1, 0.9)
                                     .expand([args.hidden_dim, data_dim])
                                     .to_event(2))

    def transition_fn(carry, y):
        x_prev, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask_array=(t < lengths)[..., None]):
                x = numpyro.sample("x", dist.Categorical(probs_x[x_prev]))
                with numpyro.plate("tones", data_dim, dim=-1):
                    numpyro.sample("y", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=y)
        return (x, t + 1), None

    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    # NB swapaxes: we move time dimension of `sequences` to the front to scan over it
    scan(transition_fn, (x_init, 0), jnp.swapaxes(sequences, 0, 1))


# Next let's add a dependency of y[t] on y[t-1].
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
def model_2(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with mask(mask_array=include_prior):
        probs_x = numpyro.sample("probs_x",
                                 dist.Dirichlet(0.9 * jnp.eye(args.hidden_dim) + 0.1)
                                     .to_event(1))

        probs_y = numpyro.sample("probs_y",
                                 dist.Beta(0.1, 0.9)
                                     .expand([args.hidden_dim, 2, data_dim])
                                     .to_event(3))

    def transition_fn(carry, y):
        x_prev, y_prev, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask_array=(t < lengths)[..., None]):
                x = numpyro.sample("x", dist.Categorical(probs_x[x_prev]))
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with numpyro.plate("tones", data_dim, dim=-1) as tones:
                    y = numpyro.sample("y",
                                       dist.Bernoulli(probs_y[x, y_prev, tones]),
                                       obs=y)
        return (x, y, t + 1), None

    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    y_init = jnp.zeros((num_sequences, data_dim), dtype=jnp.int32)
    scan(transition_fn, (x_init, y_init, 0), jnp.swapaxes(sequences, 0, 1))


# Next consider a Factorial HMM with two hidden states.
#
#    w[t-1] ----> w[t] ---> w[t+1]
#        \ x[t-1] --\-> x[t] --\-> x[t+1]
#         \  /       \  /       \  /
#          \/         \/         \/
#        y[t-1]      y[t]      y[t+1]
#
# Note that since the joint distribution of each y[t] depends on two variables,
# those two variables become dependent. Therefore during enumeration, the
# entire joint space of these variables w[t],x[t] needs to be enumerated.
# For that reason, we set the dimension of each to the square root of the
# target hidden dimension.
def model_3(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with mask(mask_array=include_prior):
        probs_w = numpyro.sample("probs_w",
                                 dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                     .to_event(1))
        probs_x = numpyro.sample("probs_x",
                                 dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                     .to_event(1))
        probs_y = numpyro.sample("probs_y",
                                 dist.Beta(0.1, 0.9)
                                     .expand([args.hidden_dim, 2, data_dim])
                                     .to_event(3))

    def transition_fn(carry, y):
        w_prev, x_prev, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask_array=(t < lengths)[..., None]):
                w = numpyro.sample("w", dist.Categorical(probs_w[w_prev]))
                x = numpyro.sample("x", dist.Categorical(probs_x[x_prev]))
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with numpyro.plate("tones", data_dim, dim=-1) as tones:
                    numpyro.sample("y",
                                   dist.Bernoulli(probs_y[w, x, tones]),
                                   obs=y)
        return (w, x, t + 1), None

    w_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    scan(transition_fn, (w_init, x_init, 0), jnp.swapaxes(sequences, 0, 1))


# By adding a dependency of x on w, we generalize to a
# Dynamic Bayesian Network.
#
#     w[t-1] ----> w[t] ---> w[t+1]
#        |  \       |  \       |   \
#        | x[t-1] ----> x[t] ----> x[t+1]
#        |   /      |   /      |   /
#        V  /       V  /       V  /
#     y[t-1]       y[t]      y[t+1]
#
# Note that message passing here has roughly the same cost as with the
# Factorial HMM, but this model has more parameters.
def model_4(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with mask(mask_array=include_prior):
        probs_w = numpyro.sample("probs_w",
                                 dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                     .to_event(1))
        probs_x = numpyro.sample("probs_x",
                                 dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                     .expand_by([hidden_dim])
                                     .to_event(2))
        probs_y = numpyro.sample("probs_y",
                                 dist.Beta(0.1, 0.9)
                                     .expand([hidden_dim, hidden_dim, data_dim])
                                     .to_event(3))

    def transition_fn(carry, y):
        w_prev, x_prev, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask_array=(t < lengths)[..., None]):
                w = numpyro.sample("w", dist.Categorical(probs_w[w_prev]))
                x = numpyro.sample("x", dist.Categorical(Vindex(probs_x)[w, x_prev]))
                with numpyro.plate("tones", data_dim, dim=-1) as tones:
                    numpyro.sample("y",
                                   dist.Bernoulli(probs_y[w, x, tones]),
                                   obs=y)
        return (w, x, t + 1), None

    w_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    scan(transition_fn, (w_init, x_init, 0), jnp.swapaxes(sequences, 0, 1))


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):

    model = models[args.model]

    _, fetch = load_dataset(JSB_CHORALES, split='train', shuffle=False)
    lengths, sequences = fetch()
    if args.num_sequences:
        sequences = sequences[0:args.num_sequences]
        lengths = lengths[0:args.num_sequences]

    logger.info('-' * 40)
    logger.info('Training {} on {} sequences'.format(
        model.__name__, len(sequences)))

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes with default args)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clip(0, args.truncate)
        sequences = sequences[:, :args.truncate]

    logger.info('Each sequence has shape {}'.format(sequences[0].shape))
    logger.info('Starting inference...')
    rng_key = random.PRNGKey(2)
    start = time.time()
    kernel = {'nuts': NUTS, 'hmc': HMC}[args.kernel](model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, args.num_chains,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, sequences, lengths, args=args)
    mcmc.print_summary()
    logger.info('\nMCMC elapsed time: {}'.format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMC for HMMs")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument('-n', '--num-samples', nargs='?', default=1000, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument('-t', "--truncate", type=int)
    parser.add_argument("--num-sequences", type=int)
    parser.add_argument("--kernel", default='nuts', type=str)
    parser.add_argument('--num-warmup', nargs='?', default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')

    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
