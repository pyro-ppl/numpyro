# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import pickle
import time

from jax import random
import jax.numpy as jnp

import funsor

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.handlers import mask, seed
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.contrib.indexing import Vindex


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
        x, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask_array=(t < lengths)[..., None]):
                x = numpyro.sample("x", dist.Categorical(probs_x[x]))
                with numpyro.plate("tones", data_dim, dim=-1):
                    numpyro.sample("y", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=y)
        return (x, t + 1), None

    scan(transition_fn, (0, 0), jnp.swapaxes(sequences, 0, 1))


# Next let's add a dependency of y[t] on y[t-1].
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
"""
def model_2(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * jnp.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))

        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, 2, data_dim])
                                  .to_event(3))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths)[..., None]):
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                logging.info(f"x[{t}]: {x.shape}")
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with tones_plate as tones:
                    y = pyro_sample("y_{}".format(t),
                                    dist.Bernoulli(probs_y[x, y, tones]),
                                    obs=sequences[batch, t]).astype(jnp.int32)


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
    with numpyro_mask(mask_array=include_prior):
        probs_w = pyro_sample("probs_w",
                              dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, 2, data_dim])
                                  .to_event(3))

    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        w, x = 0, 0
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths)[..., None]):
                w = pyro_sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                logging.info(f"w[{t}]: {w.shape}")
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                logging.info(f"x[{t}]: {x.shape}")
                with tones_plate as tones:
                    pyro_sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])


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
    with numpyro_mask(mask_array=include_prior):
        probs_w = pyro_sample("probs_w",
                              dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                  .expand_by([hidden_dim])
                                  .to_event(2))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([hidden_dim, hidden_dim, data_dim])
                                  .to_event(3))

    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        # Note the broadcasting tricks here: we declare a hidden arange and
        # ensure that w and x are always tensors so we can unsqueeze them below,
        # thus ensuring that the x sample sites have correct distribution shape.
        w = x = jnp.array(0)
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths).reshape(lengths.shape + (1,))):
                w = pyro_sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                x = pyro_sample("x_{}".format(t),
                                dist.Categorical(Vindex(probs_x)[w, x]),
                                infer={"enumerate": "parallel"})
                with tones_plate as tones:
                    pyro_sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])
"""


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):

    model = models[args.model]

    with open('./hmm_enum_data.pkl', 'rb') as f:
        data = pickle.load(f)
    data['sequences'] = data['sequences'][0:args.num_sequences]
    data['sequence_lengths'] = data['sequence_lengths'][0:args.num_sequences]

    logging.info('-' * 40)
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['sequences'])))
    sequences = jnp.array(data['sequences'])
    lengths = jnp.array(data['sequence_lengths'])

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clip(0, args.truncate)
        sequences = sequences[:, :args.truncate]

    logging.info('Starting inference...')
    rng_key = random.PRNGKey(2)
    start = time.time()
    kernel = {'nuts': NUTS, 'hmc': HMC}[args.kernel](model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, progress_bar=True)
    mcmc.run(rng_key, sequences, lengths, args=args)
    mcmc.print_summary()
    logging.info('\nMCMC elapsed time: {}'.format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMC for HMMs")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument('-n', '--num-samples', nargs='?', default=1000, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument('-t', "--truncate", type=int)
    parser.add_argument("--num-sequences", default=17, type=int)
    parser.add_argument("--print-shapes", action="store_true")
    parser.add_argument("--kernel", default='nuts', type=str)
    parser.add_argument('--num-warmup', nargs='?', default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')

    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
