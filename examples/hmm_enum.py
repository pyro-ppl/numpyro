# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import pickle
import sys
import time

import jax
from jax import random
import jax.numpy as np

import funsor

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed
from numpyro.handlers import mask as numpyro_mask
from numpyro.primitives import sample as pyro_sample
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.contrib.indexing import Vindex

import numpyro.contrib.funsor.enum_messenger
from numpyro.contrib.funsor.enum_messenger import enum, to_funsor
from numpyro.contrib.funsor.enum_messenger import plate as pyro_plate
from numpyro.contrib.funsor.enum_messenger import markov as pyro_markov
from numpyro.contrib.funsor.enum_messenger import trace as packed_trace

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


def model_0(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              # the parameter expansion here is unfortunate, and
                              # necessitated by the fact that NumPyro allows some
                              # batch dimensions that are not plate or enum dims
                              dist.Beta(0.1 * np.ones((args.hidden_dim, data_dim)),
                                        0.9 * np.ones((args.hidden_dim, data_dim))
                                        ).to_event(2))

    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    for i in pyro_plate("sequences", len(sequences)):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro_markov(range(length)):
            x = pyro_sample("x_{}_{}".format(i, t), dist.Categorical(probs_x[x]),
                            infer={"enumerate": "parallel"})
            logging.info(f"x[{i}, {t}]: {x.shape}")
            with tones_plate:
                pyro_sample("y_{}_{}".format(i, t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                            obs=sequence[t])


def model_1(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              # the parameter expansion here is unfortunate, and
                              # necessitated by the fact that NumPyro allows some
                              # batch dimensions that are not plate or enum dims
                              dist.Beta(0.1 * np.ones((args.hidden_dim, data_dim)),
                                        0.9 * np.ones((args.hidden_dim, data_dim))
                                        ).to_event(2))

    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths).reshape(lengths.shape + (1,))):
                probs_xx = probs_x[x]
                probs_xx = np.broadcast_to(probs_xx, probs_xx.shape[:-3] + (num_sequences, 1) + probs_xx.shape[-1:])
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_xx),
                                infer={"enumerate": "parallel"})
                logging.info(f"x[{t}]: {x.shape}")
                with tones_plate:
                    probs_yx = probs_y[x.squeeze(-1)]
                    probs_yx = np.broadcast_to(probs_yx, probs_yx.shape[:-2] + (num_sequences,) + probs_yx.shape[-1:])
                    pyro_sample("y_{}".format(t),
                                dist.Bernoulli(probs_yx),
                                obs=sequences[batch, t])


# Next let's add a dependency of y[t] on y[t-1].
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
def model_2(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))

        probs_y_shape = (args.hidden_dim, 2, data_dim)
        probs_y = pyro_sample("probs_y",
                              dist.Beta(np.full(probs_y_shape, 0.1),
                                        np.full(probs_y_shape, 0.9))
                                  .to_event(len(probs_y_shape)))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths).reshape(lengths.shape + (1,))):
                probs_xx = probs_x[x]
                probs_xx = np.broadcast_to(probs_xx, probs_xx.shape[:-3] + (num_sequences, 1) + probs_xx.shape[-1:])
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_xx),
                                infer={"enumerate": "parallel"})
                logging.info(f"x[{t}]: {x.shape}")
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with tones_plate as tones:
                    probs_yx = probs_y[x, y, tones]
                    probs_yx = np.broadcast_to(probs_yx, probs_yx.shape[:-2] + (num_sequences,) + probs_yx.shape[-1:])
                    y = pyro_sample("y_{}".format(t),
                                    dist.Bernoulli(probs_yx),
                                    obs=sequences[batch, t]).astype(np.int32)
                    # dist.Bernoulli(probs_y[x, y, tones]),


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
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y_shape = (hidden_dim, hidden_dim, data_dim)
        probs_y = pyro_sample("probs_y",
                              dist.Beta(np.full(probs_y_shape, 0.1),
                                        np.full(probs_y_shape, 0.9))
                                  .to_event(len(probs_y_shape)))

    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        w, x = 0, 0
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths).reshape(lengths.shape + (1,))):
                probs_ww = probs_w[w]
                probs_ww = np.broadcast_to(probs_ww, probs_ww.shape[:-3] + (num_sequences, 1) + probs_ww.shape[-1:])
                w = pyro_sample("w_{}".format(t), dist.Categorical(probs_ww),
                                infer={"enumerate": "parallel"})
                logging.info(f"w[{t}]: {w.shape}")

                probs_xx = probs_x[x]
                probs_xx = np.broadcast_to(probs_xx, probs_xx.shape[:-3] + (num_sequences, 1) + probs_xx.shape[-1:])
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_xx),
                                infer={"enumerate": "parallel"})
                logging.info(f"x[{t}]: {x.shape}")

                with tones_plate as tones:
                    probs_ywx = probs_y[w, x, tones]
                    probs_ywx = np.broadcast_to(
                        probs_ywx, probs_ywx.shape[:-2] + (num_sequences,) + probs_ywx.shape[-1:])
                    pyro_sample("y_{}".format(t), dist.Bernoulli(probs_ywx),
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
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(
                                  np.broadcast_to(0.9 * np.eye(hidden_dim) + 0.1,
                                                  (hidden_dim, hidden_dim, hidden_dim)))
                                  .to_event(2))

        probs_y_shape = (hidden_dim, hidden_dim, data_dim)
        probs_y = pyro_sample("probs_y",
                              dist.Beta(np.full(probs_y_shape, 0.1),
                                        np.full(probs_y_shape, 0.9))
                                  .to_event(len(probs_y_shape)))

    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        # Note the broadcasting tricks here: we declare a hidden arange and
        # ensure that w and x are always tensors so we can unsqueeze them below,
        # thus ensuring that the x sample sites have correct distribution shape.
        w = x = np.array(0)
        for t in pyro_markov(range(max_length)):
            with numpyro_mask(mask_array=(t < lengths).reshape(lengths.shape + (1,))):
                probs_ww = probs_w[w]
                probs_ww = np.broadcast_to(probs_ww, probs_ww.shape[:-3] + (num_sequences, 1) + probs_ww.shape[-1:])
                w = pyro_sample("w_{}".format(t), dist.Categorical(probs_ww),
                                infer={"enumerate": "parallel"})

                probs_xwx = Vindex(probs_x)[w, x]
                probs_xwx = np.broadcast_to(probs_xwx, probs_xwx.shape[:-3] + (num_sequences, 1) + probs_xwx.shape[-1:])
                x = pyro_sample("x_{}".format(t),
                                dist.Categorical(probs_xwx),
                                infer={"enumerate": "parallel"})

                with tones_plate as tones:
                    probs_ywx = probs_y[w, x, tones]
                    probs_ywx = np.broadcast_to(
                        probs_ywx, probs_ywx.shape[:-2] + (num_sequences,) + probs_ywx.shape[-1:])
                    pyro_sample("y_{}".format(t), dist.Bernoulli(probs_ywx),
                                obs=sequences[batch, t])


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
    sequences = np.array(data['sequences'])
    lengths = np.array(data['sequence_lengths'])

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clip(0, args.truncate)
        sequences = sequences[:, :args.truncate]

    # All of our models have two plates: "data" and "tones".
    max_plate_nesting = 1 if model is model_0 else 2

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        model_trace = packed_trace(enum(seed(model, 42), -max_plate_nesting - 1)).get_trace(
            sequences, lengths, args=args)
        for name in model_trace:
            if model_trace[name]['is_observed'] or model_trace[name]['infer'].get('enumerate', None) == 'parallel':
                dim_to_name = model_trace[name]['infer']['dim_to_name']
                logging.info(to_funsor(model_trace[name]['fn'].log_prob(model_trace[name]['value']),
                                       output=funsor.reals(), dim_to_name=dim_to_name).inputs)

    logging.info('Starting inference...')
    rng_key = random.PRNGKey(2)
    start = time.time()
    kernel = {'nuts': NUTS, 'hmc': HMC}[args.kernel](enum(model, -max_plate_nesting - 1))
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, progress_bar=True)
    mcmc.run(rng_key, sequences, lengths, args=args)
    # samples = mcmc.get_samples()  # TODO do something with this
    logging.info('\nMCMC elapsed time:', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMC for HMMs")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument('-n', '--num-samples', nargs='?', default=1000, type=int)

    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("--truncate", type=int)
    parser.add_argument("--num-sequences", default=17, type=int)
    parser.add_argument("--print-shapes", action="store_true")
    parser.add_argument("--kernel", default='nuts', type=str)
    parser.add_argument('--num-warmup', nargs='?', default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument('--time-compilation', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
