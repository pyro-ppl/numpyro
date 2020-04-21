# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import logging
import pickle
import sys

import jax.numpy as np

from numpyro.distributions import constraints

import numpyro.distributions as dist
from numpyro.handlers import seed
from numpyro.handlers import mask as numpyro_mask
from numpyro.primitives import sample as pyro_sample
from numpyro.primitives import param as pyro_param

import numpyro.infer.enum_messenger
from numpyro.infer.enum_messenger import enum, infer_config
from numpyro.infer.enum_messenger import plate as pyro_plate
from numpyro.infer.enum_messenger import markov as pyro_markov
from numpyro.infer.enum_messenger import trace as packed_trace

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


@contextlib.contextmanager
def ignore_jit_warnings():
    yield None


def model_0(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9),
                              sample_shape=(args.hidden_dim, data_dim))
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
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9),
                              sample_shape=(args.hidden_dim, data_dim))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
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
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with numpyro_mask(mask_array=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9),
                              sample_shape=(args.hidden_dim, 2, data_dim))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask_array=(t < lengths).unsqueeze(-1)):
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with tones_plate as tones:
                    y = pyro_sample("y_{}".format(t), dist.Bernoulli(probs_y[x, y, tones]),
                                    obs=sequences[batch, t]).long()


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
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with numpyro_mask(mask_array=include_prior):
        probs_w = pyro_sample("probs_w",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9),
                              sample_shape=(hidden_dim, hidden_dim, data_dim))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        w, x = 0, 0
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask_array=(t < lengths).unsqueeze(-1)):
                w = pyro_sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
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
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with numpyro_mask(mask_array=include_prior):
        probs_w = pyro_sample("probs_w",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1).to_event(1),
                              sample_shape=(hidden_dim,))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9),
                              sample_shape=(hidden_dim, hidden_dim, data_dim))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, dim=-2) as batch:
        lengths = lengths[batch]
        # Note the broadcasting tricks here: we declare a hidden arange and
        # ensure that w and x are always tensors so we can unsqueeze them below,
        # thus ensuring that the x sample sites have correct distribution shape.
        w = x = np.array(0)
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask_array=(t < lengths).unsqueeze(-1)):
                w = pyro_sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                # TODO work around lack of Vindex in numpyro
                Vindex = lambda x: x
                x = pyro_sample("x_{}".format(t),
                                dist.Categorical(Vindex(probs_x)[w, x]),
                                infer={"enumerate": "parallel"})
                with tones_plate as tones:
                    pyro_sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):

    model = models[args.model]

    with open('./hmm_enum_data.pkl', 'rb') as f:
        data = pickle.load(f)

    logging.info('-' * 40)
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['sequences'])))
    sequences = data['sequences']
    lengths = data['sequence_lengths']

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())

    # All of our models have two plates: "data" and "tones".
    max_plate_nesting = 1 if model is model_0 else 2

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    model_trace = packed_trace(enum(seed(model, 42), -max_plate_nesting - 1)).get_trace(
        sequences, lengths, args=args)
    # TODO implement format_shapes?
    # logging.info(model_trace.format_shapes())

    # TODO use HMC for inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMC for HMMs")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    args = parser.parse_args()
    main(args)
