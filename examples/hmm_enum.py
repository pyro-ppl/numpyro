# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys

import jax.numpy as np

from numpyro.distributions import constraints

import numpyro.distributions as dist
from numpyro.handlers import mask as numpyro_mask
from numpyro.primitives import sample as pyro_sample
from numpyro.primitives import param as pyro_param

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


def ignore_jit_warnings():
    raise NotImplementedError("TODO")


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 88 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y. We'll drive
# probs_* with the guide, enumerate over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has
# narrow treewidth, therefore admitting efficient inference by message passing.
# Pyro's TraceEnum_ELBO will find an efficient message passing scheme if one
# exists.
def model_0(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with numpyro_mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, data_dim])
                                  .to_event(2))
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    for i in pyro_plate("sequences", len(sequences), batch_size):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro_markov(range(length)):
            # On the next line, we'll overwrite the value of x with an updated
            # value. If we wanted to record all x values, we could instead
            # write x[t] = pyro_sample(...x[t-1]...).
            x = pyro_sample("x_{}_{}".format(i, t), dist.Categorical(probs_x[x]),
                            infer={"enumerate": "parallel"})
            with tones_plate:
                pyro_sample("y_{}_{}".format(i, t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                            obs=sequence[t])
# To see how enumeration changes the shapes of these sample sites, we can use
# the Trace.format_shapes() to print shapes at each site:
# $ python examples/hmm.py -m 0 -n 1 -b 1 -t 5 --print-shapes
# ...
#  Sample Sites:
#   probs_x dist          | 16 16
#          value          | 16 16
#   probs_y dist          | 16 88
#          value          | 16 88
#     tones dist          |
#          value       88 |
# sequences dist          |
#          value        1 |
#   x_178_0 dist          |
#          value    16  1 |
#   y_178_0 dist    16 88 |
#          value       88 |
#   x_178_1 dist    16  1 |
#          value 16  1  1 |
#   y_178_1 dist 16  1 88 |
#          value       88 |
#   x_178_2 dist 16  1  1 |
#          value    16  1 |
#   y_178_2 dist    16 88 |
#          value       88 |
#   x_178_3 dist    16  1 |
#          value 16  1  1 |
#   y_178_3 dist 16  1 88 |
#          value       88 |
#   x_178_4 dist 16  1  1 |
#          value    16  1 |
#   y_178_4 dist    16 88 |
#          value       88 |
#
# Notice that enumeration (over 16 states) alternates between two dimensions:
# -2 and -3.  If we had not used pyro_markov above, each enumerated variable
# would need its own enumeration dimension.


# Next let's make our simple model faster in two ways: first we'll support
# vectorized minibatches of data, and second we'll support the PyTorch jit
# compiler.  To add batch support, we'll introduce a second plate "sequences"
# and randomly subsample data to size batch_size.  To add jit support we
# silence some warnings and try to avoid dynamic program structure.

# Note that this is the "HMM" model in reference [1] (with the difference that
# in [1] the probabilities probs_x and probs_y are not MAP-regularized with
# Dirichlet and Beta distributions for any of the models)
def model_1(sequences, lengths, args, batch_size=None, include_prior=True):
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with numpyro_mask(mask=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, data_dim])
                                  .to_event(2))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    # We subsample batch_size items out of num_sequences items. Note that since
    # we're using dim=-1 for the notes plate, we need to batch over a different
    # dimension, here dim=-2.
    with pyro_plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro_sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                with tones_plate:
                    pyro_sample("y_{}".format(t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                                obs=sequences[batch, t])
# Let's see how batching changes the shapes of sample sites:
# $ python examples/hmm.py -m 1 -n 1 -t 5 --batch-size=10 --print-shapes
# ...
#  Sample Sites:
#   probs_x dist             | 16 16
#          value             | 16 16
#   probs_y dist             | 16 88
#          value             | 16 88
#     tones dist             |
#          value          88 |
# sequences dist             |
#          value          10 |
#       x_0 dist       10  1 |
#          value    16  1  1 |
#       y_0 dist    16 10 88 |
#          value       10 88 |
#       x_1 dist    16 10  1 |
#          value 16  1  1  1 |
#       y_1 dist 16  1 10 88 |
#          value       10 88 |
#       x_2 dist 16  1 10  1 |
#          value    16  1  1 |
#       y_2 dist    16 10 88 |
#          value       10 88 |
#       x_3 dist    16 10  1 |
#          value 16  1  1  1 |
#       y_3 dist 16  1 10 88 |
#          value       10 88 |
#       x_4 dist 16  1 10  1 |
#          value    16  1  1 |
#       y_4 dist    16 10 88 |
#          value       10 88 |
#
# Notice that we're now using dim=-2 as a batch dimension (of size 10),
# and that the enumeration dimensions are now dims -3 and -4.


# Next let's add a dependency of y[t] on y[t-1].
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
#
# Note that this is the "arHMM" model in reference [1].
def model_2(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with numpyro_mask(mask=include_prior):
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, 2, data_dim])
                                  .to_event(3))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask=(t < lengths).unsqueeze(-1)):
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
#
# Note that this is the "FHMM" model in reference [1].
def model_3(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with numpyro_mask(mask=include_prior):
        probs_w = pyro_sample("probs_w",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([hidden_dim, hidden_dim, data_dim])
                                  .to_event(3))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        w, x = 0, 0
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask=(t < lengths).unsqueeze(-1)):
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
#
# Note that this is the "PFHMM" model in reference [1].
def model_4(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with numpyro_mask(mask=include_prior):
        probs_w = pyro_sample("probs_w",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .to_event(1))
        probs_x = pyro_sample("probs_x",
                              dist.Dirichlet(0.9 * np.eye(hidden_dim) + 0.1)
                                  .expand_by([hidden_dim])
                                  .to_event(2))
        probs_y = pyro_sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([hidden_dim, hidden_dim, data_dim])
                                  .to_event(3))
    tones_plate = pyro_plate("tones", data_dim, dim=-1)
    with pyro_plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        # Note the broadcasting tricks here: we declare a hidden arange and
        # ensure that w and x are always tensors so we can unsqueeze them below,
        # thus ensuring that the x sample sites have correct distribution shape.
        w = x = np.array(0)
        for t in pyro_markov(range(max_length if args.jit else lengths.max())):
            with numpyro_mask(mask=(t < lengths).unsqueeze(-1)):
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

    def load_data(filename):
        raise NotImplementedError("TODO")

    logging.info('Loading data')
    data = load_data(args.filename)

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequences'])))
    sequences = data['train']['sequences']
    lengths = data['train']['sequence_lengths']

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        first_available_dim = -2 if model is model_0 else -3
        model_trace = packed_trace(enum(model, first_available_dim)).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        # TODO implement format_shapes?
        # logging.info(model_trace.format_shapes())

    # All of our models have two plates: "data" and "tones".
    max_plate_nesting = 1 if model is model_0 else 2

    # TODO use HMC for inference
    # # We'll train on small minibatches.
    # logging.info('Step\tLoss')
    # for step in range(args.num_steps):
    #     loss = svi.step(sequences, lengths, args=args, batch_size=args.batch_size)
    #     logging.info('{: >5d}\t{}'.format(step, loss / num_observations))


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
