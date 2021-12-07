# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Deep Markov Model inferred using SteinVI
=================================================
An implementation of a Deep Markov Model in NumPyro based on reference [1][2] and the Pyro DMM example.
This is essentially the DKS variant outlined in the paper.


**Reference:**
    1. Pathwise Derivatives for Multivariate Distributions
    Martin Jankowiak and Theofanis Karaletsos (2019)
    2. Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit and David Sontag (2016)
"""
import argparse

import jax
from jax.example_libraries import stax
import jax.numpy as jnp
import jax.ops

import numpyro
from numpyro.contrib.einstein import SteinVI
from numpyro.contrib.einstein.kernels import RBFKernel
import numpyro.distributions as dist
from numpyro.examples.datasets import JSB_CHORALES, load_dataset
from numpyro.infer import Trace_ELBO
from numpyro.optim import Adam


def _reverse_padded(padded, lengths):
    def _reverse_single(p, length):
        new = jnp.zeros_like(p)
        reverse = jnp.roll(p[::-1], length, axis=0)
        return jax.ops.index_update(new, jax.ops.index[:], reverse)

    return jax.vmap(_reverse_single)(padded, lengths)


def load_data(split="train"):
    _, fetch = load_dataset(JSB_CHORALES, split=split)
    lengths, seqs = fetch(0)
    return (seqs, _reverse_padded(seqs, lengths), lengths)


def _one_hot_chorales(seqs, num_nodes=88):
    return jnp.sum(jnp.array((seqs[..., None] == jnp.arange(num_nodes + 1))), axis=-2)[
        ..., 1:
    ]


def emitter(hidden_dim1, hidden_dim2, out_dim):
    """Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`"""
    return stax.serial(
        stax.Dense(hidden_dim1),
        stax.Relu,
        stax.Dense(hidden_dim2),
        stax.Relu,
        stax.Dense(out_dim),
        stax.Sigmoid,
    )


def transition(gate_hidden_dim, prop_mean_hidden_dim, out_dim):
    """Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in [1].

    **Reference:**
        1. Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
        Rahul G. Krishnan, Uri Shalit and David Sontag (2016)
    """
    gate_init_fun, gate_apply_fun = stax.serial(
        stax.Dense(gate_hidden_dim), stax.Relu, stax.Dense(out_dim), stax.Sigmoid
    )

    prop_mean_init_fun, prop_mean_apply_fun = stax.serial(
        stax.Dense(prop_mean_hidden_dim), stax.Relu, stax.Dense(out_dim)
    )

    mean_init_fun, mean_apply_fun = stax.Dense(out_dim)

    stddev_init_fun, stddev_apply_fun = stax.serial(
        stax.Relu, stax.Dense(out_dim), stax.Softplus
    )

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4 = jax.random.split(rng, num=4)
        _, gate_params = gate_init_fun(k1, input_shape)
        prop_mean_output_shape, prop_mean_params = prop_mean_init_fun(k2, input_shape)
        _, mean_params = mean_init_fun(k3, input_shape)
        _, stddev_params = stddev_init_fun(k4, prop_mean_output_shape)
        return (output_shape, output_shape), (
            gate_params,
            prop_mean_params,
            mean_params,
            stddev_params,
        )

    def apply_fun(params, inputs, **kwargs):
        gate_params, prop_mean_params, mean_params, stddev_params = params
        gt = gate_apply_fun(gate_params, inputs)
        ht = prop_mean_apply_fun(prop_mean_params, inputs)
        mut = (1 - gt) * mean_apply_fun(mean_params, inputs) + gt * ht
        sigmat = stddev_apply_fun(stddev_params, ht)
        return mut, sigmat

    return init_fun, apply_fun


def combiner(hidden_dim, out_dim):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of a Gated Recurrent Unit (GRU) [1], see the `gru` method below.

    **Reference**
        1. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
        Junyoung Chung, Caglar Gulcehre, KyungHyun Cho and Yoshua Bengio (2014)
    """
    mean_init_fun, mean_apply_fun = stax.Dense(out_dim)

    stddev_init_fun, stddev_apply_fun = stax.serial(stax.Dense(out_dim), stax.Softplus)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = jax.random.split(rng, num=2)
        _, mean_params = mean_init_fun(k1, input_shape)
        _, stddev_params = stddev_init_fun(k2, input_shape)
        return (output_shape, output_shape), (mean_params, stddev_params)

    def apply_fun(params, inputs, **kwargs):
        mean_params, stddev_params = params
        mut = mean_apply_fun(mean_params, inputs)
        sigmat = stddev_apply_fun(stddev_params, inputs)
        return mut, sigmat

    return init_fun, apply_fun


def gru(hidden_dim, W_init=stax.glorot_normal()):
    """RNN with GRU. Based on https://github.com/google/jax/pull/2298"""
    input_update_init_fun, input_update_apply_fun = stax.Dense(hidden_dim)
    input_reset_init_fun, input_reset_apply_fun = stax.Dense(hidden_dim)
    input_output_init_fun, input_output_apply_fun = stax.Dense(hidden_dim)

    def init_fun(rng, input_shape):
        indv_input_shape = input_shape[1:]
        output_shape = input_shape[:-1] + (hidden_dim,)
        rng, k1, k2 = jax.random.split(rng, num=3)
        hidden_update_w = W_init(k1, (hidden_dim, hidden_dim))
        _, input_update_params = input_update_init_fun(k2, indv_input_shape)

        rng, k1, k2 = jax.random.split(rng, num=3)
        hidden_reset_w = W_init(k1, (hidden_dim, hidden_dim))
        _, input_reset_params = input_reset_init_fun(k2, indv_input_shape)

        rng, k1, k2 = jax.random.split(rng, num=3)
        hidden_output_w = W_init(k1, (hidden_dim, hidden_dim))
        _, input_output_params = input_output_init_fun(k2, indv_input_shape)

        return output_shape, (
            hidden_update_w,
            input_update_params,
            hidden_reset_w,
            input_reset_params,
            hidden_output_w,
            input_output_params,
        )

    def apply_fun(params, inputs, **kwargs):
        (
            hidden_update_w,
            input_update_params,
            hidden_reset_w,
            input_reset_params,
            hidden_output_w,
            input_output_params,
        ) = params
        inps, lengths, init_hidden = inputs

        def apply_fun_single(prev_hidden, inp):
            i, inpv = inp
            inp_update = input_update_apply_fun(input_update_params, inpv)
            hidden_update = jnp.dot(prev_hidden, hidden_update_w)
            update_gate = stax.sigmoid(inp_update + hidden_update)
            reset_gate = stax.sigmoid(
                input_reset_apply_fun(input_reset_params, inpv)
                + jnp.dot(prev_hidden, hidden_reset_w)
            )
            output_gate = update_gate * prev_hidden + (1 - update_gate) * jnp.tanh(
                input_output_apply_fun(input_output_params, inpv)
                + jnp.dot(reset_gate * prev_hidden, hidden_output_w)
            )
            hidden = jnp.where(
                (i < lengths)[:, None], output_gate, jnp.zeros_like(prev_hidden)
            )
            return hidden, hidden

        return jax.lax.scan(
            apply_fun_single, init_hidden, (jnp.arange(inps.shape[0]), inps)
        )

    return init_fun, apply_fun


def model(
    seqs,
    seqs_rev,
    lengths,
    *,
    max_seq_length=129,
    subsample_size=77,
    latent_dim=32,
    emission_dim=100,
    transition_dim=200,
    data_dim=88,
    gru_dim=150,
    annealing_factor=1.0,
    predict=False,
):
    transition_fn = numpyro.module(
        "transition",
        transition(transition_dim, transition_dim, latent_dim),
        input_shape=(subsample_size, latent_dim),
    )
    emitter_fn = numpyro.module(
        "emitter",
        emitter(emission_dim, emission_dim, data_dim),
        input_shape=(subsample_size, latent_dim),
    )

    z0 = numpyro.param("z0", jnp.zeros((subsample_size, 1, latent_dim)))
    with numpyro.plate(
        "data", seqs.shape[0], subsample_size=subsample_size, dim=-1
    ) as idx:
        seqs_batch = seqs[idx]
        lengths_batch = lengths[idx]

        ones = jnp.ones((subsample_size, max_seq_length, latent_dim))
        masks = jnp.repeat(
            jnp.expand_dims(jnp.arange(max_seq_length), axis=0), subsample_size, axis=0
        ) < jnp.expand_dims(lengths_batch, axis=-1)
        # NB: Mask is to avoid scoring 'z' using distribution at this point
        z = numpyro.sample("z", dist.Normal(0.0, ones).mask(False).to_event(2))
        z_shift = jnp.concatenate([z0, z[:, :-1, :]], axis=-2)
        z_loc, z_scale = transition_fn(z_shift)

        with numpyro.handlers.scale(scale=annealing_factor):
            # Actually score 'z'
            numpyro.sample(
                "z_aux",
                dist.Normal(z_loc, z_scale)
                .mask(jnp.expand_dims(masks, axis=-1))
                .to_event(2),
                obs=z,
            )

        emission_probs = emitter_fn(z)
        oh_x = _one_hot_chorales(seqs_batch)
        if predict:
            oh_x = None
        numpyro.sample(
            "obs_x",
            dist.Bernoulli(emission_probs)
            .mask(jnp.expand_dims(masks, axis=-1))
            .to_event(2),
            obs=oh_x,
        )


def guide(
    seqs,
    seqs_rev,
    lengths,
    *,
    max_seq_length=129,
    subsample_size=77,
    latent_dim=32,
    emission_dim=100,
    transition_dim=200,
    data_dim=88,
    gru_dim=150,
    annealing_factor=1.0,
    predict=False,
):
    seqs_rev = jnp.transpose(seqs_rev, axes=(1, 0, 2))
    combiner_fn = numpyro.module(
        "combiner", combiner(gru_dim, latent_dim), input_shape=(subsample_size, gru_dim)
    )

    with numpyro.plate(
        "data", seqs.shape[0], subsample_size=subsample_size, dim=-1
    ) as idx:
        seqs_rev_batch = seqs_rev[:, idx, :]
        lengths_batch = lengths[idx]

        gru_fn = numpyro.module(
            "gru", gru(gru_dim), input_shape=(max_seq_length, subsample_size, data_dim)
        )
        masks = jnp.repeat(
            jnp.expand_dims(jnp.arange(max_seq_length), axis=0), subsample_size, axis=0
        ) < jnp.expand_dims(lengths_batch, axis=-1)

        h0 = numpyro.param("h0", jnp.zeros((subsample_size, gru_dim)))
        _, hs = gru_fn((_one_hot_chorales(seqs_rev_batch), lengths_batch, h0))
        hs = _reverse_padded(jnp.transpose(hs, axes=(1, 0, 2)), lengths_batch)
        z_loc, z_scale = combiner_fn(hs)
        with numpyro.handlers.scale(scale=annealing_factor):
            numpyro.sample(
                "z",
                dist.Normal(z_loc, z_scale)
                .mask(jnp.expand_dims(masks, axis=-1))
                .to_event(2),
            )


def main(args):
    svgd = SteinVI(
        model,
        guide,
        Adam(1e-5),
        Trace_ELBO(),
        RBFKernel(),
        reinit_hide_fn=lambda site: site["name"].endswith("$params"),
        num_particles=args.num_particles,
    )

    rng_key = jax.random.PRNGKey(seed=args.rng_key)
    seqs, rev_seqs, lengths = load_data()
    results = svgd.run(
        rng_key,
        args.max_iter,
        seqs,
        rev_seqs,
        lengths,
        gru_dim=args.gru_dim,
        subsample_size=args.subsample_size,
        max_seq_length=seqs.shape[1],
    )

    test_seqs, test_rev_seqs, test_lengths = load_data("test")

    negative_elbo = svgd.evaluate(
        results.state,
        test_seqs,
        test_rev_seqs,
        test_lengths,
        gru_dim=args.gru_dim,
        subsample_size=args.subsample_size,
        max_seq_length=test_seqs.shape[1],
    )

    print(f"Negative ELBO: {negative_elbo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample-size", type=int, default=77)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--repulsion", type=float, default=1.0)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--num-particles", type=int, default=5)
    parser.add_argument("--progress-bar", type=bool, default=True)
    parser.add_argument("--gru-dim", type=int, default=150)
    parser.add_argument("--rng-key", type=int, default=142)
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--rng-seed", default=142, type=int)

    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
