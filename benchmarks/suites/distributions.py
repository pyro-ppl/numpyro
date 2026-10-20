# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Distribution benchmarks: ``log_prob``, ``sample`` and transform hot paths.

These are the cheapest and least noisy measurements in the report. The
distribution object is usually constructed *inside* the jitted function so that
argument validation and constraint machinery show up in the compile-time
column, where a regression in that layer is visible.
"""

import jax
from jax import random
import jax.numpy as jnp

from benchmarks.harness import benchmark
import numpyro.distributions as dist
from numpyro.distributions.transforms import StickBreakingTransform, biject_to

# Elementwise kernels need a big batch to rise above the sub-millisecond noise
# floor of a shared CI runner; quadratic ones (Cholesky, rejection sampling)
# get their own smaller sizes so a single benchmark cannot dominate the suite.
BATCH = 65536
DIM = 64
MATRIX_BATCH = 4096
MATRIX_DIM = 32
SAMPLE_BATCH = 2048
TRANSFORM_BATCH = 16384
# Drawing the Dirichlet test data is itself expensive, so it gets a smaller
# batch than the other elementwise log_prob benchmarks.
DIRICHLET_BATCH = 16384


def _jit_call(fn, *args):
    """Return a runner that calls ``fn`` under ``jax.jit`` with fixed args."""
    compiled = jax.jit(fn)

    def run():
        return compiled(*args)

    return run


# --------------------------------------------------------------------------- #
# log_prob
# --------------------------------------------------------------------------- #


@benchmark(suite="distributions", warm_repeats=20)
def normal_log_prob():
    """Normal.log_prob over a 65536x64 batch."""
    x = random.normal(random.key(0), (BATCH, DIM))
    return _jit_call(lambda x: dist.Normal(0.0, 1.0).log_prob(x).sum(), x)


@benchmark(suite="distributions", warm_repeats=20)
def multivariate_normal_log_prob():
    """MultivariateNormal.log_prob -- triangular solve dominated."""
    key = random.key(0)
    scale_tril = jnp.linalg.cholesky(
        jnp.eye(MATRIX_DIM) + 0.1 * jnp.ones((MATRIX_DIM, MATRIX_DIM))
    )
    x = random.normal(key, (MATRIX_BATCH, MATRIX_DIM))

    def fn(x, scale_tril):
        d = dist.MultivariateNormal(jnp.zeros(MATRIX_DIM), scale_tril=scale_tril)
        return d.log_prob(x).sum()

    return _jit_call(fn, x, scale_tril)


@benchmark(suite="distributions", warm_repeats=20)
def dirichlet_log_prob():
    """Dirichlet.log_prob -- lgamma heavy, simplex constrained."""
    key = random.key(0)
    conc = jnp.full((DIM,), 2.0)
    x = dist.Dirichlet(conc).sample(key, (DIRICHLET_BATCH,))
    return _jit_call(lambda x, c: dist.Dirichlet(c).log_prob(x).sum(), x, conc)


@benchmark(suite="distributions", warm_repeats=20)
def categorical_log_prob():
    """Categorical(logits).log_prob -- gather plus log_softmax."""
    key = random.key(0)
    logits = random.normal(key, (BATCH, DIM))
    x = random.randint(key, (BATCH,), 0, DIM)
    return _jit_call(
        lambda x, lg: dist.Categorical(logits=lg).log_prob(x).sum(), x, logits
    )


@benchmark(suite="distributions", warm_repeats=20)
def gamma_log_prob():
    """Gamma.log_prob over a 65536x64 batch."""
    key = random.key(0)
    x = dist.Gamma(2.0, 1.0).sample(key, (BATCH, DIM))
    return _jit_call(lambda x: dist.Gamma(2.0, 1.0).log_prob(x).sum(), x)


@benchmark(suite="distributions", warm_repeats=20)
def student_t_log_prob():
    """StudentT.log_prob over a 65536x64 batch."""
    x = random.normal(random.key(0), (BATCH, DIM))
    return _jit_call(lambda x: dist.StudentT(3.0, 0.0, 1.0).log_prob(x).sum(), x)


@benchmark(suite="distributions", warm_repeats=20)
def truncated_normal_log_prob():
    """TruncatedNormal.log_prob -- CDF normalisation path."""
    x = jnp.abs(random.normal(random.key(0), (BATCH, DIM)))
    return _jit_call(
        lambda x: dist.TruncatedNormal(0.0, 1.0, low=0.0).log_prob(x).sum(), x
    )


@benchmark(suite="distributions", warm_repeats=20)
def mixture_same_family_log_prob():
    """MixtureSameFamily.log_prob with 8 Normal components."""
    key = random.key(0)
    locs = random.normal(key, (8,))
    x = random.normal(key, (BATCH,))

    def fn(x, locs):
        mixing = dist.Categorical(logits=jnp.zeros(8))
        component = dist.Normal(locs, 1.0)
        return dist.MixtureSameFamily(mixing, component).log_prob(x).sum()

    return _jit_call(fn, x, locs)


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #


@benchmark(suite="distributions", warm_repeats=20)
def normal_sample():
    """Normal.sample of a 65536x64 batch."""
    return _jit_call(
        lambda k: dist.Normal(0.0, 1.0).sample(k, (BATCH, DIM)), random.key(0)
    )


@benchmark(suite="distributions", warm_repeats=10)
def gamma_sample():
    """Gamma.sample -- rejection sampler with custom JVP."""
    return _jit_call(
        lambda k: dist.Gamma(2.0, 1.0).sample(k, (SAMPLE_BATCH, MATRIX_DIM)),
        random.key(0),
    )


@benchmark(suite="distributions", warm_repeats=10)
def dirichlet_sample():
    """Dirichlet.sample of a 2048x64 batch -- gamma draws plus normalisation."""
    conc = jnp.full((DIM,), 2.0)
    return _jit_call(
        lambda k, c: dist.Dirichlet(c).sample(k, (SAMPLE_BATCH,)),
        random.key(0),
        conc,
    )


@benchmark(suite="distributions", warm_repeats=10)
def lkj_cholesky_sample():
    """LKJCholesky.sample -- onion method over 512 draws."""
    return _jit_call(
        lambda k: dist.LKJCholesky(8, concentration=1.0).sample(k, (512,)),
        random.key(0),
    )


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #


@benchmark(suite="distributions", warm_repeats=20)
def stick_breaking_transform():
    """StickBreakingTransform forward, inverse and log-det."""
    x = random.normal(random.key(0), (TRANSFORM_BATCH, MATRIX_DIM - 1))
    transform = StickBreakingTransform()

    def fn(x):
        y = transform(x)
        return transform.inv(y).sum() + transform.log_abs_det_jacobian(x, y).sum()

    return _jit_call(fn, x)


@benchmark(suite="distributions", warm_repeats=20)
def biject_to_constraints():
    """biject_to over the constraints NUTS unconstrains most often."""
    x = random.normal(random.key(0), (TRANSFORM_BATCH, MATRIX_DIM))

    def fn(x):
        total = biject_to(dist.constraints.positive)(x).sum()
        total += biject_to(dist.constraints.unit_interval)(x).sum()
        total += biject_to(dist.constraints.simplex)(x[..., :-1]).sum()
        total += biject_to(dist.constraints.ordered_vector)(x).sum()
        return total

    return _jit_call(fn, x)
