# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import digamma

import numpyro.distributions as dist
from numpyro.ops.gamma_gaussian import (
    GammaFactor,
    GammaGaussian,
    gamma_and_mvn_to_gamma_gaussian,
    gamma_gaussian_tensordot,
    matrix_and_mvn_to_gamma_gaussian,
    sequential_gamma_gaussian_tensordot,
)
from numpyro.ops.gaussian import (
    Gaussian,
    gaussian_tensordot,
    sequential_gaussian_tensordot,
)


def random_gaussian(key, batch_shape, dim, rank=None):
    rank = 2 * dim if rank is None else rank
    k1, k2, k3 = random.split(key, 3)
    log_normalizer = random.normal(k1, batch_shape)
    info_vec = random.normal(k2, batch_shape + (dim,))
    factor = random.normal(k3, batch_shape + (dim, rank))
    return Gaussian(log_normalizer, info_vec, factor @ jnp.swapaxes(factor, -1, -2))


def random_mvn(key, batch_shape, dim):
    k1, k2 = random.split(key)
    factor = random.normal(k2, batch_shape + (dim, dim))
    return dist.MultivariateNormal(
        random.normal(k1, batch_shape + (dim,)),
        covariance_matrix=factor @ jnp.swapaxes(factor, -1, -2) + jnp.eye(dim),
    )


def random_gamma_gaussian(key, batch_shape, dim, rank=None):
    k1, k2, k3 = random.split(key, 3)
    g = random_gaussian(k1, batch_shape, dim, rank)
    loc = random.normal(k2, batch_shape + (dim,))
    info_vec = jnp.einsum("...ij,...j->...i", g.precision, loc)
    alpha = 1.0 + jnp.exp(random.normal(k3, batch_shape)) + 0.5 * dim - 1
    beta = jnp.exp(random.normal(random.fold_in(k3, 1), batch_shape)) + 0.5 * (
        info_vec * loc
    ).sum(-1)
    return GammaGaussian(g.log_normalizer, info_vec, g.precision, alpha, beta)


def gaussian_at(gg, s):
    """The Gaussian factor obtained by fixing the scale variable ``s``."""
    s = jnp.asarray(s)
    return Gaussian(
        gg.log_normalizer + gg.alpha * jnp.log(s) - s * gg.beta,
        s[..., None] * gg.info_vec,
        s[..., None, None] * gg.precision,
    )


def assert_close_gamma_gaussian(actual, expected, rtol=1e-4, atol=1e-4):
    assert actual.dim == expected.dim and actual.batch_shape == expected.batch_shape
    for name in ("log_normalizer", "info_vec", "precision", "alpha", "beta"):
        assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=rtol, atol=atol
        )


def normalized_gamma_gaussian(key, batch_shape, dim):
    """A normalized joint factor and the exact Student-t marginal of ``x``."""
    k1, k2, k3 = random.split(key, 3)
    concentration = 1.0 + jnp.exp(random.normal(k1, batch_shape))
    rate = jnp.exp(random.normal(k2, batch_shape))
    mvn = random_mvn(k3, batch_shape, dim)
    gg = gamma_and_mvn_to_gamma_gaussian(dist.Gamma(concentration, rate), mvn)
    covariance = (rate / concentration)[..., None, None] * mvn.covariance_matrix
    student_t = dist.MultivariateStudentT(
        2 * concentration, mvn.mean, jnp.linalg.cholesky(covariance)
    )
    return gg, student_t


def test_shape_ops():
    gg = random_gamma_gaussian(random.key(0), (6,), 2)
    other = random_gamma_gaussian(random.key(3), (6,), 2)
    assert gg.expand((3, 6)).batch_shape == (3, 6)
    reshaped = gg.reshape((2, 3))
    assert_close_gamma_gaussian(reshaped.reshape((6,)), gg)
    assert_close_gamma_gaussian(GammaGaussian.cat([gg[:2], gg[2:]], axis=0), gg)
    assert_close_gamma_gaussian(
        GammaGaussian.cat([reshaped[:, :1], reshaped[:, 1:]], axis=1),
        reshaped,
        rtol=0,
        atol=0,
    )
    assert gg.event_pad(left=1, right=1).dim == 4
    perm = jnp.array([1, 0])
    assert_close_gamma_gaussian(gg.event_permute(perm).event_permute(perm), gg)
    added = gg + other
    assert_close_gamma_gaussian(
        added, GammaGaussian(*(a + b for a, b in zip(gg._fields(), other._fields())))
    )


def test_construction_rejects_mismatched_shapes():
    zero = jnp.zeros(())
    with pytest.raises(ValueError, match="precision"):
        GammaGaussian(zero, jnp.zeros(2), jnp.eye(3), zero, zero)
    with pytest.raises(ValueError, match="rank"):
        GammaGaussian(zero, zero, jnp.eye(2), zero, zero)
    with pytest.raises(ValueError, match="broadcast"):
        GammaFactor(zero, jnp.ones(2), jnp.ones(3))


def test_construction_rejects_non_broadcastable_alpha_beta():
    zero = jnp.zeros(())
    with pytest.raises(ValueError, match="alpha and beta"):
        GammaGaussian(zero, jnp.zeros(2), jnp.eye(2), jnp.ones(3), jnp.ones(4))
    # Here alpha and beta agree with each other; the batch shape of the
    # Gaussian fields is the term that fails to broadcast.
    with pytest.raises(ValueError, match="alpha and beta"):
        GammaGaussian(
            jnp.zeros(2),
            jnp.zeros((2, 3)),
            jnp.zeros((2, 3, 3)),
            jnp.ones(4),
            jnp.ones(4),
        )


def test_validation_matches_the_gaussian_factor():
    # Without these guards a negative ``dims`` or an oversize conditioning
    # value silently slices the wrong blocks and returns a finite result.
    x = random_gamma_gaussian(random.key(0), (), 2)
    y = random_gamma_gaussian(random.key(1), (), 1)
    with pytest.raises(ValueError, match="dims must be non-negative"):
        gamma_gaussian_tensordot(x, y, -1)
    with pytest.raises(ValueError, match="at most 2 coordinates"):
        x.condition(jnp.zeros(3))


@pytest.mark.parametrize("batch_shape", [(), (2, 3)], ids=str)
def test_shape_ops_and_log_density_across_batch_ranks(batch_shape):
    gg = random_gamma_gaussian(random.key(0), batch_shape, 2)
    assert gg.batch_shape == batch_shape
    expanded = gg.expand((4,) + batch_shape)
    assert expanded.batch_shape == (4,) + batch_shape
    assert_close_gamma_gaussian(expanded[1], gg)
    flat = gg.reshape((math.prod(batch_shape),))
    assert flat.batch_shape == (math.prod(batch_shape),)
    assert_close_gamma_gaussian(flat.reshape(batch_shape), gg)
    x = random.normal(random.key(1), batch_shape + (2,))
    s = jnp.exp(random.normal(random.key(2), batch_shape))
    assert gg.log_density(x, s).shape == batch_shape
    assert_allclose(gg.log_density(x, s), gaussian_at(gg, s).log_density(x), rtol=1e-4)
    assert_allclose(
        gg.marginalize(left=1).log_density(x[..., 1:], s),
        gaussian_at(gg, s).marginalize(left=1).log_density(x[..., 1:]),
        rtol=1e-4,
        atol=1e-4,
    )
    assert gg.condition(x[..., 1:]).batch_shape == batch_shape
    assert gg.event_logsumexp().log_density(s).shape == batch_shape


def test_event_pad_content():
    gg = random_gamma_gaussian(random.key(0), (3,), 2)
    padded = gg.event_pad(left=1, right=2)
    assert padded.dim == 5 and padded.batch_shape == (3,)
    assert_allclose(padded.info_vec[..., 1:3], gg.info_vec)
    assert_allclose(padded.precision[..., 1:3, 1:3], gg.precision)
    assert_allclose(padded.info_vec[..., [0, 3, 4]], jnp.zeros((3, 3)))
    assert_allclose(padded.precision[..., [0, 3, 4], :], jnp.zeros((3, 3, 5)))
    assert_allclose(padded.precision[..., :, [0, 3, 4]], jnp.zeros((3, 5, 3)))
    for name in ("log_normalizer", "alpha", "beta"):
        assert_allclose(getattr(padded, name), getattr(gg, name))
    x = random.normal(random.key(1), (3, 2))
    s = jnp.exp(random.normal(random.key(2), (3,)))
    pad = random.normal(random.key(3), (3, 3))
    value = jnp.concatenate([pad[:, :1], x, pad[:, 1:]], -1)
    assert_allclose(padded.log_density(value, s), gg.log_density(x, s), rtol=1e-5)


def test_event_permute_static_and_traced_paths_agree():
    gg = random_gamma_gaussian(random.key(0), (3,), 4)
    perm = np.array([2, 3, 0, 1])
    assert_close_gamma_gaussian(
        gg.event_permute(perm), gg.event_permute(jnp.asarray(perm)), rtol=0, atol=0
    )
    assert_close_gamma_gaussian(
        jax.jit(lambda g: g.event_permute(perm))(gg),
        gg.event_permute(jnp.asarray(perm)),
        rtol=0,
        atol=0,
    )


def test_log_density_dim_zero_broadcasts_value_and_scale():
    gg = random_gamma_gaussian(random.key(0), (3,), 2).marginalize(right=2)
    assert gg.dim == 0
    x = jnp.zeros((5, 3, 0))
    s = jnp.exp(random.normal(random.key(2), (3,)))
    assert gg.log_density(x, s).shape == (5, 3)
    assert gg.log_density(x, jnp.ones((4, 1, 1))).shape == (4, 5, 3)
    assert_allclose(
        gg.log_density(x, s), jnp.broadcast_to(gg.log_density(x[0], s), (5, 3))
    )


def test_condition_broadcasts_leading_value_dims():
    gg = random_gamma_gaussian(random.key(0), (3,), 4)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    a = random.normal(random.key(1), (7, 3, 2))
    b = random.normal(random.key(3), (7, 3, 2))
    conditioned = gg.condition(b)
    assert conditioned.batch_shape == (7, 3) and conditioned.dim == 2
    assert_allclose(
        conditioned.log_density(a, s),
        gg.log_density(jnp.concatenate([a, b], -1), s),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("left,right", [(1, 0), (0, 1), (0, 2), (1, 1)])
def test_marginalize_and_condition(left, right):
    gg = random_gamma_gaussian(random.key(0), (3,), 4)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    value = random.normal(random.key(1), (3, 4 - left - right))
    marginal = gg.marginalize(left=left, right=right)
    assert_allclose(
        marginal.log_density(value, s),
        gaussian_at(gg, s).marginalize(left=left, right=right).log_density(value),
        rtol=1e-4,
        atol=1e-4,
    )
    if right == 0:
        assert_allclose(
            marginal.log_density(value, s),
            gg.condition(value).event_logsumexp().log_density(s),
            rtol=1e-4,
            atol=1e-4,
        )


def test_marginalize_and_condition_rank_deficient_with_sample_dims():
    # Precision of rank 1 in dimension 4: the marginal over the remaining
    # coordinates has zero precision, and the conditioned 1x1 block is the
    # only factor that must be inverted or Cholesky-factored.
    gg = random_gamma_gaussian(random.key(0), (3,), 4, rank=1)
    assert jnp.all(jnp.linalg.matrix_rank(gg.precision) == 1)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    value = random.normal(random.key(1), (7, 3, 3))
    marginal = gg.marginalize(left=1)
    actual = marginal.log_density(value, s)
    assert actual.shape == (7, 3)
    assert_allclose(
        actual,
        gaussian_at(gg, s).marginalize(left=1).log_density(value),
        rtol=1e-4,
        atol=1e-4,
    )
    conditioned = gg.condition(value)
    assert conditioned.batch_shape == (7, 3) and conditioned.dim == 1
    assert_allclose(
        actual, conditioned.event_logsumexp().log_density(s), rtol=1e-4, atol=1e-4
    )
    a = random.normal(random.key(3), (7, 3, 1))
    assert_allclose(
        conditioned.log_density(a, s),
        gg.log_density(jnp.concatenate([a, value], -1), s),
        rtol=1e-4,
        atol=1e-4,
    )


def test_event_logsumexp_jit_and_grad():
    gg = random_gamma_gaussian(random.key(0), (3,), 2)

    def total(gg):
        return gg.event_logsumexp().logsumexp()

    expected = total(gg)
    assert_allclose(jax.jit(total)(gg), expected, rtol=1e-5)
    grads = jax.grad(lambda gg: total(gg).sum())(gg)
    assert isinstance(grads, GammaGaussian)
    factor = gg.event_logsumexp()
    assert_allclose(grads.log_normalizer, jnp.ones(3))
    assert_allclose(
        grads.alpha, digamma(factor.concentration) - jnp.log(factor.rate), rtol=1e-5
    )
    assert_allclose(grads.beta, -factor.concentration / factor.rate, rtol=1e-5)
    for name in ("info_vec", "precision"):
        assert jnp.all(jnp.isfinite(getattr(grads, name)))


def test_event_logsumexp_and_compound():
    gg = random_gamma_gaussian(random.key(0), (3,), 2)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    factor = gg.event_logsumexp()
    assert isinstance(factor, GammaFactor)
    assert_allclose(
        factor.log_density(s), gaussian_at(gg, s).event_logsumexp(), rtol=1e-4
    )
    assert_allclose(
        factor.log_density(s) - factor.logsumexp(),
        dist.Gamma(factor.concentration, factor.rate).log_prob(s),
        rtol=1e-4,
    )
    gg, student_t = normalized_gamma_gaussian(random.key(3), (3,), 2)
    x = random.normal(random.key(1), (3, 2))
    assert_allclose(gg.compound().log_prob(x), student_t.log_prob(x), rtol=1e-4)
    assert_allclose(gg.event_logsumexp().logsumexp(), jnp.zeros(3), atol=1e-4)


def test_compound_matches_student_t_x64():
    if jnp.result_type(float) == jnp.float32:
        pytest.skip("the exact Student-t oracle is checked tightly with x64 only")
    gg, student_t = normalized_gamma_gaussian(random.key(3), (3,), 4)
    x = random.normal(random.key(1), (5, 3, 4))
    assert_allclose(gg.compound().log_prob(x), student_t.log_prob(x), rtol=1e-9)
    assert_allclose(gg.event_logsumexp().logsumexp(), jnp.zeros(3), atol=1e-9)


def test_gamma_and_mvn_to_gamma_gaussian():
    gamma = dist.Gamma(jnp.array([2.0, 3.0]), jnp.array([1.5, 0.5]))
    mvn = random_mvn(random.key(0), (2,), 3)
    gg = gamma_and_mvn_to_gamma_gaussian(gamma, mvn)
    x = random.normal(random.key(1), (2, 3))
    s = jnp.array([0.7, 2.5])
    expected = gamma.log_prob(s) + dist.MultivariateNormal(
        mvn.mean, precision_matrix=s[:, None, None] * mvn.precision_matrix
    ).log_prob(x)
    assert_allclose(gg.log_density(x, s), expected, rtol=1e-4)
    assert gg.batch_shape == (2,)


def test_matrix_and_mvn_to_gamma_gaussian():
    x_dim, y_dim = 2, 3
    matrix = random.normal(random.key(0), (4, y_dim, x_dim))
    mvn = random_mvn(random.key(1), (4,), y_dim)
    x = random.normal(random.key(2), (4, x_dim))
    y = random.normal(random.key(3), (4, y_dim))
    s = jnp.exp(random.normal(random.key(4), (4,)))
    gg = matrix_and_mvn_to_gamma_gaussian(matrix, mvn)
    expected = dist.MultivariateNormal(
        jnp.einsum("...ij,...j->...i", matrix, x) + mvn.mean,
        precision_matrix=s[:, None, None] * mvn.precision_matrix,
    ).log_prob(y)
    assert_allclose(gg.log_density(jnp.concatenate([x, y], -1), s), expected, rtol=1e-4)
    diag = dist.Normal(
        random.normal(random.key(5), (4, y_dim)),
        jnp.exp(0.1 * random.normal(random.key(6), (4, y_dim))),
    ).to_event(1)
    full = dist.MultivariateNormal(
        diag.mean, covariance_matrix=jnp.eye(y_dim) * diag.variance[..., None, :]
    )
    assert_close_gamma_gaussian(
        matrix_and_mvn_to_gamma_gaussian(matrix, diag),
        matrix_and_mvn_to_gamma_gaussian(matrix, full),
    )
    with pytest.raises(TypeError):
        matrix_and_mvn_to_gamma_gaussian(
            matrix, dist.StudentT(3.0, jnp.zeros((4, y_dim)), 1.0).to_event(1)
        )


@pytest.mark.parametrize(
    "na,nb,nc", [(1, 1, 1), (2, 1, 0), (0, 2, 1), (2, 2, 2), (0, 2, 0)]
)
def test_gamma_gaussian_tensordot_matches_fixed_scale(na, nb, nc):
    x = random_gamma_gaussian(random.key(0), (3,), na + nb)
    y = random_gamma_gaussian(random.key(1), (3,), nb + nc)
    s = jnp.exp(random.normal(random.key(2), (3,)))
    z = random.normal(random.key(3), (3, na + nc))
    actual = gamma_gaussian_tensordot(x, y, nb)
    assert actual.dim == na + nc and actual.batch_shape == (3,)
    expected = gaussian_tensordot(gaussian_at(x, s), gaussian_at(y, s), nb)
    assert_allclose(
        actual.log_density(z, s), expected.log_density(z), rtol=1e-4, atol=1e-4
    )


# Covers every branch of the log2 reduction: no loop (1), a single even step
# (2, 4, 8, 16), an odd length with a leftover tail (3, 5, 7, 9, 17).
@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5, 7, 8, 9, 16, 17])
@pytest.mark.parametrize("state_dim", [1, 2])
def test_sequential_gamma_gaussian_tensordot_matches_fold(num_steps, state_dim):
    g = random_gamma_gaussian(random.key(num_steps), (2, num_steps), 2 * state_dim)
    expected = g[..., 0]
    for t in range(1, num_steps):
        expected = gamma_gaussian_tensordot(expected, g[..., t], state_dim)
    assert_close_gamma_gaussian(
        sequential_gamma_gaussian_tensordot(g), expected, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("num_steps", [3, 4])
def test_sequential_reductions_agree_at_fixed_scale(num_steps):
    g = random_gamma_gaussian(random.key(num_steps), (2, num_steps), 4)
    s = jnp.array([0.7, 2.5])
    expected = sequential_gaussian_tensordot(gaussian_at(g, s[:, None]))
    actual = gaussian_at(sequential_gamma_gaussian_tensordot(g), s)
    assert actual.batch_shape == expected.batch_shape == (2,)
    for name in ("log_normalizer", "info_vec", "precision"):
        assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-4, atol=1e-4
        )
