# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp
from jax.test_util import check_grads

import numpyro.distributions as dist
from numpyro.ops.gaussian import (
    AffineNormal,
    Gaussian,
    gaussian_tensordot,
    loc_and_scale_tril,
    matrix_and_gaussian_to_gaussian,
    matrix_and_mvn_to_gaussian,
    mvn_to_gaussian,
    sequential_gaussian_filter_sample,
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


def assert_close_gaussian(actual, expected, rtol=1e-4, atol=1e-4):
    assert actual.dim == expected.dim
    assert actual.batch_shape == expected.batch_shape
    assert_allclose(
        actual.log_normalizer, expected.log_normalizer, rtol=rtol, atol=atol
    )
    assert_allclose(actual.info_vec, expected.info_vec, rtol=rtol, atol=atol)
    assert_allclose(actual.precision, expected.precision, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "old_shape,new_shape", [((), (4, 2)), ((3,), (2, 3)), ((1, 3), (5, 3))]
)
def test_expand(old_shape, new_shape):
    g = random_gaussian(random.key(0), old_shape, 2)
    expanded = g.expand(new_shape)
    assert expanded.batch_shape == new_shape
    assert expanded.precision.shape == new_shape + (2, 2)
    value = random.normal(random.key(1), old_shape + (2,))
    assert_allclose(
        expanded.log_density(value),
        jnp.broadcast_to(g.log_density(value), new_shape),
        rtol=1e-6,
    )


def test_cat_and_expand_middle_axis_with_unbroadcast_fields():
    info_vec = random.normal(random.key(0), (2, 4, 1, 3))
    factor = random.normal(random.key(1), (1, 3, 3, 3))
    g = Gaussian(jnp.zeros(()), info_vec, factor @ jnp.swapaxes(factor, -1, -2))
    assert g.batch_shape == (2, 4, 3)
    full = g.expand((2, 4, 3))
    assert full.log_normalizer.shape == (2, 4, 3)
    assert full.precision.shape == (2, 4, 3, 3, 3)
    value = random.normal(random.key(2), (2, 4, 3, 3))
    assert_allclose(full.log_density(value), g.log_density(value), rtol=1e-6)
    assert_close_gaussian(Gaussian.cat([g[:, :1], g[:, 1:]], axis=1), full, 0, 0)
    assert_close_gaussian(Gaussian.cat([g[:, :3], g[:, 3:]], axis=-2), full, 0, 0)
    assert_close_gaussian(Gaussian.cat([g[:1], g[1:]], axis=0), full, 0, 0)
    lifted = g.expand((5, 2, 4, 3))
    assert_allclose(
        lifted.log_density(value),
        jnp.broadcast_to(g.log_density(value), (5, 2, 4, 3)),
        rtol=1e-6,
    )


def test_reshape_round_trip():
    g = random_gaussian(random.key(0), (6,), 3)
    assert_close_gaussian(g.reshape((2, 3)).reshape((6,)), g)


def test_getitem_and_cat():
    g = random_gaussian(random.key(0), (5, 4), 2)
    assert g[..., 1].batch_shape == (5,)
    assert_close_gaussian(Gaussian.cat([g[..., :2], g[..., 2:]], axis=-1), g)
    assert_close_gaussian(Gaussian.cat([g[:2], g[2:]], axis=0), g)
    assert_close_gaussian(
        Gaussian.cat([g[..., 0:4:2], g[..., 1:4:2]], axis=-1)[..., [0, 2, 1, 3]], g
    )


def test_pad_and_permute():
    g = random_gaussian(random.key(0), (3,), 2)
    padded = g.event_pad(left=1, right=2)
    assert padded.dim == 5
    assert_allclose(padded.precision[..., 1:3, 1:3], g.precision)
    assert_allclose(padded.info_vec[..., 0], 0.0)
    perm = jnp.array([1, 0])
    assert_close_gaussian(g.event_permute(perm).event_permute(perm), g)


def test_add_is_additive_in_log_density():
    x = random_gaussian(random.key(0), (3,), 2)
    y = random_gaussian(random.key(1), (), 2)
    value = random.normal(random.key(2), (3, 2))
    assert_allclose(
        (x + y).log_density(value),
        x.log_density(value) + y.log_density(value),
        rtol=1e-4,
    )
    assert_allclose((x + 1.5).log_density(value), x.log_density(value) + 1.5, rtol=1e-4)
    assert (x + y).batch_shape == (3,)
    assert (x + y).precision.shape == (3, 2, 2)


def test_vmap_over_factory_derives_batch_shape():
    def make(loc):
        return Gaussian(jnp.zeros(()), loc, jnp.eye(2))

    g = jax.vmap(make)(jnp.zeros((5, 2)))
    assert g.batch_shape == (5,)
    assert g.log_density(jnp.zeros((5, 2))).shape == (5,)


def test_construction_rejects_mismatched_event_dims():
    with pytest.raises(ValueError, match="precision"):
        Gaussian(jnp.zeros(()), jnp.zeros(2), jnp.eye(3))
    with pytest.raises(ValueError, match="loc and scale"):
        AffineNormal(jnp.zeros((2, 3)), jnp.zeros(2), jnp.ones(5))
    with pytest.raises(ValueError, match="loc and scale"):
        AffineNormal(jnp.zeros((2, 3)), jnp.zeros(5), jnp.ones(2))
    with pytest.raises(ValueError, match="rank"):
        Gaussian(jnp.zeros(()), jnp.zeros(()), jnp.eye(2))
    with pytest.raises(ValueError, match="rank"):
        AffineNormal(jnp.zeros((2, 3)), jnp.zeros(()), jnp.ones(2))


def test_unbroadcast_fields_round_trip_shape_ops():
    g = Gaussian(jnp.zeros(()), random.normal(random.key(0), (3, 2)), jnp.eye(2))
    assert g.batch_shape == (3,)
    assert g[0].batch_shape == ()
    assert_allclose(g[0].info_vec, g.info_vec[0])
    assert_close_gaussian(g.reshape((3, 1)).reshape((3,)), g.expand((3,)))
    assert_close_gaussian(Gaussian.cat([g[:1], g[1:]]), g.expand((3,)))
    with pytest.raises(ValueError, match="batch"):
        Gaussian.cat([g[0], g[0]])


def test_factors_compare_by_identity():
    g = random_gaussian(random.key(0), (), 0)
    h = Gaussian(g.log_normalizer, g.info_vec, g.precision)
    assert len({g, h}) == 2
    assert g == g and g != h
    a = AffineNormal(jnp.zeros((2, 3)), jnp.zeros(2), jnp.ones(2))
    assert len({a, AffineNormal(a.matrix, a.loc, a.scale)}) == 2


def test_numpy_fields():
    g = Gaussian(np.zeros(()), np.zeros(2), np.eye(2))
    assert_allclose(g.event_logsumexp(), 2 * 0.5 * jnp.log(2 * jnp.pi), rtol=1e-5)
    assert g.sample(random.key(0), (4,)).shape == (4, 2)
    assert g.sample(random.key(0)).dtype == jnp.result_type(float)


def test_noise_shape_is_checked():
    g = random_gaussian(random.key(0), (2,), 3)
    with pytest.raises(ValueError, match="noise must have shape"):
        g.sample(noise=jnp.zeros((4, 6)))
    with pytest.raises(ValueError, match="either key or noise"):
        g.sample()


@pytest.mark.parametrize("left,right", [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)])
def test_marginalize_condition_identity(left, right):
    g = random_gaussian(random.key(0), (3,), 4)
    value = random.normal(random.key(1), (3, 4 - left - right))
    marginal = g.marginalize(left=left, right=right)
    assert marginal.batch_shape == (3,)
    assert marginal.precision.shape == (3, 4 - left - right, 4 - left - right)
    if right == 0:
        assert_allclose(
            marginal.log_density(value),
            g.condition(value).event_logsumexp(),
            rtol=1e-4,
        )
    assert_allclose(marginal.event_logsumexp(), g.event_logsumexp(), rtol=1e-4)


def test_marginalize_static_permutations_avoid_gather():
    g = random_gaussian(random.key(0), (3,), 4)
    for f in [lambda g: g.marginalize(left=2), lambda g: g.marginalize(right=1)]:
        assert "gather" not in jax.jit(f).lower(g).as_text()


def test_permute_condition_identity():
    g = random_gaussian(random.key(0), (3,), 5)
    value = random.normal(random.key(1), (3, 5))
    for perm in [np.array([2, 3, 4, 0, 1]), np.array([4, 2, 3, 0, 1])]:
        assert_allclose(
            g.event_permute(perm).log_density(value[..., perm]),
            g.log_density(value),
            rtol=1e-4,
        )
        assert_close_gaussian(
            g.event_permute(perm), g.event_permute(jnp.asarray(perm)), rtol=0, atol=0
        )


def test_condition_and_left_condition():
    g = random_gaussian(random.key(0), (3,), 5)
    a = random.normal(random.key(1), (3, 2))
    b = random.normal(random.key(2), (3, 3))
    ab = jnp.concatenate([a, b], -1)
    assert_allclose(g.condition(b).log_density(a), g.log_density(ab), rtol=1e-4)
    assert_allclose(g.left_condition(a).log_density(b), g.log_density(ab), rtol=1e-4)
    assert g.condition(b).batch_shape == (3,)


def test_event_logsumexp_matches_closed_form():
    g = random_gaussian(random.key(0), (3,), 3)
    mean = jnp.linalg.solve(g.precision, g.info_vec[..., None])[..., 0]
    expected = (
        g.log_normalizer
        + 0.5 * g.dim * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.linalg.slogdet(g.precision)[1]
        + 0.5 * (g.info_vec * mean).sum(-1)
    )
    assert_allclose(g.event_logsumexp(), expected, rtol=1e-5)


def test_event_logsumexp_coupled_precision_float32():
    P = jnp.array([[1e6, 1e3], [1e3, 2.0]], jnp.float32)
    g = Gaussian(jnp.zeros((), jnp.float32), jnp.zeros(2, jnp.float32), P)
    expected = jnp.log(2 * jnp.pi) - 0.5 * jnp.linalg.slogdet(P)[1]
    assert_allclose(g.event_logsumexp(), expected, atol=1e-5)


def test_sample_moments_and_noise():
    g = random_gaussian(random.key(0), (2,), 3)
    samples = g.sample(random.key(1), (20000,))
    assert samples.shape == (20000, 2, 3)
    cov = jnp.linalg.inv(g.precision)
    mean = jnp.einsum("...ij,...j->...i", cov, g.info_vec)
    assert_allclose(samples.mean(0), mean, atol=0.05)
    assert_allclose(jax.vmap(lambda s: jnp.cov(s.T), in_axes=1)(samples), cov, atol=0.1)
    noise = random.normal(random.key(2), (4, 2, 3))
    assert_allclose(
        g.sample(sample_shape=(4,), noise=noise),
        g.sample(random.key(9), (4,), noise=noise),
    )
    assert_allclose(g.sample(noise=jnp.zeros((2, 3))), mean, rtol=1e-4)


@pytest.mark.parametrize(
    "make",
    [
        lambda: random_mvn(random.key(0), (3,), 2),
        lambda: random_mvn(random.key(0), (), 2).expand((3,)),
        lambda: dist.Normal(random.normal(random.key(0), (3, 2)), 0.7).to_event(1),
        lambda: dist.Normal(jnp.zeros(2), 1.0).to_event(1),
        lambda: dist.Normal(0.0, 1.0).expand((3, 2)).to_event(1),
        lambda: dist.ExpandedDistribution(
            dist.Normal(jnp.zeros(2), 0.7).to_event(1), (3,)
        ),
    ],
)
def test_mvn_to_gaussian_matches_log_prob(make):
    d = make()
    value = random.normal(random.key(1), (3, 2))
    g = mvn_to_gaussian(d)
    assert g.batch_shape == d.batch_shape
    assert g.precision.shape == d.batch_shape + (2, 2)
    assert_allclose(g.log_density(value), d.log_prob(value), rtol=1e-4)


def test_mvn_to_gaussian_diagonal_matches_sqrt_form():
    loc = random.normal(random.key(0), (3, 2))
    scale = jnp.exp(0.5 * random.normal(random.key(1), (3, 2)))
    diag = mvn_to_gaussian(dist.Normal(loc, scale).to_event(1))
    full = mvn_to_gaussian(
        dist.MultivariateNormal(loc, scale_tril=jnp.eye(2) * scale[..., None])
    )
    assert_close_gaussian(diag, full, rtol=1e-6, atol=0)


def test_mvn_to_gaussian_rejects_other_types():
    with pytest.raises(TypeError, match=r"got Independent\(StudentT\)"):
        mvn_to_gaussian(dist.StudentT(3.0, jnp.zeros(2), 1.0).to_event(1))


def test_matrix_and_gaussian_to_gaussian_broadcasts_batch():
    x_dim, y_dim = 3, 2
    matrix = random.normal(random.key(0), (4, y_dim, x_dim))
    y_gaussian = random_gaussian(random.key(1), (), y_dim)
    x = random.normal(random.key(2), (4, x_dim))
    y = random.normal(random.key(3), (4, y_dim))
    g = matrix_and_gaussian_to_gaussian(matrix, y_gaussian)
    assert g.batch_shape == (4,)
    assert g.precision.shape == (4, x_dim + y_dim, x_dim + y_dim)
    expected = y_gaussian.log_density(y - jnp.einsum("...ij,...j->...i", matrix, x))
    assert_allclose(g.log_density(jnp.concatenate([x, y], -1)), expected, rtol=1e-4)


@pytest.mark.parametrize("diag", [False, True])
def test_matrix_and_mvn_to_gaussian_density(diag):
    x_dim, y_dim = 3, 2
    matrix = random.normal(random.key(0), (4, y_dim, x_dim))
    noise = (
        dist.Normal(random.normal(random.key(1), (4, y_dim)), 0.5).to_event(1)
        if diag
        else random_mvn(random.key(1), (4,), y_dim)
    )
    x = random.normal(random.key(2), (4, x_dim))
    y = random.normal(random.key(3), (4, y_dim))
    g = matrix_and_mvn_to_gaussian(matrix, noise)
    assert isinstance(g, AffineNormal if diag else Gaussian)
    assert g.dim == x_dim + y_dim
    expected = noise.log_prob(y - jnp.einsum("...ij,...j->...i", matrix, x))
    assert_allclose(g.log_density(jnp.concatenate([x, y], -1)), expected, rtol=1e-4)
    if diag:
        full = g.to_gaussian()
        assert_close_gaussian(g.condition(y), full.condition(y))
        assert isinstance(g.left_condition(x), AffineNormal)
        assert_allclose(g.left_condition(x).log_density(y), expected, rtol=1e-4)
        assert_allclose(g.marginalize(right=y_dim).precision, 0.0)


def test_matrix_and_mvn_to_gaussian_with_prior():
    x_dim, y_dim = 2, 3
    matrix = random.normal(random.key(0), (y_dim, x_dim))
    x_prior = random_mvn(random.key(1), (), x_dim)
    noise = random_mvn(random.key(2), (), y_dim)
    joint = gaussian_tensordot(
        mvn_to_gaussian(x_prior), matrix_and_mvn_to_gaussian(matrix, noise), x_dim
    )
    y_dist = dist.MultivariateNormal(
        matrix @ x_prior.mean + noise.mean,
        covariance_matrix=matrix @ x_prior.covariance_matrix @ matrix.T
        + noise.covariance_matrix,
    )
    y = random.normal(random.key(3), (5, y_dim))
    assert_allclose(joint.log_density(y), y_dist.log_prob(y), rtol=1e-4)


def _tensordot_oracle(x, y, na, nb, nc):
    """Contract ``x`` and ``y`` by marginalizing the shared block of the padded joint."""
    joint = x.event_pad(right=nc) + y.event_pad(left=na)
    if nb == 0:
        return joint
    perm = jnp.concatenate(
        [jnp.arange(na), jnp.arange(na + nb, na + nb + nc), jnp.arange(na, na + nb)]
    )
    return joint.event_permute(perm).marginalize(right=nb)


@pytest.mark.parametrize(
    "na,nb,nc", [(1, 1, 1), (2, 1, 0), (0, 2, 1), (2, 2, 2), (1, 0, 1)]
)
def test_gaussian_tensordot_against_dense(na, nb, nc):
    x = random_gaussian(random.key(0), (3,), na + nb)
    y = random_gaussian(random.key(1), (3,), nb + nc)
    xy = gaussian_tensordot(x, y, nb)
    assert xy.dim == na + nc
    assert xy.batch_shape == (3,)
    assert_close_gaussian(xy, _tensordot_oracle(x, y, na, nb, nc))


@pytest.mark.parametrize("na,nb,nc", [(1, 1, 1), (2, 2, 1)])
@pytest.mark.parametrize("rank", [1, None])
def test_gaussian_tensordot_broadcasts_batch_and_rank_deficient(na, nb, nc, rank):
    x = random_gaussian(random.key(0), (2, 1), na + nb, rank=rank)
    y = random_gaussian(random.key(1), (3,), nb + nc, rank=rank)
    if rank == 1:
        assert jnp.linalg.matrix_rank(x.precision).max() == 1
    xy = gaussian_tensordot(x, y, nb)
    assert xy.dim == na + nc
    assert xy.batch_shape == (2, 3)
    assert xy.precision.shape == (2, 3, na + nc, na + nc)
    assert_close_gaussian(xy, _tensordot_oracle(x, y, na, nb, nc))


def test_sequential_gaussian_tensordot_check_grads():
    T, s = 4, 2
    g = random_gaussian(random.key(0), (T,), 2 * s)
    factor = random.normal(random.key(1), (T, 2 * s, 2 * s))

    def value(info_vec, factor):
        precision = factor @ jnp.swapaxes(factor, -1, -2) + jnp.eye(2 * s)
        chain = Gaussian(g.log_normalizer, info_vec, precision)
        return sequential_gaussian_tensordot(chain).event_logsumexp()

    check_grads(
        value,
        (g.info_vec, factor),
        order=1,
        modes=["rev"],
        eps=1e-2,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("num_steps", list(range(1, 20)))
@pytest.mark.parametrize("state_dim", [1, 2, 3])
def test_sequential_gaussian_tensordot_matches_fold(num_steps, state_dim):
    g = random_gaussian(random.key(num_steps), (2, num_steps), 2 * state_dim)
    expected = g[..., 0]
    for t in range(1, num_steps):
        expected = gaussian_tensordot(expected, g[..., t], state_dim)
    actual = sequential_gaussian_tensordot(g)
    assert_close_gaussian(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("T", [1000, 100_000])
def test_sequential_gaussian_tensordot_long_horizon_is_normalized(T):
    """Contracting ``N(z_0; 0, I)`` with ``T`` zero-mean transition densities
    and marginalizing every state integrates to one, so the exact log
    normalizer and its gradient with respect to the matrix are zero."""
    s = 2
    dtype = jnp.result_type(float)
    if dtype == jnp.float32:
        if T > 1000:
            pytest.skip("float32 drifts by 7.6 nats at 100k steps; x64 covers it")
        # Measured float32 error at T=1000: value 4.2e-2, gradient max 20.5.
        value_atol, grad_atol = 0.1, 62.0
    else:
        # Measured float64 error at T=100k: value 1.1e-9, gradient max 1.0e-6.
        value_atol, grad_atol = 1e-8, 1e-5
    matrix = jnp.array([[0.9, 0.1], [0.0, 0.999]], dtype)
    noise = dist.MultivariateNormal(
        jnp.zeros(s, dtype), covariance_matrix=0.1 * jnp.eye(s, dtype=dtype)
    )
    init = mvn_to_gaussian(
        dist.MultivariateNormal(
            jnp.zeros(s, dtype), covariance_matrix=jnp.eye(s, dtype=dtype)
        )
    )

    def value(matrix):
        g = matrix_and_mvn_to_gaussian(matrix, noise).expand((T,))
        reduced = sequential_gaussian_tensordot(g)
        return gaussian_tensordot(init, reduced, s).event_logsumexp()

    result, grad = jax.jit(jax.value_and_grad(value))(matrix)
    assert result.dtype == dtype and grad.dtype == dtype
    assert_allclose(result, 0.0, atol=value_atol)
    assert_allclose(grad, jnp.zeros_like(grad), atol=grad_atol)


def test_sequential_gaussian_tensordot_rejects_empty_time_axis():
    g = random_gaussian(random.key(0), (2, 3), 4)[..., :0]
    with pytest.raises(ValueError, match="empty time axis"):
        sequential_gaussian_tensordot(g)


def test_loc_and_scale_tril():
    g = random_gaussian(random.key(0), (3,), 2)
    loc, scale_tril = loc_and_scale_tril(g.info_vec, g.precision)
    cov = jnp.linalg.inv(g.precision)
    assert_allclose(loc, jnp.einsum("...ij,...j->...i", cov, g.info_vec), rtol=1e-3)
    assert_allclose(scale_tril @ jnp.swapaxes(scale_tril, -1, -2), cov, rtol=1e-3)


def test_loc_and_scale_tril_x64_keeps_dtype():
    if jnp.result_type(float) == jnp.float32:
        pytest.skip("dtype promotion is only observable with x64")
    g = random_gaussian(random.key(0), (3,), 2)
    loc, scale_tril = loc_and_scale_tril(
        g.info_vec.astype(jnp.float32), g.precision.astype(jnp.float32)
    )
    assert loc.dtype == jnp.float32 and scale_tril.dtype == jnp.float32


def _posterior_marginals(init, trans):
    """Marginal mean of every state via prefix and suffix reductions (oracle)."""
    T, s = trans.batch_shape[-1], init.dim
    means = []
    for t in range(T + 1):
        left = (
            init
            if t == 0
            else gaussian_tensordot(
                init, sequential_gaussian_tensordot(trans[..., :t]), s
            )
        )
        right = (
            sequential_gaussian_tensordot(trans[..., t:]).marginalize(right=s)
            if t < T
            else None
        )
        marginal = left if right is None else left + right
        loc, _ = loc_and_scale_tril(marginal.info_vec, marginal.precision)
        means.append(loc)
    return jnp.stack(means, -2)


@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 7, 8])
@pytest.mark.parametrize("sample_shape", [(), (5,)])
def test_filter_sample_shape_mean_and_grads(num_steps, sample_shape):
    s = 2
    init = random_gaussian(random.key(0), (3,), s)
    trans = random_gaussian(random.key(1), (3, num_steps), 2 * s)
    z = sequential_gaussian_filter_sample(random.key(2), init, trans, sample_shape)
    assert z.shape == sample_shape + (3, num_steps + 1, s)
    mean = sequential_gaussian_filter_sample(
        None, init, trans, noise=jnp.zeros((3, num_steps + 1, s))
    )
    assert_allclose(mean, _posterior_marginals(init, trans), rtol=1e-3, atol=1e-3)
    grad = jax.grad(
        lambda info: sequential_gaussian_filter_sample(
            random.key(2), Gaussian(init.log_normalizer, info, init.precision), trans
        ).sum()
    )(init.info_vec)
    assert jnp.isfinite(grad).all()
    assert jnp.any(grad != 0)


@pytest.mark.parametrize("num_steps", [5, 6])
def test_filter_sample_jit_2d_batch_and_sample_shape(num_steps):
    s, batch, sample_shape = 2, (2, 2), (2, 3)
    init = random_gaussian(random.key(0), batch, s)
    trans = random_gaussian(random.key(1), batch + (num_steps,), 2 * s)
    sampler = jax.jit(sequential_gaussian_filter_sample, static_argnums=3)
    z = sampler(random.key(2), init, trans, sample_shape)
    assert z.shape == sample_shape + batch + (num_steps + 1, s)
    assert_allclose(
        z, sequential_gaussian_filter_sample(random.key(2), init, trans, sample_shape)
    )
    mean = sampler(
        None, init, trans, sample_shape, jnp.zeros(sample_shape + z.shape[2:])
    )
    expected = jnp.broadcast_to(_posterior_marginals(init, trans), mean.shape)
    assert_allclose(mean, expected, rtol=1e-3, atol=1e-3)


def test_filter_sample_check_grads():
    T, s = 3, 2
    init = random_gaussian(random.key(0), (), s)
    trans = random_gaussian(random.key(1), (T,), 2 * s)
    factor = random.normal(random.key(2), (T, 2 * s, 2 * s))
    noise = random.normal(random.key(3), (T + 1, s))

    def sample(init_info, trans_info, factor):
        precision = factor @ jnp.swapaxes(factor, -1, -2)
        return sequential_gaussian_filter_sample(
            None,
            Gaussian(init.log_normalizer, init_info, init.precision),
            Gaussian(trans.log_normalizer, trans_info, precision),
            noise=noise,
        )

    check_grads(
        sample,
        (init.info_vec, trans.info_vec, factor),
        order=1,
        modes=["rev"],
        eps=1e-2,
        rtol=1e-2,
        atol=1e-2,
    )


def test_filter_sample_lowers_without_gather():
    init = random_gaussian(random.key(0), (), 2)
    trans = random_gaussian(random.key(1), (64,), 4)
    lowered = jax.jit(sequential_gaussian_filter_sample).lower(
        random.key(2), init, trans
    )
    assert "gather" not in lowered.as_text()


def test_filter_sample_antithetic():
    init = random_gaussian(random.key(0), (), 2)
    trans = random_gaussian(random.key(1), (5,), 4)
    noise = random.normal(random.key(2), (6, 2))
    z = sequential_gaussian_filter_sample(
        None, init, trans, (3,), noise=jnp.stack([noise, 0 * noise, -noise])
    )
    assert_allclose(z[1], (z[0] + z[2]) / 2, rtol=1e-4, atol=1e-4)
    assert_allclose(
        z,
        sequential_gaussian_filter_sample(
            random.key(9),
            init,
            trans,
            (3,),
            noise=jnp.stack([noise, 0 * noise, -noise]),
        ),
    )


def test_filter_sample_moments():
    init = random_gaussian(random.key(0), (), 1)
    trans = random_gaussian(random.key(1), (3,), 2)
    z = sequential_gaussian_filter_sample(random.key(2), init, trans, (20000,))
    joint = init.event_pad(right=3)
    for t in range(3):
        joint = joint + trans[..., t].event_pad(left=t, right=2 - t)
    cov = jnp.linalg.inv(joint.precision)
    mean = jnp.einsum("...ij,...j->...i", cov, joint.info_vec)
    assert_allclose(z[..., 0].mean(0), mean, atol=0.05)
    assert_allclose(jnp.cov(z[..., 0].T), cov, atol=0.1)


def test_affine_normal_batch_shape_broadcasts_all_fields():
    matrix = jnp.ones((2, 3))
    loc = jnp.zeros((4, 2))
    scale = jnp.ones((2,))
    affine = AffineNormal(matrix, loc, scale)
    assert affine.batch_shape == (4,)
    assert affine.condition(jnp.zeros((4, 2))).batch_shape == (4,)
    assert affine.marginalize(right=2).batch_shape == (4,)
    assert affine.marginalize(right=2).log_normalizer.shape == (4,)
    assert affine.expand((5, 4)).batch_shape == (5, 4)
