# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import lax, random
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.distributions.hmm import GaussianHMM, IndependentHMM


def _spd(key, n, scale=1.0):
    w = random.normal(key, (n, n))
    return scale * (w @ w.T / n + jnp.eye(n))


def _hmm(key, T, n, m, *, batch=(), homogeneous=False, diag=False, num_steps=None):
    ks = random.split(key, 6)
    tshape = ((1,) if batch else ()) if homogeneous else (T,)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], batch + tshape + (n, n))
    H = random.normal(ks[1], batch + tshape + (m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], batch + (n,)), covariance_matrix=_spd(ks[3], n)
    )
    if diag:
        trans = dist.Normal(jnp.zeros(batch + tshape + (n,)), 0.5).to_event(1)
        obs = dist.Normal(jnp.zeros(batch + tshape + (m,)), 0.3).to_event(1)
    else:
        trans = dist.MultivariateNormal(
            jnp.zeros(batch + tshape + (n,)), covariance_matrix=_spd(ks[4], n, 0.5)
        )
        obs = dist.MultivariateNormal(
            jnp.zeros(batch + tshape + (m,)), covariance_matrix=_spd(ks[5], m, 0.3)
        )
    return GaussianHMM(
        init, A, trans, H, obs, num_steps=num_steps or (T if homogeneous else None)
    )


def dense_joint(m0, P0, A, b, Q, H, d, R):
    """Exact mean and covariance of ``(z_{0:T}, x_{1:T})`` from the Jacobian of the sequential sampler."""
    T, n, m = A.shape[0], A.shape[-1], H.shape[-2]
    L0, LQ, LR = jnp.linalg.cholesky(P0), jnp.linalg.cholesky(Q), jnp.linalg.cholesky(R)

    def f(eps):
        e0 = eps[:n]
        e = eps[n : n + T * n].reshape(T, n)
        w = eps[n + T * n :].reshape(T, m)
        z0 = m0 + L0 @ e0
        c = b + jnp.einsum("tij,tj->ti", LQ, e)
        _, z = lax.scan(lambda zp, inp: (inp[0] @ zp + inp[1],) * 2, z0, (A, c))
        z = jnp.concatenate([z0[None], z])
        x = jnp.einsum("tij,tj->ti", H, z[1:]) + d + jnp.einsum("tij,tj->ti", LR, w)
        return jnp.concatenate([z.ravel(), x.ravel()])

    D = n + T * n + T * m
    J = jax.jacobian(f)(jnp.zeros(D))
    return f(jnp.zeros(D)), J @ J.T


def dense_reference(init, A, trans, H, obs, T):
    """Return ``(mean, cov)`` of the dense joint for MVN components with time-leading shapes."""

    def bt(a, k):
        if a.ndim == k or a.shape[0] == 1:
            return jnp.broadcast_to(a, (T,) + a.shape[a.ndim - k :])
        return a

    return dense_joint(
        init.mean,
        init.covariance_matrix,
        bt(A, 2),
        bt(trans.mean, 1),
        bt(trans.covariance_matrix, 2),
        bt(H, 2),
        bt(obs.mean, 1),
        bt(obs.covariance_matrix, 2),
    )


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("diag", [False, True])
def test_gaussian_hmm_shapes(batch, homogeneous, diag):
    T, n, m = 5, 3, 2
    hmm = _hmm(random.key(0), T, n, m, batch=batch, homogeneous=homogeneous, diag=diag)
    assert hmm.batch_shape == batch
    assert hmm.event_shape == (T, m)
    assert hmm.hidden_dim == n and hmm.obs_dim == m
    x = random.normal(random.key(1), (4,) + batch + (T, m))
    assert hmm.log_prob(x).shape == (4,) + batch
    assert hmm.log_prob(x[0]).shape == batch
    assert hmm.filter(x).batch_shape == (4,) + batch
    assert hmm.filter(x).event_shape == (n,)
    expanded = hmm.expand((7,) + batch)
    assert isinstance(expanded, GaussianHMM)
    assert expanded.log_prob(x[0]).shape == (7,) + batch


def test_gaussian_hmm_expanded_components():
    T, n, m = 4, 2, 1
    init = dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)).expand((3,))
    trans = dist.Normal(0.0, 1.0).expand((3, T, n)).to_event(1)
    obs = dist.MultivariateNormal(jnp.zeros(m), jnp.eye(m)).expand((3, T))
    hmm = GaussianHMM(init, jnp.eye(n), trans, jnp.ones((m, n)), obs)
    assert hmm.batch_shape == (3,)
    assert hmm.event_shape == (T, m)
    assert jnp.isfinite(hmm.log_prob(jnp.zeros((3, T, m)))).all()


@pytest.mark.parametrize(
    "T,n,m", [(1, 2, 1), (2, 3, 2), (7, 3, 2), (8, 1, 1), (5, 1, 3)]
)
def test_gaussian_hmm_log_prob_and_filter_match_dense(T, n, m):
    ks = random.split(random.key(T), 6)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        0.3 * random.normal(ks[4], (T, n)), covariance_matrix=_spd(ks[5], n, 0.5)
    )
    obs = dist.MultivariateNormal(
        0.3 * random.normal(ks[0], (T, m)), covariance_matrix=_spd(ks[1], m, 0.3)
    )
    hmm = GaussianHMM(init, A, trans, H, obs, num_steps=T)
    x = random.normal(random.key(1), (T, m))
    mean, cov = dense_reference(init, A, trans, H, obs, T)
    nz = (T + 1) * n
    expected = dist.MultivariateNormal(
        mean[nz:], covariance_matrix=cov[nz:, nz:]
    ).log_prob(x.ravel())
    assert_allclose(hmm.log_prob(x), expected, rtol=1e-4, atol=1e-4)
    Szz, Sxx, Szx = cov[:nz, :nz], cov[nz:, nz:], cov[:nz, nz:]
    K = jnp.linalg.solve(Sxx, Szx.T).T
    post_mean = (mean[:nz] + K @ (x.ravel() - mean[nz:]))[-n:]
    post_cov = (Szz - K @ Szx.T)[-n:, -n:]
    posterior = hmm.filter(x)
    assert_allclose(posterior.mean, post_mean, rtol=1e-3, atol=1e-3)
    assert_allclose(posterior.covariance_matrix, post_cov, rtol=1e-3, atol=1e-3)


def test_gaussian_hmm_jit_vmap_scan_and_treedef():
    T, n, m = 4, 2, 1
    hmm = _hmm(random.key(0), T, n, m, homogeneous=True)
    x = random.normal(random.key(1), (5, T, m))
    assert_allclose(jax.jit(hmm.log_prob)(x), hmm.log_prob(x), rtol=1e-5)
    assert_allclose(jax.vmap(hmm.log_prob)(x), hmm.log_prob(x), rtol=1e-5)

    def make(scale):
        return GaussianHMM(
            dist.Normal(jnp.zeros(n), 1.0).to_event(1),
            jnp.eye(n),
            dist.Normal(jnp.zeros(n), scale).to_event(1),
            jnp.ones((m, n)),
            dist.Normal(jnp.zeros(m), 0.3).to_event(1),
            num_steps=T,
        )

    mapped = jax.vmap(make)(jnp.array([0.5, 1.0, 2.0]))
    assert mapped.batch_shape == (3,)
    assert mapped.log_prob(x[:3]).shape == (3,)
    assert mapped.filter(x[:3]).batch_shape == (3,)
    expanded = hmm.expand((3,))
    fresh = _hmm(random.key(0), T, n, m, homogeneous=True).expand((3,))
    assert jax.tree.structure(expanded) == jax.tree.structure(fresh)
    carried, _ = lax.scan(lambda h, _: (h, None), expanded, None, length=2)
    assert carried.batch_shape == (3,)
    assert "_batch_shape" not in hmm.__dict__
    lifted = jax.tree.map(lambda a: a[None], expanded)
    assert lifted.batch_shape == (1, 3)


def test_gaussian_hmm_invalid_arguments():
    n, m, T = 2, 1, 3
    init = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    trans = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    obs = dist.Normal(jnp.zeros(m), 1.0).to_event(1)
    with pytest.raises(ValueError, match="num_steps"):
        GaussianHMM(init, jnp.eye(n), trans, jnp.ones((m, n)), obs)
    with pytest.raises(ValueError, match="num_steps"):
        GaussianHMM(
            init,
            jnp.broadcast_to(jnp.eye(n), (T, n, n)),
            trans,
            jnp.ones((m, n)),
            obs,
            num_steps=T + 1,
        )
    with pytest.raises(ValueError, match="event_shape"):
        GaussianHMM(init, jnp.eye(n), obs, jnp.ones((m, n)), obs, num_steps=T)
    with pytest.raises(TypeError):
        GaussianHMM(
            dist.StudentT(3.0, jnp.zeros(n), 1.0).to_event(1),
            jnp.eye(n),
            trans,
            jnp.ones((m, n)),
            obs,
            num_steps=T,
        )


@pytest.mark.parametrize("diag", [False, True])
def test_gaussian_hmm_sample_shapes(diag):
    hmm = _hmm(random.key(0), 5, 3, 2, batch=(3,), diag=diag)
    x = hmm.sample(random.key(1), (4,))
    assert x.shape == (4, 3, 5, 2)
    assert hmm.sample_posterior(random.key(2), x, (6,)).shape == (6, 4, 3, 5, 3)
    assert hmm.has_rsample


def test_gaussian_hmm_sample_moments_match_dense():
    T, n, m = 4, 2, 1
    ks = random.split(random.key(0), 6)
    A = jnp.broadcast_to(0.7 * jnp.eye(n), (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        jnp.zeros((T, n)), covariance_matrix=_spd(ks[4], n, 0.5)
    )
    obs = dist.MultivariateNormal(
        jnp.zeros((T, m)), covariance_matrix=_spd(ks[5], m, 0.3)
    )
    hmm = GaussianHMM(init, A, trans, H, obs)
    mean, cov = dense_reference(init, A, trans, H, obs, T)
    nz = (T + 1) * n
    N = 20000
    x = hmm.sample(random.key(1), (N,)).reshape(N, -1)
    se = jnp.sqrt(jnp.diag(cov[nz:, nz:]) / N)
    assert (jnp.abs(x.mean(0) - mean[nz:]) < 5 * se).all()
    assert_allclose(jnp.cov(x.T), cov[nz:, nz:], atol=0.15 * jnp.abs(cov).max())

    x_obs = random.normal(random.key(2), (T, m))
    z = hmm.sample_posterior(random.key(3), x_obs, (N,)).reshape(N, -1)
    Szz, Sxx, Szx = cov[:nz, :nz], cov[nz:, nz:], cov[:nz, nz:]
    K = jnp.linalg.solve(Sxx, Szx.T).T
    post_mean = (mean[:nz] + K @ (x_obs.ravel() - mean[nz:]))[n:]
    post_cov = (Szz - K @ Szx.T)[n:, n:]
    se = jnp.sqrt(jnp.diag(post_cov) / N)
    assert (jnp.abs(z.mean(0) - post_mean) < 5 * se).all()
    assert_allclose(jnp.cov(z.T), post_cov, atol=0.1 * jnp.abs(post_cov).max())


def test_gaussian_hmm_matches_gaussian_state_space_moments():
    T, n = 6, 2
    A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
    Q = _spd(random.key(10), n, 0.4)
    z0 = jnp.array([1.0, -0.5])
    S0 = 0.3 * jnp.eye(n)
    R = 0.2 * jnp.eye(n)
    ssm = dist.GaussianStateSpace(T, A, covariance_matrix=Q, initial_value=z0)
    hmm = GaussianHMM(
        dist.MultivariateNormal(z0, covariance_matrix=S0),
        A,
        dist.MultivariateNormal(jnp.zeros(n), covariance_matrix=Q),
        jnp.eye(n),
        dist.MultivariateNormal(jnp.zeros(n), covariance_matrix=R),
        num_steps=T,
    )
    N = 40000
    x = hmm.sample(random.key(11), (N,))
    powers = [jnp.linalg.matrix_power(A, t) for t in range(1, T + 1)]
    extra = jnp.stack([jnp.diag(P @ S0 @ P.T) for P in powers]) + jnp.diag(R)
    assert_allclose(x.mean(0), ssm.mean, atol=0.05)
    assert_allclose(x.var(0), ssm.variance + extra, rtol=0.05, atol=0.02)


@pytest.mark.parametrize("other_kind", ["normal", "mvn", "expanded"])
def test_gaussian_hmm_conjugate_update_identity(other_kind):
    T, n, m = 5, 2, 2
    hmm = _hmm(random.key(0), T, n, m, batch=(3,))
    x = random.normal(random.key(1), (3, T, m))
    if other_kind == "normal":
        other = dist.Normal(x, 0.7).to_event(2)
    elif other_kind == "mvn":
        other = dist.MultivariateNormal(
            x, covariance_matrix=_spd(random.key(2), m)
        ).to_event(1)
    else:
        other = dist.Normal(0.0, 0.7).expand((3, T, m)).to_event(2)
    updated, log_normalizer = hmm.conjugate_update(other)
    assert isinstance(updated, GaussianHMM)
    assert log_normalizer.shape == (3,)
    assert updated.batch_shape == (3,)
    y = random.normal(random.key(3), (4, 3, T, m))
    assert_allclose(
        hmm.log_prob(y) + other.log_prob(y),
        updated.log_prob(y) + log_normalizer,
        rtol=1e-3,
        atol=1e-3,
    )
    assert updated.sample(random.key(4)).shape == (3, T, m)


def test_gaussian_hmm_prefix_condition_chain_rule():
    T, n, m, t = 9, 3, 2, 4
    ks = random.split(random.key(5), 6)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        jnp.zeros((T, n)), covariance_matrix=_spd(ks[4], n, 0.5)
    )
    obs = dist.MultivariateNormal(
        jnp.zeros((T, m)), covariance_matrix=_spd(ks[5], m, 0.3)
    )
    hmm = GaussianHMM(init, A, trans, H, obs)
    x = hmm.sample(random.key(6))
    head = GaussianHMM(
        init,
        A[:t],
        dist.MultivariateNormal(trans.mean[:t], scale_tril=trans.scale_tril[:t]),
        H[:t],
        dist.MultivariateNormal(obs.mean[:t], scale_tril=obs.scale_tril[:t]),
    )
    tail = hmm.prefix_condition(x[:t])
    assert tail.event_shape == (T - t, m)
    assert_allclose(
        hmm.log_prob(x),
        head.log_prob(x[:t]) + tail.log_prob(x[t:]),
        rtol=1e-4,
        atol=1e-4,
    )
    with pytest.raises(ValueError):
        hmm.prefix_condition(x)
    batched = hmm.prefix_condition(jnp.stack([x[:t], x[:t]]))
    assert batched.batch_shape == (2,)
    assert batched.log_prob(x[t:]).shape == (2,)


def test_gaussian_hmm_reshape_batch():
    hmm = _hmm(random.key(0), 4, 2, 1, batch=(3,), homogeneous=True)
    reshaped = hmm.reshape_batch((3, 1))
    assert reshaped.batch_shape == (3, 1)
    x = hmm.sample(random.key(1))
    assert_allclose(reshaped.log_prob(x[:, None]), hmm.log_prob(x)[:, None], rtol=1e-5)
    assert isinstance(reshaped.prefix_condition(x[:, None, :2]), GaussianHMM)


def test_independent_hmm():
    T, n, m = 5, 2, 3
    base = _hmm(random.key(0), T, n, 1, batch=(4, m))
    hmm = IndependentHMM(base)
    assert hmm.batch_shape == (4,)
    assert hmm.event_shape == (T, m)
    assert hmm.num_steps == T
    assert hmm.has_rsample
    x = hmm.sample(random.key(1), (2,))
    assert x.shape == (2, 4, T, m)
    assert hmm.log_prob(x).shape == (2, 4)
    assert_allclose(
        hmm.log_prob(x),
        base.log_prob(jnp.swapaxes(x, -1, -2)[..., None]).sum(-1),
        rtol=1e-5,
    )
    assert hmm.expand((6, 4)).batch_shape == (6, 4)
    tail = hmm.prefix_condition(x[0, :, :2])
    assert tail.batch_shape == (4,) and tail.event_shape == (T - 2, m)
    assert hmm.reshape_batch((4, 1)).batch_shape == (4, 1)
    assert hmm.support(x).shape == (2, 4)
