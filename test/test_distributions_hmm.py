# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import lax, random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms
from numpyro.distributions.hmm import (
    GammaGaussianHMM,
    GaussianHMM,
    GaussianMRF,
    HiddenMarkovModel,
    IndependentHMM,
    LinearHMM,
)
from numpyro.ops.gaussian import mvn_to_gaussian


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


def kalman_log_prob(x, m0, P0, A, Q, C, R):
    """Covariance-form Kalman filter log-likelihood for time-homogeneous parameters,
    in the dtype of the inputs."""
    from jax.scipy.linalg import cho_solve, solve_triangular

    obs_dim = x.shape[-1]

    def step(carry, x_t):
        m, P = carry
        m_pred = A @ m
        P_pred = A @ P @ A.T + Q
        S = C @ P_pred @ C.T + R
        L = jnp.linalg.cholesky(S)
        r = x_t - C @ m_pred
        u = solve_triangular(L, r, lower=True)
        ll = (
            -0.5 * u @ u
            - jnp.log(jnp.diagonal(L)).sum()
            - 0.5 * obs_dim * jnp.log(2 * jnp.pi)
        )
        K = cho_solve((L, True), C @ P_pred).T
        return (m_pred + K @ r, P_pred - K @ S @ K.T), ll

    return lax.scan(step, (m0, P0), x)[1].sum()


def _small_noise_model(T, obs_sd, dtype):
    """Two-state model with unit initial covariance, process noise ``0.1 * I`` and
    observation noise ``obs_sd``; data simulated in float64 with numpy."""
    import numpy as np

    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    Q = 0.1 * np.eye(2)
    C = np.array([[1.0, 0.5], [0.0, 1.0]])
    R = obs_sd**2 * np.eye(2)
    rng = np.random.default_rng(1)
    z = np.zeros(2)
    xs = []
    for _ in range(T):
        z = A @ z + rng.multivariate_normal(np.zeros(2), Q)
        xs.append(C @ z + rng.multivariate_normal(np.zeros(2), R))
    x = np.stack(xs)
    c = lambda a: jnp.asarray(a, dtype)  # noqa: E731
    params = (jnp.zeros(2, dtype), jnp.eye(2, dtype=dtype), c(A), c(Q), c(C), c(R))
    return c(x), params


@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("diag", [False, True])
def test_gaussian_hmm_shapes(batch, homogeneous, diag):
    T, n, m = 5, 3, 2
    hmm = _hmm(random.key(0), T, n, m, batch=batch, homogeneous=homogeneous, diag=diag)
    assert hmm.batch_shape == batch
    assert hmm.event_shape == (T, m)
    assert hmm.hidden_dim == n and hmm.obs_dim == m
    x = random.normal(random.key(1), (4,) + batch + (T, m))
    log_prob = jax.jit(hmm.log_prob)
    assert log_prob(x).shape == (4,) + batch
    assert log_prob(x[0]).shape == batch
    posterior = jax.jit(hmm.filter)(x)
    assert posterior.batch_shape == (4,) + batch
    assert posterior.event_shape == (n,)
    expanded = hmm.expand((7,) + batch)
    assert isinstance(expanded, GaussianHMM)
    assert jax.jit(expanded.log_prob)(x[0]).shape == (7,) + batch


def test_gaussian_hmm_expanded_components():
    T, n, m = 4, 2, 1
    init = dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)).expand((3,))
    trans = dist.Normal(0.0, 1.0).expand((3, T, n)).to_event(1)
    obs = dist.MultivariateNormal(jnp.zeros(m), jnp.eye(m)).expand((3, T))
    hmm = GaussianHMM(init, jnp.eye(n), trans, jnp.ones((m, n)), obs)
    assert hmm.batch_shape == (3,)
    assert hmm.event_shape == (T, m)
    plain = GaussianHMM(
        dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)),
        jnp.eye(n),
        dist.Normal(jnp.zeros((T, n)), 1.0).to_event(1),
        jnp.ones((m, n)),
        dist.MultivariateNormal(jnp.zeros((T, m)), jnp.eye(m)),
    )
    x = random.normal(random.key(2), (3, T, m))
    assert_allclose(hmm.log_prob(x), plain.log_prob(x), rtol=1e-5)


def test_gaussian_hmm_expand_broadcasts_log_prob():
    T, n, m = 4, 2, 1
    hmm = _hmm(random.key(0), T, n, m, batch=(3,))
    for shape in [(), (1,), (2,)]:
        with pytest.raises(ValueError, match="Cannot broadcast distribution"):
            hmm.expand(shape)
    expanded = hmm.expand((2, 3))
    assert expanded.batch_shape == (2, 3)
    x = random.normal(random.key(1), (3, T, m))
    assert_allclose(
        expanded.log_prob(x), jnp.broadcast_to(hmm.log_prob(x), (2, 3)), rtol=1e-5
    )


def test_gaussian_hmm_sample_shape_after_expand():
    hmm = _hmm(random.key(0), 5, 2, 1, batch=(3,), homogeneous=True)
    expanded = hmm.expand((2, 3))
    assert expanded.sample(random.key(1), (2,)).shape == (2, 2, 3, 5, 1)
    assert expanded.sample_posterior(
        random.key(2), expanded.sample(random.key(3)), (4,)
    ).shape == (4, 2, 3, 5, 2)


def test_gaussian_hmm_value_broadcasts_against_batch():
    T, m = 5, 1
    hmm = _hmm(random.key(0), T, 2, m, batch=(3,), homogeneous=True)
    x = random.normal(random.key(1), (1, T, m))
    assert_allclose(
        hmm.log_prob(x), hmm.log_prob(jnp.broadcast_to(x, (3, T, m))), rtol=1e-6
    )
    assert hmm.log_prob(random.normal(random.key(2), (4, 1, T, m))).shape == (4, 3)


def test_gaussian_hmm_latent_site_predictive_and_initialize_model():
    from numpyro.infer import Predictive
    from numpyro.infer.util import initialize_model

    T, n, m = 4, 2, 1

    def model():
        scale = numpyro.sample("scale", dist.LogNormal(0.0, 0.5))
        hmm = GaussianHMM(
            dist.Normal(jnp.zeros(n), 1.0).to_event(1),
            0.9 * jnp.eye(n),
            dist.Normal(jnp.zeros(n), scale).to_event(1),
            jnp.ones((m, n)),
            dist.Normal(jnp.zeros(m), 0.3).to_event(1),
            num_steps=T,
        )
        y = numpyro.sample("y", hmm)
        numpyro.sample("w", dist.Normal(y.sum(), 1.0), obs=1.0)

    samples = Predictive(model, num_samples=5)(random.key(0))
    assert samples["y"].shape == (5, T, m)
    param_info, potential_fn, *_ = initialize_model(random.key(1), model)
    assert param_info.z["y"].shape == (T, m)
    assert jnp.isfinite(param_info.potential_energy)


_T = 4
_LAYOUTS = [
    # (init, transition_matrix, transition_dist, observation_matrix, observation_dist)
    # batch-plus-time shape prefixes; time axis is (), (1,) or (T,); batch is () or (2,)
    ((), (), (), (), ()),
    ((), (_T,), (), (), ()),
    ((), (), (_T,), (), ()),
    ((), (), (), (_T,), ()),
    ((), (), (), (), (_T,)),
    ((), (1,), (_T,), (), (1,)),
    ((2,), (), (), (), ()),
    ((), (2, _T), (), (), ()),
    ((), (), (2, 1), (), (_T,)),
    ((2,), (_T,), (2, 1), (1,), (2, _T)),
    ((), (1,), (), (2, _T), (1,)),
    ((2,), (2, 1), (2, _T), (), ()),
]


def _layout_params(key, n, m, init_shape, A_shape, trans_shape, H_shape, obs_shape):
    ks = random.split(key, 8)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], A_shape + (n, n))
    H = random.normal(ks[1], H_shape + (m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], init_shape + (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        random.normal(ks[4], trans_shape + (n,)), covariance_matrix=_spd(ks[5], n, 0.5)
    )
    obs = dist.MultivariateNormal(
        random.normal(ks[6], obs_shape + (m,)), covariance_matrix=_spd(ks[7], m, 0.3)
    )
    return init, A, trans, H, obs


@pytest.mark.parametrize("layout", _LAYOUTS, ids=[str(s) for s in _LAYOUTS])
def test_gaussian_hmm_mixed_time_and_batch_layouts(layout):
    T, n, m = _T, 2, 1
    batch = (2,) if any(2 in shape for shape in layout) else ()
    init, A, trans, H, obs = _layout_params(random.key(3), n, m, *layout)
    hmm = GaussianHMM(init, A, trans, H, obs, num_steps=T)
    assert hmm.batch_shape == batch
    assert hmm.event_shape == (T, m)
    x = random.normal(random.key(4), (3,) + batch + (T, m))
    log_prob = jax.jit(hmm.log_prob)(x)
    assert log_prob.shape == (3,) + batch
    assert hmm.filter(x).batch_shape == (3,) + batch

    def full_mvn(d):
        loc = jnp.broadcast_to(d.mean, batch + (T,) + d.event_shape)
        return dist.MultivariateNormal(loc, covariance_matrix=d.covariance_matrix)

    full = GaussianHMM(
        dist.MultivariateNormal(
            jnp.broadcast_to(init.mean, batch + (n,)),
            covariance_matrix=init.covariance_matrix,
        ),
        jnp.broadcast_to(A, batch + (T, n, n)),
        full_mvn(trans),
        jnp.broadcast_to(H, batch + (T, m, n)),
        full_mvn(obs),
    )
    assert full.batch_shape == batch
    assert_allclose(log_prob, jax.jit(full.log_prob)(x), rtol=1e-4, atol=1e-4)


def test_hidden_markov_model_is_exported():
    assert dist.HiddenMarkovModel is HiddenMarkovModel
    assert isinstance(_hmm(random.key(0), 3, 2, 1), dist.HiddenMarkovModel)


@pytest.mark.parametrize(
    "T,n,m", [(1, 2, 1), (2, 3, 2), (7, 3, 2), (8, 1, 1), (5, 1, 3)]
)
def test_gaussian_hmm_log_prob_and_filter_match_dense(T, n, m):
    ks = random.split(random.key(T), 8)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        0.3 * random.normal(ks[4], (T, n)), covariance_matrix=_spd(ks[5], n, 0.5)
    )
    obs = dist.MultivariateNormal(
        0.3 * random.normal(ks[6], (T, m)), covariance_matrix=_spd(ks[7], m, 0.3)
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


@pytest.mark.parametrize("homogeneous", [False, True])
def test_gaussian_hmm_diag_matches_full_covariance(homogeneous):
    T, n, m = 6, 2, 2
    ks = random.split(random.key(21), 6)
    tshape = () if homogeneous else (T,)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], tshape + (n, n))
    H = random.normal(ks[1], tshape + (m, n))
    init_loc = random.normal(ks[2], (n,))
    init_scale = 0.5 + random.uniform(ks[3], (n,))
    trans_scale = 0.3 + random.uniform(ks[4], tshape + (n,))
    obs_scale = 0.2 + random.uniform(ks[5], tshape + (m,))
    diag = GaussianHMM(
        dist.Normal(init_loc, init_scale).to_event(1),
        A,
        dist.Normal(jnp.zeros(tshape + (n,)), trans_scale).to_event(1),
        H,
        dist.Normal(jnp.zeros(tshape + (m,)), obs_scale).to_event(1),
        num_steps=T,
    )
    full = GaussianHMM(
        dist.MultivariateNormal(init_loc, covariance_matrix=jnp.diag(init_scale**2)),
        A,
        dist.MultivariateNormal(
            jnp.zeros(tshape + (n,)),
            covariance_matrix=jnp.eye(n) * (trans_scale**2)[..., None, :],
        ),
        H,
        dist.MultivariateNormal(
            jnp.zeros(tshape + (m,)),
            covariance_matrix=jnp.eye(m) * (obs_scale**2)[..., None, :],
        ),
        num_steps=T,
    )
    x = full.sample(random.key(22), (3,))
    assert_allclose(diag.log_prob(x), full.log_prob(x), rtol=1e-4, atol=1e-4)
    assert_allclose(diag.filter(x).mean, full.filter(x).mean, rtol=1e-3, atol=1e-3)
    assert_allclose(
        diag.filter(x).covariance_matrix,
        full.filter(x).covariance_matrix,
        rtol=1e-3,
        atol=1e-3,
    )


def test_gaussian_hmm_rejects_wrong_time_axis():
    T, n, m = 5, 2, 1
    hmm = _hmm(random.key(0), T, n, m)
    x = random.normal(random.key(1), (T, m))
    with pytest.raises(ValueError, match="trailing shape"):
        hmm.log_prob(x[..., :1, :])
    with pytest.raises(ValueError, match="trailing shape"):
        hmm.filter(x[..., :1, :])


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
    carried, _ = lax.scan(lambda h, _: (h, None), expanded, None, length=2)
    assert carried.batch_shape == (3,)
    lifted = jax.tree.map(lambda a: a[None], expanded)
    assert lifted.batch_shape == (1, 3)


@pytest.mark.parametrize("independent", [False, True])
def test_hmm_pytree_round_trip(independent):
    T, n, m = 4, 2, 3
    hmm = _hmm(random.key(0), T, n, 1, batch=(m,))
    if independent:
        hmm = IndependentHMM(hmm)
    x = hmm.sample(random.key(1), (2,))
    leaves, treedef = jax.tree_util.tree_flatten(hmm)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(hmm)
    assert rebuilt.batch_shape == hmm.batch_shape
    assert rebuilt.event_shape == hmm.event_shape
    assert rebuilt.num_steps == T
    assert_allclose(rebuilt.log_prob(x), hmm.log_prob(x), rtol=1e-6)


def test_hmm_validate_args():
    T, n, m = 4, 2, 1
    init, A, trans, H, obs = _layout_params(random.key(2), n, m, (), (T,), (), (), ())
    hmm = GaussianHMM(init, A, trans, H, obs, validate_args=True)
    x = random.normal(random.key(3), (3, T, m))
    assert_allclose(
        hmm.log_prob(x), GaussianHMM(init, A, trans, H, obs).log_prob(x), rtol=1e-6
    )
    with pytest.warns(UserWarning, match="Out-of-support"):
        log_prob = hmm.log_prob(x.at[0, 0, 0].set(jnp.inf))
    assert log_prob[0] == -jnp.inf and jnp.isfinite(log_prob[1:]).all()
    posterior = hmm.filter(x)
    assert posterior.batch_shape == (3,)
    with pytest.warns(UserWarning, match="Out-of-support"):
        posterior.log_prob(jnp.full((3, n), jnp.nan))
    with pytest.warns(UserWarning, match="Out-of-support"):
        hmm.expand((3,)).log_prob(x.at[0, 0, 0].set(jnp.nan))
    with pytest.warns(UserWarning, match="Out-of-support"):
        hmm.prefix_condition(x[:, :2]).log_prob(x[:, 2:].at[0, 0, 0].set(jnp.nan))
    independent = IndependentHMM(
        GaussianHMM(init, A, trans, H, obs).expand((m,)), validate_args=True
    )
    with pytest.warns(UserWarning, match="Out-of-support"):
        assert independent.log_prob(x.at[0, 0, 0].set(jnp.nan))[0] == -jnp.inf
    with pytest.warns(UserWarning, match="Out-of-support"):
        independent.prefix_condition(x[:, :2]).log_prob(
            x[:, 2:].at[0, 0, 0].set(jnp.nan)
        )
    with pytest.warns(UserWarning, match="Out-of-support"):
        independent.expand((3,)).log_prob(x.at[0, 0, 0].set(jnp.nan))


def test_gaussian_hmm_invalid_arguments():
    n, m, T = 2, 1, 3
    init = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    trans = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    obs = dist.Normal(jnp.zeros(m), 1.0).to_event(1)
    A, H = jnp.eye(n), jnp.ones((m, n))
    with pytest.raises(ValueError, match="num_steps"):
        GaussianHMM(init, A, trans, H, obs)
    with pytest.raises(ValueError, match="num_steps"):
        GaussianHMM(
            init, jnp.broadcast_to(A, (T, n, n)), trans, H, obs, num_steps=T + 1
        )
    with pytest.raises(ValueError, match="event_shape"):
        GaussianHMM(init, A, obs, H, obs, num_steps=T)
    with pytest.raises(TypeError, match=r"got Independent\(StudentT\)"):
        GaussianHMM(
            dist.StudentT(3.0, jnp.zeros(n), 1.0).to_event(1),
            A,
            trans,
            H,
            obs,
            num_steps=T,
        )
    with pytest.raises(ValueError, match="positive"):
        GaussianHMM(init, A, trans, H, obs, num_steps=0)
    for num_steps in [5.5, 5.0]:
        with pytest.raises(TypeError):
            GaussianHMM(init, A, trans, H, obs, num_steps=num_steps)
    with pytest.raises(ValueError, match=r"time axis sizes are \[1, 3, 1, 2, 1\]"):
        GaussianHMM(
            init, jnp.broadcast_to(A, (T, n, n)), trans, jnp.ones((2, m, n)), obs
        )


@pytest.mark.parametrize("diag", [False, True])
def test_gaussian_hmm_sample_shapes(diag):
    hmm = _hmm(random.key(0), 5, 3, 2, batch=(3,), diag=diag)
    x = hmm.sample(random.key(1), (4,))
    assert x.shape == (4, 3, 5, 2)
    assert hmm.sample_posterior(random.key(2), x, (6,)).shape == (6, 4, 3, 5, 3)


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
    # Standard error of a sample covariance entry is about
    # sqrt(2 / N) * sigma_i * sigma_j; use 5 standard errors.
    se_cov = (
        5
        * jnp.sqrt(2.0 / N)
        * jnp.sqrt(jnp.outer(jnp.diag(cov[nz:, nz:]), jnp.diag(cov[nz:, nz:])))
    )
    assert (jnp.abs(jnp.cov(x.T) - cov[nz:, nz:]) < se_cov).all()

    x_obs = random.normal(random.key(2), (T, m))
    z = hmm.sample_posterior(random.key(3), x_obs, (N,)).reshape(N, -1)
    Szz, Sxx, Szx = cov[:nz, :nz], cov[nz:, nz:], cov[:nz, nz:]
    K = jnp.linalg.solve(Sxx, Szx.T).T
    post_mean = (mean[:nz] + K @ (x_obs.ravel() - mean[nz:]))[n:]
    post_cov = (Szz - K @ Szx.T)[n:, n:]
    se = jnp.sqrt(jnp.diag(post_cov) / N)
    assert (jnp.abs(z.mean(0) - post_mean) < 5 * se).all()
    se_post = (
        5
        * jnp.sqrt(2.0 / N)
        * jnp.sqrt(jnp.outer(jnp.diag(post_cov), jnp.diag(post_cov)))
    )
    assert (jnp.abs(jnp.cov(z.T) - post_cov) < se_post).all()


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
    total_var = ssm.variance + extra
    assert (jnp.abs(x.mean(0) - ssm.mean) < 5 * jnp.sqrt(total_var / N)).all()
    assert (jnp.abs(x.var(0) - total_var) < 5 * jnp.sqrt(2.0 / N) * total_var).all()


@pytest.mark.parametrize(
    "other_kind", ["normal", "mvn", "expanded", "expanded_independent"]
)
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
    elif other_kind == "expanded":
        other = dist.Normal(0.0, 0.7).expand((3, T, m)).to_event(2)
    else:
        other = dist.Normal(x[0], 0.7).to_event(2).expand((3,))
    updated, log_normalizer = hmm.conjugate_update(other)
    assert isinstance(updated, GaussianHMM)
    assert log_normalizer.shape == (3,)
    assert updated.batch_shape == (3,)
    y = random.normal(random.key(3), (4, 3, T, m))
    assert_allclose(
        hmm.log_prob(y) + other.log_prob(y),
        updated.log_prob(y) + log_normalizer,
        rtol=1e-4,
        atol=1e-3,
    )
    assert updated.sample(random.key(4)).shape == (3, T, m)


def test_gaussian_hmm_sample_after_conjugate_update_matches_closed_form():
    T, n, m = 2, 1, 1
    init, A, trans, H, obs = _layout_params(
        random.key(0), n, m, (), (T,), (T,), (T,), (T,)
    )
    hmm = GaussianHMM(init, A, trans, H, obs)
    x0 = random.normal(random.key(1), (T, m))
    sd = 0.7
    updated, _ = hmm.conjugate_update(dist.Normal(x0, sd).to_event(2))
    mean, cov = dense_reference(init, A, trans, H, obs, T)
    nz = (T + 1) * n
    Sigma, mu = cov[nz:, nz:], mean[nz:]
    cov_upd = jnp.linalg.inv(jnp.linalg.inv(Sigma) + jnp.eye(T * m) / sd**2)
    mean_upd = cov_upd @ (jnp.linalg.solve(Sigma, mu) + x0.ravel() / sd**2)
    N = 20000
    x = updated.sample(random.key(2), (N,)).reshape(N, T * m)
    se = jnp.sqrt(jnp.diag(cov_upd) / N)
    assert (jnp.abs(x.mean(0) - mean_upd) < 5 * se).all()
    d = jnp.diag(cov_upd)
    cov_se = jnp.sqrt(2.0 / N) * jnp.sqrt(jnp.outer(d, d))
    assert (jnp.abs(jnp.cov(x.T) - cov_upd) < 5 * cov_se).all()


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


def test_gaussian_hmm_homogeneous_prefix_condition_chain_rule():
    T, n, m, t = 6, 2, 1, 2
    hmm = _hmm(random.key(5), T, n, m, homogeneous=True)
    head = _hmm(random.key(5), t, n, m, homogeneous=True)
    x = hmm.sample(random.key(6), (2,))
    tail = hmm.prefix_condition(x[:, :t])
    assert isinstance(tail, GaussianHMM)
    assert tail.batch_shape == (2,) and tail.event_shape == (T - t, m)
    assert_allclose(
        hmm.log_prob(x),
        head.log_prob(x[:, :t]) + tail.log_prob(x[:, t:]),
        rtol=1e-4,
        atol=1e-4,
    )
    assert_allclose(
        tail.prefix_condition(x[:, t : t + 1]).log_prob(x[:, t + 1 :]),
        hmm.prefix_condition(x[:, : t + 1]).log_prob(x[:, t + 1 :]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_hidden_markov_model_base_is_abstract():
    hmm = _hmm(random.key(0), 3, 2, 1, homogeneous=True)
    base = HiddenMarkovModel(hmm._init, hmm._trans, hmm._obs, hmm.num_steps)
    with pytest.raises(NotImplementedError, match="subclass"):
        base.log_prob(jnp.zeros((3, 1)))
    with pytest.raises(NotImplementedError, match="subclass"):
        base.sample(random.key(0))


def test_prefix_condition_initial_factor_is_normalized_posterior():
    T, n, m, t = 6, 2, 1, 3
    hmm = _hmm(random.key(1), T, n, m, homogeneous=True)
    head = _hmm(random.key(1), t, n, m, homogeneous=True)
    x = hmm.sample(random.key(2))
    tail = hmm.prefix_condition(x[:t])
    assert_allclose(tail._init.event_logsumexp(), 0.0, atol=1e-5)
    # Information-form posterior vs the moment path; float32 agreement measured at
    # 6e-8 (precision) and 3e-8 (info_vec).
    expected = mvn_to_gaussian(head.filter(x[:t]))
    assert_allclose(tail._init.precision, expected.precision, rtol=1e-4, atol=1e-4)
    assert_allclose(tail._init.info_vec, expected.info_vec, rtol=1e-4, atol=1e-4)


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
    x = hmm.sample(random.key(1), (2,))
    assert x.shape == (2, 4, T, m)
    assert hmm.log_prob(x).shape == (2, 4)
    assert jnp.isfinite(hmm.log_prob(x)).all()
    assert hmm.expand((6, 4)).batch_shape == (6, 4)
    tail = hmm.prefix_condition(x[0, :, :2])
    assert tail.batch_shape == (4,) and tail.event_shape == (T - 2, m)
    assert hmm.reshape_batch((4, 1)).batch_shape == (4, 1)
    assert hmm.support(x).shape == (2, 4)
    plain = IndependentHMM(dist.Normal(jnp.zeros((4, 3, T, 1)), 1.0).to_event(2))
    with pytest.raises(TypeError):
        plain.prefix_condition(x[0, :, :2])
    with pytest.raises(TypeError):
        plain.reshape_batch((4, 1))
    with pytest.raises(ValueError):
        IndependentHMM(dist.Normal(jnp.zeros((4, 3, T, 1)), 1.0).to_event(1))
    with pytest.raises(ValueError, match="Cannot broadcast distribution"):
        hmm.expand((3,))


def test_independent_hmm_matches_block_diagonal_gaussian_hmm():
    T, n, m = 4, 2, 3
    ks = random.split(random.key(7), 7)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (m, T, n, n))
    H = random.normal(ks[1], (m, T, 1, n))
    init_loc = random.normal(ks[2], (m, n))
    init_cov = jax.vmap(lambda k: _spd(k, n))(random.split(ks[3], m))
    trans_loc = random.normal(ks[4], (m, T, n))
    trans_cov = jax.vmap(lambda k: _spd(k, n, 0.5))(random.split(ks[5], m))
    obs_loc = random.normal(ks[6], (m, T, 1))
    obs_scale = jnp.array([0.3, 0.5, 0.7])
    base = GaussianHMM(
        dist.MultivariateNormal(init_loc, covariance_matrix=init_cov),
        A,
        dist.MultivariateNormal(trans_loc, covariance_matrix=trans_cov[:, None]),
        H,
        dist.Normal(obs_loc, obs_scale[:, None, None]).to_event(1),
    )
    hmm = IndependentHMM(base)
    block_diag = jax.vmap(lambda *blocks: jax.scipy.linalg.block_diag(*blocks))
    full = GaussianHMM(
        dist.MultivariateNormal(
            init_loc.reshape(-1),
            covariance_matrix=jax.scipy.linalg.block_diag(*init_cov),
        ),
        block_diag(*A),
        dist.MultivariateNormal(
            jnp.moveaxis(trans_loc, 0, 1).reshape(T, m * n),
            covariance_matrix=jax.scipy.linalg.block_diag(*trans_cov),
        ),
        block_diag(*H),
        dist.Normal(obs_loc[..., 0].T, obs_scale).to_event(1),
    )
    assert full.batch_shape == () and full.event_shape == (T, m)
    assert hmm.batch_shape == () and hmm.event_shape == (T, m)
    x = hmm.sample(random.key(8), (3,))
    assert_allclose(hmm.log_prob(x), full.log_prob(x), rtol=1e-4, atol=1e-4)
    assert_allclose(
        hmm.prefix_condition(x[:, :2]).log_prob(x[:, 2:]),
        full.prefix_condition(x[:, :2]).log_prob(x[:, 2:]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_gaussian_hmm_log_prob_grad_matches_finite_differences():
    from jax.test_util import check_grads

    T, n, m = 5, 2, 1
    x = _hmm(random.key(0), T, n, m, homogeneous=True).sample(random.key(1))
    init = dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n))

    def log_prob_of(transition_matrix, noise_scale):
        return GaussianHMM(
            init,
            transition_matrix,
            dist.Normal(jnp.zeros(n), noise_scale).to_event(1),
            jnp.ones((m, n)),
            dist.Normal(jnp.zeros(m), 0.3).to_event(1),
            num_steps=T,
        ).log_prob(x)

    # Along the direction check_grads draws, the central-difference truncation
    # error is 1.9e-2 at eps=1e-2 (the same in float64, so it is not a gradient
    # error) and 5.3e-4 at eps=1e-3 in float32 (1.9e-4 in float64); float32
    # rounding noise at eps=1e-3 is at most 5.6e-3 over four random directions.
    # The float32 analytic gradient agrees with float64 on the same data to 1e-5.
    check_grads(
        log_prob_of,
        (0.9 * jnp.eye(n), jnp.float32(0.5)),
        order=1,
        modes=["rev"],
        eps=1e-3,
        rtol=1e-2,
        atol=1e-2,
    )
    lp = log_prob_of(0.9 * jnp.eye(n), jnp.float32(0.5))
    assert lp.dtype == jnp.float32


def test_gaussian_hmm_sample_is_reparameterized():
    T, n, m = 4, 2, 1
    init = dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n))

    def sample_mean(scale):
        hmm = GaussianHMM(
            init,
            0.9 * jnp.eye(n),
            dist.Normal(jnp.zeros(n), scale).to_event(1),
            jnp.ones((m, n)),
            dist.Normal(jnp.zeros(m), 0.3).to_event(1),
            num_steps=T,
        )
        return (hmm.sample(random.key(0), (64,)) ** 2).mean()

    grad = jax.grad(sample_mean)(jnp.float32(0.5))
    assert jnp.isfinite(grad) and grad > 0
    assert GaussianHMM(
        init,
        jnp.eye(n),
        dist.Normal(jnp.zeros(n), 1.0).to_event(1),
        jnp.ones((m, n)),
        dist.Normal(jnp.zeros(m), 1.0).to_event(1),
        num_steps=T,
    ).has_rsample


def test_gaussian_hmm_marginalizes_local_level_in_nuts():
    from numpyro.infer import MCMC, NUTS

    T = 60
    key = random.key(0)
    level = jnp.cumsum(0.3 * random.normal(key, (T,)))
    data = (level + 0.5 * random.normal(random.fold_in(key, 1), (T,)))[:, None]

    def model(data):
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        noise_scale = numpyro.sample("noise_scale", dist.LogNormal(-1.0, 1.0))
        hmm = GaussianHMM(
            dist.Normal(jnp.zeros(1), 5.0).to_event(1),
            jnp.eye(1),
            dist.Normal(jnp.zeros(1), drift_scale).to_event(1),
            jnp.eye(1),
            dist.Normal(jnp.zeros(1), noise_scale).to_event(1),
            num_steps=T,
        )
        numpyro.sample("obs", hmm, obs=data)

    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300, progress_bar=False)
    mcmc.run(random.key(1), data)
    samples = mcmc.get_samples()
    assert jnp.isfinite(samples["drift_scale"]).all()
    # Measured medians (drift_scale, noise_scale) with 300 warmup and 300
    # samples: key(1) 0.2705, 0.5060; key(2) 0.2634, 0.5141; key(3) 0.2581,
    # 0.5148. The data were simulated with drift 0.3 and noise 0.5.
    assert 0.2 < jnp.median(samples["drift_scale"]) < 0.45
    assert 0.35 < jnp.median(samples["noise_scale"]) < 0.7


def test_gaussian_hmm_x64_extreme_scales():
    if jnp.result_type(float) == jnp.float32:
        pytest.skip("extreme noise scales are tested with x64 only")
    T = 500
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    Q = 1e-6 * jnp.eye(2)
    C = jnp.array([[1.0, 0.0]])
    R = jnp.eye(1)
    m0, P0 = jnp.zeros(2), jnp.eye(2)

    def log_prob_of(transition_matrix, x):
        return GaussianHMM(
            dist.MultivariateNormal(m0, P0),
            transition_matrix,
            dist.MultivariateNormal(jnp.zeros(2), Q),
            C,
            dist.MultivariateNormal(jnp.zeros(1), R),
            num_steps=T,
        ).log_prob(x)

    x = GaussianHMM(
        dist.MultivariateNormal(m0, P0),
        A,
        dist.MultivariateNormal(jnp.zeros(2), Q),
        C,
        dist.MultivariateNormal(jnp.zeros(1), R),
        num_steps=T,
    ).sample(random.key(0))
    lp, grad = jax.value_and_grad(log_prob_of)(A, x)
    reference, reference_grad = jax.value_and_grad(
        lambda A_: kalman_log_prob(x, m0, P0, A_, Q, C, R)
    )(A)
    # Measured against the Kalman reference (which agrees with a 60-digit
    # Decimal Kalman filter to 4e-12): value error 3.3e-3, relative 4.5e-6;
    # gradient max absolute deviation 7.7e-2, max relative 8.8e-6. The
    # information form loses float64 accuracy on this unit-root model with an
    # error that scales like T**3 / Q; the tolerances are 3x the measurements.
    assert_allclose(lp, reference, rtol=1e-5)
    assert_allclose(grad, reference_grad, rtol=3e-5, atol=0.25)


def test_gaussian_hmm_x64_small_observation_noise_matches_kalman():
    if jnp.result_type(float) == jnp.float32:
        pytest.skip("float64 accuracy is tested with x64 only")
    T, obs_sd = 256, 0.001
    x, (m0, P0, A, Q, C, R) = _small_noise_model(T, obs_sd, jnp.float64)
    hmm = GaussianHMM(
        dist.MultivariateNormal(m0, P0),
        A,
        dist.MultivariateNormal(jnp.zeros(2), Q),
        C,
        dist.MultivariateNormal(jnp.zeros(2), R),
        num_steps=T,
    )
    # Measured relative error 2.6e-11 (3.6e-9 absolute on -136.26); 3.8x margin.
    assert_allclose(hmm.log_prob(x), kalman_log_prob(x, m0, P0, A, Q, C, R), rtol=1e-10)


def test_gaussian_hmm_float32_small_observation_noise_matches_kalman():
    # Measured before the retry-on-failure jitter: error 8.1e-2 at T=64,
    # obs_sd=0.01; after: 1.3e-2. A float32 covariance-form Kalman filter is
    # within 1.5e-5 of its float64 value on the same data.
    T, obs_sd = 64, 0.01
    x, (m0, P0, A, Q, C, R) = _small_noise_model(T, obs_sd, jnp.float32)
    hmm = GaussianHMM(
        dist.MultivariateNormal(m0, P0),
        A,
        dist.MultivariateNormal(jnp.zeros(2, jnp.float32), Q),
        C,
        dist.MultivariateNormal(jnp.zeros(2, jnp.float32), R),
        num_steps=T,
    )
    reference = kalman_log_prob(x, m0, P0, A, Q, C, R)
    assert abs(float(hmm.log_prob(x)) - float(reference)) < 3e-2


@pytest.mark.parametrize("layout", _LAYOUTS, ids=[str(s) for s in _LAYOUTS])
def test_gaussian_hmm_sequential_matches_parallel(layout):
    T, n, m = _T, 2, 1
    init, A, trans, H, obs = _layout_params(random.key(7), n, m, *layout)
    parallel = GaussianHMM(init, A, trans, H, obs, num_steps=T)
    sequential = GaussianHMM(init, A, trans, H, obs, num_steps=T, sequential=True)
    assert sequential.sequential and not parallel.sequential
    assert sequential.batch_shape == parallel.batch_shape
    x = random.normal(random.key(8), (3,) + parallel.batch_shape + (T, m))
    assert_allclose(sequential.log_prob(x), parallel.log_prob(x), rtol=1e-4, atol=1e-4)
    assert_allclose(
        jax.jit(sequential.log_prob)(x), parallel.log_prob(x), rtol=1e-4, atol=1e-4
    )
    a, b = sequential.filter(x), parallel.filter(x)
    assert_allclose(a.mean, b.mean, rtol=1e-3, atol=1e-3)
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-3, atol=1e-3)


def test_gaussian_hmm_sequential_float32_small_noise_is_accurate():
    # Measured: sequential float32 is 7.6e-6 nats from the float32 reference
    # and 5.4e-6 from float64; the parallel information form is off by 1.65
    # nats (review.md 3.1).
    T, obs_sd = 64, 0.001
    x, (m0, P0, A, Q, C, R) = _small_noise_model(T, obs_sd, jnp.float32)
    hmm = GaussianHMM(
        dist.MultivariateNormal(m0, P0),
        A,
        dist.MultivariateNormal(jnp.zeros(2, jnp.float32), Q),
        C,
        dist.MultivariateNormal(jnp.zeros(2, jnp.float32), R),
        num_steps=T,
        sequential=True,
    )
    reference = kalman_log_prob(x, m0, P0, A, Q, C, R)
    assert_allclose(hmm.log_prob(x), reference, rtol=1e-5, atol=1e-3)


def test_gaussian_hmm_sequential_filter_is_positive_definite_at_tiny_noise():
    # With the update cov_pred - K S K^T the float32 filtered covariance at
    # obs_sd=1e-4 had eigenvalues [-7.5e-9, 5.2e-8] (not positive definite);
    # with the Joseph form they are [6.1e-9, 1.6e-8].
    T, obs_sd = 64, 1e-4
    x, (m0, P0, A, Q, C, R) = _small_noise_model(T, obs_sd, jnp.float32)
    hmm = GaussianHMM(
        dist.MultivariateNormal(m0, P0),
        A,
        dist.MultivariateNormal(jnp.zeros(2, jnp.float32), Q),
        C,
        dist.MultivariateNormal(jnp.zeros(2, jnp.float32), R),
        num_steps=T,
        sequential=True,
        validate_args=True,
    )
    posterior = hmm.filter(x)
    assert jnp.isfinite(posterior.scale_tril).all()
    assert jnp.linalg.eigvalsh(posterior.covariance_matrix).min() > 0
    assert jnp.isfinite(hmm.prefix_condition(x[:32]).sample(random.key(0))).all()


def test_gaussian_hmm_sequential_derived_models():
    T, n, m, t = 6, 2, 1, 2
    seq = GaussianHMM(
        dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)),
        jnp.eye(n),
        dist.Normal(jnp.zeros(n), 0.5).to_event(1),
        jnp.ones((m, n)),
        dist.Normal(jnp.zeros(m), 0.3).to_event(1),
        num_steps=T,
        sequential=True,
    )
    x = seq.sample(random.key(10), (2,))
    assert seq.expand((4,)).sequential and seq.expand((4,)).log_prob(x[0]).shape == (4,)
    assert seq.reshape_batch((1,)).log_prob(x[:, None]).shape == (2, 1)
    tail = seq.prefix_condition(x[:, :t])
    assert tail.sequential and tail.batch_shape == (2,)
    head = GaussianHMM(
        dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)),
        jnp.eye(n),
        dist.Normal(jnp.zeros(n), 0.5).to_event(1),
        jnp.ones((m, n)),
        dist.Normal(jnp.zeros(m), 0.3).to_event(1),
        num_steps=t,
        sequential=True,
    )
    assert_allclose(
        seq.log_prob(x),
        head.log_prob(x[:, :t]) + tail.log_prob(x[:, t:]),
        rtol=1e-4,
        atol=1e-4,
    )
    with pytest.raises(NotImplementedError, match="sequential"):
        seq.conjugate_update(dist.Normal(x[0], 0.7).to_event(2))
    mapped = jax.vmap(
        lambda s: GaussianHMM(
            dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)),
            jnp.eye(n),
            dist.Normal(jnp.zeros(n), s).to_event(1),
            jnp.ones((m, n)),
            dist.Normal(jnp.zeros(m), 0.3).to_event(1),
            num_steps=T,
            sequential=True,
        )
    )(jnp.array([0.5, 1.0]))
    assert mapped.batch_shape == (2,) and mapped.log_prob(x[0]).shape == (2,)
    leaves, treedef = jax.tree_util.tree_flatten(seq)
    assert jax.tree_util.tree_unflatten(treedef, leaves).sequential


def test_gaussian_hmm_sample_requires_key():
    hmm = _hmm(random.key(0), 3, 2, 1, homogeneous=True)
    with pytest.raises(ValueError, match="PRNG key"):
        hmm.sample(None)
    with pytest.raises(ValueError, match="PRNG key"):
        IndependentHMM(hmm.expand((2,))).sample(None)


def test_gaussian_hmm_coerces_integer_matrices_to_float():
    import numpy as np

    n, m = 2, 1
    hmm = GaussianHMM(
        dist.Normal(jnp.zeros(n), 1.0).to_event(1),
        np.eye(n, dtype=np.int64),
        dist.Normal(jnp.zeros(n), 1.0).to_event(1),
        np.ones((m, n), dtype=np.int32),
        dist.Normal(jnp.zeros(m), 0.3).to_event(1),
        num_steps=3,
    )
    assert hmm._trans.matrix.dtype == jnp.result_type(float)
    assert hmm._obs.matrix.dtype == jnp.result_type(float)


def test_gaussian_hmm_reports_non_broadcastable_value_batch():
    hmm = _hmm(random.key(0), 3, 2, 1, batch=(3,), homogeneous=True)
    with pytest.raises(
        ValueError, match="does not broadcast with batch_shape \\(3,\\)"
    ):
        hmm.log_prob(jnp.zeros((2, 3, 1)))


def test_gamma_gaussian_hmm_x64_extreme_scales():
    if jnp.result_type(float) == jnp.float32:
        pytest.skip("extreme noise scales are tested with x64 only")
    T = 200
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    init = dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2))
    trans = dist.MultivariateNormal(jnp.zeros(2), 1e-6 * jnp.eye(2))
    H = jnp.array([[1.0, 0.0]])
    obs = dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1))
    x = GaussianHMM(init, A, trans, H, obs, num_steps=T).sample(random.key(0))

    def log_prob_of(transition_matrix):
        return GammaGaussianHMM(
            dist.Gamma(4.0, 3.0), init, transition_matrix, trans, H, obs, num_steps=T
        ).log_prob(x)

    lp, grad = jax.value_and_grad(log_prob_of)(A)
    assert jnp.isfinite(lp) and jnp.isfinite(grad).all()


def _gamma_hmm_components(key, T, n, m, *, batch=(), homogeneous=False):
    ks = random.split(key, 6)
    tshape = ((1,) if batch else ()) if homogeneous else (T,)
    scale_dist = dist.Gamma(jnp.full(batch, 3.0), 2.0)
    init = dist.MultivariateNormal(
        random.normal(ks[2], batch + (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        jnp.zeros(batch + tshape + (n,)), covariance_matrix=_spd(ks[4], n, 0.5)
    )
    obs = dist.MultivariateNormal(
        jnp.zeros(batch + tshape + (m,)), covariance_matrix=_spd(ks[5], m, 0.3)
    )
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], batch + tshape + (n, n))
    H = random.normal(ks[1], batch + tshape + (m, n))
    return scale_dist, init, A, trans, H, obs


@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_gamma_gaussian_hmm_shapes(batch, homogeneous):
    T, n, m = 5, 2, 2
    scale_dist, init, A, trans, H, obs = _gamma_hmm_components(
        random.key(0), T, n, m, batch=batch, homogeneous=homogeneous
    )
    tshape = ((1,) if batch else ()) if homogeneous else (T,)
    hmm = GammaGaussianHMM(
        scale_dist, init, A, trans, H, obs, num_steps=T if homogeneous else None
    )
    assert hmm.batch_shape == batch and hmm.event_shape == (T, m)
    assert not hmm.has_rsample
    x = random.normal(random.key(1), (4,) + batch + (T, m))
    assert hmm.log_prob(x).shape == (4,) + batch
    gamma, mvn = hmm.filter(x)
    assert gamma.batch_shape == (4,) + batch
    assert mvn.batch_shape == (4,) + batch and mvn.event_shape == (n,)
    assert hmm.expand((2,) + batch).log_prob(x[0]).shape == (2,) + batch
    with pytest.raises(TypeError):
        GammaGaussianHMM(
            scale_dist,
            init,
            A,
            dist.StudentT(3.0, jnp.zeros(batch + tshape + (n,)), 1.0).to_event(1),
            H,
            obs,
            num_steps=T,
        )


@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_gamma_gaussian_hmm_expand_matches_broadcast_log_prob(batch, homogeneous):
    T, n, m = 4, 2, 1
    components = _gamma_hmm_components(
        random.key(5), T, n, m, batch=batch, homogeneous=homogeneous
    )
    hmm = GammaGaussianHMM(*components, num_steps=T if homogeneous else None)
    expanded = hmm.expand((2,) + batch)
    assert expanded.batch_shape == (2,) + batch
    x = random.normal(random.key(1), (2,) + batch + (T, m))
    assert_allclose(expanded.log_prob(x), hmm.log_prob(x), rtol=1e-5)
    assert_allclose(
        expanded.log_prob(x[0]),
        jnp.broadcast_to(hmm.log_prob(x[0]), (2,) + batch),
        rtol=1e-5,
    )


def test_independent_hmm_gamma_gaussian_hmm_log_prob_matches_base():
    T, n, m, B = 4, 2, 3, 2
    scale_dist, init, A, trans, H, obs = _gamma_hmm_components(
        random.key(1), T, n, 1, batch=(B, m)
    )
    base = GammaGaussianHMM(scale_dist, init, A, trans, H, obs)
    hmm = IndependentHMM(base)
    assert hmm.batch_shape == (B,) and hmm.event_shape == (T, m)
    assert not hmm.has_rsample
    x = random.normal(random.key(3), (5, B, T, m))
    actual = hmm.log_prob(x)
    assert actual.shape == (5, B)
    expected = base.log_prob(jnp.swapaxes(x, -1, -2)[..., None]).sum(-1)
    assert_allclose(actual, expected, rtol=1e-6)
    assert hmm.expand((4, B)).log_prob(x[0]).shape == (4, B)


def test_gamma_gaussian_hmm_diag_matches_full_covariance():
    T, n, m = 6, 2, 2
    ks = random.split(random.key(21), 6)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init_loc = random.normal(ks[2], (n,))
    init_scale = 0.5 + random.uniform(ks[3], (n,))
    trans_scale = 0.3 + random.uniform(ks[4], (T, n))
    obs_scale = 0.2 + random.uniform(ks[5], (T, m))
    scale_dist = dist.Gamma(4.0, 3.0)
    diag = GammaGaussianHMM(
        scale_dist,
        dist.Normal(init_loc, init_scale).to_event(1),
        A,
        dist.Normal(jnp.zeros((T, n)), trans_scale).to_event(1),
        H,
        dist.Normal(jnp.zeros((T, m)), obs_scale).to_event(1),
    )
    full = GammaGaussianHMM(
        scale_dist,
        dist.MultivariateNormal(init_loc, covariance_matrix=jnp.diag(init_scale**2)),
        A,
        dist.MultivariateNormal(
            jnp.zeros((T, n)),
            covariance_matrix=jnp.eye(n) * (trans_scale**2)[..., None, :],
        ),
        H,
        dist.MultivariateNormal(
            jnp.zeros((T, m)),
            covariance_matrix=jnp.eye(m) * (obs_scale**2)[..., None, :],
        ),
    )
    x = random.normal(random.key(22), (3, T, m))
    assert_allclose(diag.log_prob(x), full.log_prob(x), rtol=1e-4, atol=1e-4)
    gamma_diag, mvn_diag = diag.filter(x)
    gamma_full, mvn_full = full.filter(x)
    assert_allclose(gamma_diag.rate, gamma_full.rate, rtol=1e-4)
    assert_allclose(mvn_diag.mean, mvn_full.mean, rtol=1e-3, atol=1e-3)
    assert_allclose(
        mvn_diag.covariance_matrix, mvn_full.covariance_matrix, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("T,n,m", [(1, 1, 1), (2, 2, 1), (5, 2, 2)])
def test_gamma_gaussian_hmm_log_prob_matches_student_t(T, n, m):
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
    concentration, rate = 4.0, 3.0
    hmm = GammaGaussianHMM(
        dist.Gamma(concentration, rate), init, A, trans, H, obs, num_steps=T
    )
    x = random.normal(random.key(1), (T, m))
    mean, cov = dense_reference(init, A, trans, H, obs, T)
    nz = (T + 1) * n
    cov_x = cov[nz:, nz:] * rate / concentration
    expected = dist.MultivariateStudentT(
        2 * concentration, mean[nz:], jnp.linalg.cholesky(cov_x)
    ).log_prob(x.ravel())
    assert_allclose(hmm.log_prob(x), expected, rtol=1e-4, atol=1e-4)


def test_gamma_gaussian_hmm_expanded_gamma_prior_matches_student_t():
    T, n, m, B = 3, 2, 1, 4
    ks = random.split(random.key(7), 6)
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
    concentration, rate = 4.0, 3.0
    scale_dist = dist.Gamma(concentration, rate).expand((B,))
    assert isinstance(scale_dist, dist.ExpandedDistribution)
    hmm = GammaGaussianHMM(
        scale_dist,
        init.expand((B,)),
        A,
        trans.expand((B, T)),
        H,
        obs.expand((B, T)),
        num_steps=T,
    )
    assert hmm.batch_shape == (B,)
    x = random.normal(random.key(1), (B, T, m))
    mean, cov = dense_reference(init, A, trans, H, obs, T)
    nz = (T + 1) * n
    student = dist.MultivariateStudentT(
        2 * concentration,
        mean[nz:],
        jnp.linalg.cholesky(cov[nz:, nz:] * rate / concentration),
    )
    expected = jnp.stack([student.log_prob(x[b].ravel()) for b in range(B)])
    assert_allclose(hmm.log_prob(x), expected, rtol=1e-4, atol=1e-4)
    with pytest.raises(TypeError, match="Gamma"):
        GammaGaussianHMM(
            dist.LogNormal(0.0, 1.0).expand((B,)), init, A, trans, H, obs, num_steps=T
        )


def test_gamma_gaussian_hmm_filter_matches_closed_form():
    T, n, m = 4, 2, 2
    ks = random.split(random.key(T), 6)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans_base = dist.MultivariateNormal(
        0.3 * random.normal(ks[4], (n,)), covariance_matrix=_spd(ks[5], n, 0.5)
    )
    trans = trans_base.expand((T,))
    obs = dist.MultivariateNormal(
        0.3 * random.normal(ks[0], (T, m)), covariance_matrix=_spd(ks[1], m, 0.3)
    )
    concentration, rate = 4.0, 3.0
    hmm = GammaGaussianHMM(
        dist.Gamma(concentration, rate), init, A, trans, H, obs, num_steps=T
    )
    x = random.normal(random.key(1), (T, m))
    gamma, mvn = hmm.filter(x)

    mean, cov = dense_reference(init, A, trans_base, H, obs, T)
    nz = (T + 1) * n
    cov_x = cov[nz:, nz:]
    d = x.ravel() - mean[nz:]
    assert_allclose(gamma.concentration, concentration + T * m / 2, rtol=1e-4)
    assert_allclose(gamma.rate, rate + 0.5 * d @ jnp.linalg.solve(cov_x, d), rtol=1e-4)

    reference = GaussianHMM(init, A, trans, H, obs, num_steps=T).filter(x)
    assert_allclose(mvn.mean, reference.mean, rtol=1e-3, atol=1e-3)
    assert_allclose(
        mvn.covariance_matrix, reference.covariance_matrix, rtol=1e-3, atol=1e-3
    )


def _linear_hmm(key, T, n, m, *, batch=(), obs_kind="student"):
    ks = random.split(key, 4)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], batch + (T, n, n))
    H = random.normal(ks[1], batch + (T, m, n))
    init = dist.StudentT(4.0, jnp.zeros(batch + (n,)), 1.0).to_event(1)
    trans = dist.StudentT(5.0, jnp.zeros(batch + (T, n)), 0.5).to_event(1)
    if obs_kind == "student":
        obs = dist.StudentT(6.0, jnp.zeros(batch + (T, m)), 0.3).to_event(1)
    elif obs_kind == "lognormal":
        obs = dist.LogNormal(jnp.zeros(batch + (T, m)), 0.3).to_event(1)
    else:
        obs = dist.LogNormal(0.0, 0.3).expand(batch + (T, m)).to_event(1)
    return LinearHMM(init, A, trans, H, obs)


@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("obs_kind", ["student", "lognormal", "expanded_lognormal"])
def test_linear_hmm_shapes(batch, obs_kind):
    T, n, m = 5, 2, 3
    hmm = _linear_hmm(random.key(0), T, n, m, batch=batch, obs_kind=obs_kind)
    assert hmm.batch_shape == batch and hmm.event_shape == (T, m)
    assert hmm.has_rsample
    x = hmm.sample(random.key(1), (4,))
    assert x.shape == (4,) + batch + (T, m)
    assert jnp.all(hmm.support(x))
    if obs_kind != "student":
        assert len(hmm.transforms) == 1 and jnp.all(x > 0)
    assert hmm.expand((2,) + batch).sample(random.key(2)).shape == (2,) + batch + (
        T,
        m,
    )
    assert jax.jit(lambda h, k: h.sample(k))(hmm, random.key(3)).shape == batch + (
        T,
        m,
    )
    with pytest.raises(NotImplementedError):
        hmm.log_prob(x)


def test_linear_hmm_with_normal_components_matches_gaussian_hmm_moments():
    T, n, m = 4, 2, 1
    A = jnp.broadcast_to(0.7 * jnp.eye(n), (T, n, n))
    H = jnp.broadcast_to(jnp.ones((m, n)), (T, m, n))
    init = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    trans = dist.Normal(jnp.zeros((T, n)), 0.5).to_event(1)
    obs = dist.Normal(jnp.zeros((T, m)), 0.3).to_event(1)
    linear = LinearHMM(init, A, trans, H, obs)
    gaussian = GaussianHMM(init, A, trans, H, obs)
    N = 20000
    a = linear.sample(random.key(0), (N,))
    b = gaussian.sample(random.key(1), (N,))
    assert_allclose(a.mean(0), b.mean(0), atol=0.05)
    assert_allclose(a.var(0), b.var(0), rtol=0.1)


def test_linear_hmm_single_step_marginal_matches_student_t():
    # With T = 1 and A = 0 the observation is H eps_1 + nu_1; with H = 1 and
    # negligible observation noise it follows the transition noise exactly.
    df, loc, scale = 3.0, 0.4, 0.7
    hmm = LinearHMM(
        dist.Normal(jnp.zeros(1), 1.0).to_event(1),
        jnp.zeros((1, 1, 1)),
        dist.StudentT(df, jnp.full((1, 1), loc), scale).to_event(1),
        jnp.ones((1, 1, 1)),
        dist.Normal(jnp.zeros((1, 1)), 1e-6).to_event(1),
        num_steps=1,
    )
    assert hmm.event_shape == (1, 1)
    x = hmm.sample(random.key(2), (50000,))
    q = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert_allclose(
        jnp.quantile(x[:, 0, 0], q), dist.StudentT(df, loc, scale).icdf(q), atol=0.03
    )


def test_linear_hmm_mvn_noise_with_lognormal_observation():
    T, n, m = 5, 2, 2
    ks = random.split(random.key(0), 6)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (T, n, n))
    H = random.normal(ks[1], (T, m, n))
    init = dist.MultivariateNormal(
        random.normal(ks[2], (n,)), covariance_matrix=_spd(ks[3], n)
    )
    trans = dist.MultivariateNormal(
        jnp.zeros((T, n)), covariance_matrix=_spd(ks[4], n, 0.5)
    )
    obs = dist.TransformedDistribution(
        dist.MultivariateNormal(
            jnp.zeros((T, m)), covariance_matrix=_spd(ks[5], m, 0.3)
        ),
        transforms.ExpTransform(),
    )
    hmm = LinearHMM(init, A, trans, H, obs)
    assert hmm.batch_shape == () and hmm.event_shape == (T, m)
    assert isinstance(hmm.observation_dist, dist.MultivariateNormal)
    assert len(hmm.transforms) == 1
    assert isinstance(hmm.transforms[0], transforms.ExpTransform)
    x = hmm.sample(random.key(1), (4,))
    assert x.shape == (4, T, m) and jnp.all(x > 0)
    assert hmm.support.event_dim == 2
    assert jnp.all(hmm.support(x)) and not jnp.any(hmm.support(-x))
    assert hmm.expand((3,)).sample(random.key(2)).shape == (3, T, m)


def test_linear_hmm_homogeneous_keeps_time_axis():
    T, n, m = 6, 2, 3
    ks = random.split(random.key(0), 2)
    A = 0.8 * jnp.eye(n) + 0.1 * random.normal(ks[0], (n, n))
    H = random.normal(ks[1], (m, n))
    init = dist.StudentT(4.0, jnp.zeros(n), 1.0).to_event(1)
    trans = dist.StudentT(5.0, jnp.zeros(n), 0.5).to_event(1)
    obs = dist.StudentT(6.0, jnp.zeros(m), 0.3).to_event(1)
    hmm = LinearHMM(init, A, trans, H, obs, num_steps=T)
    assert hmm.batch_shape == () and hmm.event_shape == (T, m)
    assert hmm.sample(random.key(1), (2,)).shape == (2, T, m)
    expanded = hmm.expand((5,))
    assert expanded.transition_dist.batch_shape == (5, 1)
    assert expanded.transition_matrix.shape == (5, 1, n, n)
    assert expanded.sample(random.key(2)).shape == (5, T, m)


def test_linear_hmm_vmap_reports_mapped_batch():
    T, n, m = 5, 2, 1
    A = jnp.broadcast_to(0.8 * jnp.eye(n), (T, n, n))
    H = jnp.ones((T, m, n))

    def make(scale):
        return LinearHMM(
            dist.Normal(jnp.zeros(n), scale).to_event(1),
            A,
            dist.StudentT(4.0, jnp.zeros((T, n)), scale).to_event(1),
            H,
            dist.Normal(jnp.zeros((T, m)), 0.3).to_event(1),
        )

    hmm = jax.vmap(make)(jnp.array([0.5, 1.0]))
    assert hmm.batch_shape == (2,) and hmm.event_shape == (T, m)
    assert hmm.sample(random.key(0)).shape == (2, T, m)
    assert hmm.sample(random.key(0), (3,)).shape == (3, 2, T, m)
    # a vmap-built instance must draw independent initial states per batch element
    x = hmm.sample(random.key(1), (2000,))
    assert x.shape == (2000, 2, T, m)
    z = x[..., 0, :]
    assert abs(jnp.corrcoef(z[:, 0, 0], z[:, 1, 0])[0, 1]) < 0.1


def test_linear_hmm_rejects_shape_changing_observation_transform():
    T, n, m = 5, 2, 2
    init = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    trans = dist.Normal(jnp.zeros((T, n)), 0.5).to_event(1)
    obs = dist.TransformedDistribution(
        dist.Normal(jnp.zeros((T, m - 1)), 0.3).to_event(1),
        transforms.StickBreakingTransform(),
    )
    assert obs.event_shape == (m,)
    with pytest.raises(ValueError, match="StickBreakingTransform"):
        LinearHMM(init, jnp.eye(n), trans, jnp.ones((m, n)), obs)


def _mrf(key, T, n, m, *, batch=()):
    ks = random.split(key, 6)
    init = dist.MultivariateNormal(
        random.normal(ks[0], batch + (n,)), covariance_matrix=_spd(ks[1], n)
    )
    trans = dist.MultivariateNormal(
        random.normal(ks[2], batch + (T, 2 * n)), covariance_matrix=_spd(ks[3], 2 * n)
    )
    obs = dist.MultivariateNormal(
        random.normal(ks[4], batch + (T, n + m)), covariance_matrix=_spd(ks[5], n + m)
    )
    return GaussianMRF(init, trans, obs, num_steps=T)


@pytest.mark.parametrize("batch", [(), (3,)])
def test_gaussian_mrf_shapes(batch):
    T, n, m = 4, 2, 1
    mrf = _mrf(random.key(0), T, n, m, batch=batch)
    assert mrf.batch_shape == batch and mrf.event_shape == (T, m)
    assert not mrf.has_rsample
    x = random.normal(random.key(1), (5,) + batch + (T, m))
    assert mrf.log_prob(x).shape == (5,) + batch
    assert mrf.expand((2,) + batch).log_prob(x[0]).shape == (2,) + batch
    homogeneous = (
        dist.MultivariateNormal(jnp.zeros(n), jnp.eye(n)),
        dist.MultivariateNormal(jnp.zeros(2 * n), jnp.eye(2 * n)),
        dist.MultivariateNormal(jnp.zeros(n + m), jnp.eye(n + m)),
    )
    with pytest.raises(ValueError, match="num_steps"):
        GaussianMRF(*homogeneous)
    assert GaussianMRF(*homogeneous, num_steps=T).log_prob(x[0]).shape == batch


@pytest.mark.parametrize("T,n,m", [(1, 1, 1), (2, 2, 1), (5, 2, 2)])
def test_gaussian_mrf_log_prob_matches_unrolled(T, n, m):
    mrf = _mrf(random.key(T), T, n, m)
    x = random.normal(random.key(1), (T, m))
    nz = (T + 1) * n
    total = nz + T * m
    joint = mrf._init.event_pad(right=total - n)
    for t in range(T):
        joint = joint + mrf._trans[..., t].event_pad(
            left=t * n, right=total - (t + 2) * n
        )
        placed = mrf._obs[..., t].event_pad(
            left=(t + 1) * n, right=total - (t + 2) * n - m
        )
        source = list(range(total))
        block = source[(t + 2) * n : (t + 2) * n + m]
        del source[(t + 2) * n : (t + 2) * n + m]
        source[nz + t * m : nz + t * m] = block
        joint = joint + placed.event_permute(jnp.array(source))
    log_joint = joint.condition(x.ravel()).event_logsumexp()
    log_hidden = joint.marginalize(right=T * m).event_logsumexp()
    assert_allclose(mrf.log_prob(x), log_joint - log_hidden, rtol=1e-4, atol=1e-4)


def test_gaussian_mrf_log_prob_batched_values_match_loop():
    T, n, m = 4, 2, 1
    mrf = _mrf(random.key(0), T, n, m, batch=(3,))
    x = random.normal(random.key(1), (5, 3, T, m))
    expected = jnp.stack([mrf.log_prob(x[i]) for i in range(5)])
    assert_allclose(mrf.log_prob(x), expected, rtol=1e-4, atol=1e-4)


def test_gaussian_mrf_block_diagonal_reduces_to_independent_observations():
    T, n, m = 4, 2, 2
    ks = random.split(random.key(0), 6)
    init = dist.MultivariateNormal(
        random.normal(ks[0], (n,)), covariance_matrix=_spd(ks[1], n)
    )
    trans = dist.MultivariateNormal(
        random.normal(ks[2], (T, 2 * n)), covariance_matrix=_spd(ks[3], 2 * n)
    )
    cov_obs = (
        jnp.zeros((T, n + m, n + m))
        .at[:, :n, :n]
        .set(_spd(ks[4], n))
        .at[:, n:, n:]
        .set(_spd(ks[5], m))
    )
    obs_loc = random.normal(random.fold_in(ks[5], 1), (T, n + m))
    obs = dist.MultivariateNormal(obs_loc, covariance_matrix=cov_obs)
    mrf = GaussianMRF(init, trans, obs)
    x = random.normal(random.key(1), (T, m))
    expected = (
        dist.MultivariateNormal(obs_loc[:, n:], covariance_matrix=cov_obs[:, n:, n:])
        .log_prob(x)
        .sum()
    )
    assert_allclose(mrf.log_prob(x), expected, rtol=1e-4, atol=1e-4)


def test_linear_hmm_error_messages():
    n, m = 2, 1
    init = dist.Normal(jnp.zeros(n), 1.0).to_event(1)
    obs = dist.Normal(jnp.zeros(m), 1.0).to_event(1)
    H = jnp.ones((m, n))
    with pytest.raises(TypeError, match="event_dim == 1"):
        dist.LinearHMM(init, jnp.eye(n), dist.Normal(0.0, 1.0), H, obs, num_steps=2)
    with pytest.raises(TypeError, match="reparameterized"):
        dist.LinearHMM(
            init, jnp.eye(n), dist.Poisson(jnp.ones(n)).to_event(1), H, obs, num_steps=2
        )
    assert (
        dist.LinearHMM(init, jnp.eye(n), init, H, obs, num_steps=2).has_rsample is True
    )
    with pytest.raises(TypeError, match="ExpandedDistribution\\(Normal\\)"):
        dist.GammaGaussianHMM(
            dist.Normal(1.0, 1.0).expand((2,)),
            init,
            jnp.eye(n),
            init,
            H,
            obs,
            num_steps=2,
        )
