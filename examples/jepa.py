# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Causal JEPA
=============================

Joint Embedding Predictive Architectures (JEPA) predict the *representation*
of a target view from a context view instead of reconstructing raw
observations.  This example combines such a prediction objective with a
Bayesian structural causal model: one shared ``stax`` encoder feeds both a
JEPA predictor and an outcome regression, and ``handlers.do`` is used to query
interventional outcomes.

We simulate a confounded system::

    z       ~ Normal(0, I)                     # latent context state
    T       ~ Bernoulli(sigmoid(a + b^T z))   # treatment depends on z
    Y(t)    = beta^T z + tau * t + eps        # structural outcome, tau = ATE

The learner sees two high-dimensional "views" standing in for image patches::

    x_ctx     = tanh(W_ctx @ z) + 0.02 * noise
    x_tgt(t)  = tanh(W_tgt @ [z, t, Y(t)]) + 0.02 * noise

``z`` is not hidden from the learner: ``x_ctx`` is a 32-dimensional, almost
noiseless rendering of a 4-dimensional ``z``, so ``z`` can be recovered from
the context.  The ATE is therefore identified by ordinary covariate
adjustment: ``y`` is regressed on ``t`` and on a learned proxy
``z_hat = enc_z(x_ctx)``.  The JEPA head shares the encoder's first layer but
is an auxiliary objective; it does not affect the ATE estimate (setting
``--jepa-coeff 0 --var-coeff 0`` leaves ``tau`` essentially unchanged).

For every observation we draw ``do_t ~ Bernoulli(1/2)`` and pair the context
with ``x_tgt(do_t)``.  This randomisation gives every context examples of both
treatment arms (positivity) for the embedding predictor.  It does not remove a
backdoor path: the predictor already sees ``z`` through ``x_ctx``, and pairing
with the observed ``t`` instead still trains the model.

The NumPyro model combines three ingredients:

1. **Bayesian structural causal model.**  Priors on ``a, b, beta, tau, sigma``,
   a Bernoulli likelihood for the observed treatment and a Gaussian likelihood
   for the observed outcome, with ``z_hat`` as the adjustment covariate.
2. **JEPA energy.**  A predictor maps ``[s_ctx, do_t]`` to the embedding of the
   target view.  The target branch is wrapped in ``stop_gradient`` (cheap
   stand-in for the EMA target encoder used by I-JEPA / V-JEPA).  The squared
   prediction error is added to the log density with ``numpyro.factor`` inside
   the data plate, so its scale is comparable to the summed likelihood::

       E_i(x, a) = || pred([s_ctx_i, a_i]) - sg(s_tgt_i) ||^2

   Maximizing the ELBO therefore minimizes the embedding prediction error.
3. **Anti-collapse regularizer.**  A VICReg-style variance hinge on the
   context and predicted embeddings (the target branch is stop-gradient)
   keeps the representation from collapsing to a constant.

Global causal parameters are given an ``AutoNormal`` variational guide.  After
SVI, ``handlers.do`` replaces only the ``t`` sample site, so interventions
change ``y`` but not the embeddings, which are computed before that site from
the ``do_t`` and ``x_tgt`` inputs.  We report the posterior ATE and,
separately, an embedding gap computed once from the fitted parameters by
swapping the target views fed to the predictor.  The gap comes from those
inputs, not from ``do``.  The JEPA score is the prediction MSE divided by the
spread of the target embeddings, so 1.0 is no better than predicting the mean
and a collapsed representation does not look artificially good.

Usage::

    python examples/jepa.py --num-steps 3000 --device cpu

**References:**

    1. Y. LeCun (2022), "A Path Towards Autonomous Machine Intelligence".
    2. M. Assran et al. (2023), "Self-Supervised Learning from Images with a
       Joint-Embedding Predictive Architecture" (I-JEPA).
    3. A. Bardes et al. (2024), "Revisiting Feature Prediction for Learning
       Visual Representations from Video" (V-JEPA).
    4. A. Bardes et al. (2022), "VICReg: Variance-Invariance-Covariance
       Regularization for Self-Supervised Learning".
    5. J. Pearl (2009), "Causality: Models, Reasoning, and Inference", 2nd ed.
"""

import argparse
import os
import time

import numpy as np

from jax import lax, random
from jax.example_libraries import stax
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
from numpyro import handlers, optim
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

# ---------------------------------------------------------------------------
# Networks: a shared encoder (embedding + adjustment proxy) and a predictor.
# ---------------------------------------------------------------------------


def encoder(dim_view, dim_emb, dim_z, hidden_dim):
    """Return a stax network mapping a view to ``(embedding, adjustment proxy)``."""
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Tanh,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(dim_emb, W_init=stax.randn()),
            stax.Dense(dim_z, W_init=stax.randn()),
        ),
    )


def predictor(dim_emb, hidden_dim):
    """Return a stax network mapping ``[s_ctx, do_t]`` to a target embedding."""
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Tanh,
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Tanh,
        stax.Dense(dim_emb, W_init=stax.randn()),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def model(
    x_ctx,
    x_tgt,
    do_t,
    t_obs,
    y_obs,
    dim_z,
    dim_emb,
    hidden_dim,
    jepa_coeff,
    var_coeff,
):
    """Bayesian SCM with an auxiliary JEPA energy in one ELBO.

    ``t_obs`` / ``y_obs`` may be ``None`` for posterior predictive sampling.
    The JEPA term shares the encoder with ``z_hat`` but is not used to identify
    ``tau``; it is a self-supervised auxiliary objective.
    """
    N, D = x_ctx.shape
    assert x_tgt.shape == (N, D)
    assert do_t.shape == (N,)

    encode = numpyro.module("encoder", encoder(D, dim_emb, dim_z, hidden_dim), (N, D))
    predict = numpyro.module(
        "predictor", predictor(dim_emb, hidden_dim), (N, dim_emb + 1)
    )

    # --- JEPA: predict target embeddings, never reconstruct the views -------
    s_ctx, z_hat = encode(x_ctx)
    s_tgt, _ = encode(x_tgt)
    # Standard JEPA trick: the target encoder receives no gradient.  Real
    # I-JEPA / V-JEPA implementations use an EMA of the context encoder here.
    s_tgt = lax.stop_gradient(s_tgt)
    assert s_ctx.shape == s_tgt.shape == (N, dim_emb)
    assert z_hat.shape == (N, dim_z)

    s_pred = predict(jnp.concatenate([s_ctx, do_t[:, None]], axis=-1))
    assert s_pred.shape == (N, dim_emb)
    per_sample_loss = jnp.sum((s_pred - s_tgt) ** 2, axis=-1)

    # The target branch is stop-gradient, so the variance hinge is applied to
    # the two branches that do receive gradients (VICReg-style).  Otherwise the
    # encoder could trivially collapse all embeddings to a constant.
    std_ctx = jnp.sqrt(jnp.var(s_ctx, axis=0) + 1.0e-4)
    std_pred = jnp.sqrt(jnp.var(s_pred, axis=0) + 1.0e-4)
    collapse_loss = jnp.mean(jnp.maximum(0.0, 1.0 - std_ctx)) + jnp.mean(
        jnp.maximum(0.0, 1.0 - std_pred)
    )

    # Place the prediction energy inside the plate: the plate broadcasts the
    # per-observation log factors and sums them, just like the likelihoods.
    with numpyro.plate("data", N):
        numpyro.factor("jepa_energy", -jepa_coeff * per_sample_loss)
    # The hinge is a dataset-level term; rescale by N to keep it comparable
    # with the summed per-observation likelihood and JEPA energies.
    numpyro.factor("anti_collapse", -var_coeff * N * collapse_loss)

    numpyro.deterministic("s_pred", s_pred)
    numpyro.deterministic("s_tgt", s_tgt)

    # --- Bayesian structural causal model ----------------------------------
    a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([dim_z]).to_event(1))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([dim_z]).to_event(1))
    tau = numpyro.sample("tau", dist.Normal(0.0, 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    logits_t = a + jnp.dot(z_hat, b)
    with numpyro.plate("data", N):
        # When t_obs is None this samples T; ``do`` can override it later.
        t = numpyro.sample("t", dist.Bernoulli(logits=logits_t), obs=t_obs)
        numpyro.sample(
            "y", dist.Normal(jnp.dot(z_hat, beta) + tau * t, sigma), obs=y_obs
        )


# ---------------------------------------------------------------------------
# Synthetic interventional dataset
# ---------------------------------------------------------------------------


def get_data(rng_key, args):
    """Simulate a confounded system and render both potential-outcome views."""
    N, dz, dv = args.num_data, args.dim_z, args.dim_view
    keys = random.split(rng_key, 10)

    # Fixed, unknown to the learner, linear "renderers" for the two views.
    W_ctx = random.normal(keys[0], (dz, dv)) / jnp.sqrt(dz)
    W_tgt = random.normal(keys[1], (dz + 2, dv)) / jnp.sqrt(dz + 2)
    noise_ctx = 0.02 * random.normal(keys[2], (N, dv))
    noise_tgt = 0.02 * random.normal(keys[3], (N, dv))

    z = random.normal(keys[4], (N, dz))
    a_true, tau_true = 0.3, 1.5
    b_true = 0.8 * random.normal(keys[5], (dz,))
    beta_true = random.uniform(keys[6], (dz,), minval=-1.0, maxval=1.0)

    t = random.bernoulli(keys[7], expit(a_true + jnp.dot(z, b_true))).astype(
        jnp.float32
    )
    # Paired potential outcomes share the same exogenous noise, so the only
    # difference between Y(0) and Y(1) is the structural effect tau.
    eps = 0.1 * random.normal(keys[8], (N,))
    y0 = jnp.dot(z, beta_true) + eps
    y1 = jnp.dot(z, beta_true) + tau_true + eps
    y = jnp.where(t > 0, y1, y0)

    x_ctx = jnp.tanh(jnp.dot(z, W_ctx)) + noise_ctx
    feat0 = jnp.concatenate([z, jnp.zeros((N, 1)), y0[:, None]], axis=-1)
    feat1 = jnp.concatenate([z, jnp.ones((N, 1)), y1[:, None]], axis=-1)
    x_tgt0 = jnp.tanh(jnp.dot(feat0, W_tgt)) + noise_tgt
    x_tgt1 = jnp.tanh(jnp.dot(feat1, W_tgt)) + noise_tgt

    # Randomised pairing: each context sees both treatment arms, which gives
    # the embedding predictor positivity.  It is not a backdoor adjustment;
    # z is already visible through x_ctx, and pairing with the observed t
    # instead of do_t still trains the model.
    do_t = random.bernoulli(keys[9], 0.5, (N,)).astype(jnp.float32)
    x_tgt = jnp.where(do_t[:, None] > 0, x_tgt1, x_tgt0)

    assert x_ctx.shape == x_tgt0.shape == x_tgt1.shape == (N, dv)
    assert t.shape == y.shape == do_t.shape == (N,)

    data = dict(
        x_ctx=x_ctx,
        x_tgt=x_tgt,
        do_t=do_t,
        t_obs=t,
        y_obs=y,
        x_tgt0=x_tgt0,
        x_tgt1=x_tgt1,
    )
    return data, tau_true


# ---------------------------------------------------------------------------
# Inference and interventional evaluation
# ---------------------------------------------------------------------------


def run_inference(model_kwargs, train_inputs, args, rng_key):
    """Fit the model with SVI and an AutoNormal guide over causal parameters."""
    guide = AutoNormal(model)
    adam = optim.Adam(args.learning_rate)
    svi = SVI(model, guide, adam, Trace_ELBO(), **model_kwargs)

    start = time.time()
    svi_result = svi.run(
        rng_key,
        args.num_steps,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        **train_inputs,
    )
    print("\nSVI elapsed time: {:.2f} s.".format(time.time() - start))
    assert jnp.isfinite(svi_result.losses).all()
    return svi_result, guide


def posterior_samples(guide, params, train_inputs, model_kwargs, args, rng_key):
    """Draw posterior samples of ``(a, b, beta, tau, sigma)`` from the guide."""
    # AutoNormal ignores the extra kwargs, but forwarding them keeps a
    # cold-start prototype trace (which replays the model) valid.
    predictive = Predictive(
        guide, params=params, num_samples=args.num_posterior_samples
    )
    return predictive(rng_key, **train_inputs, **model_kwargs)


def interventional_predictive(posterior, params, data, do_value, model_kwargs, rng_key):
    """Sample ``p(Y | do(T = do_value))``.

    ``handlers.do`` only replaces the ``t`` sample site, so it changes ``y``
    but not the embeddings: those are computed before that site from the
    ``do_t`` and ``x_tgt`` arguments.  The embedding gap is computed
    separately in ``embedding_diagnostics`` by swapping those inputs.
    """
    N = data["x_ctx"].shape[0]
    do_t = jnp.full((N,), float(do_value))
    x_tgt = data["x_tgt1"] if do_value == 1 else data["x_tgt0"]
    inputs = dict(
        x_ctx=data["x_ctx"],
        x_tgt=x_tgt,
        do_t=do_t,
        t_obs=None,
        y_obs=None,
    )
    intervened = handlers.do(model, data={"t": do_t})
    predictive = Predictive(
        intervened,
        posterior,
        params=params,
        return_sites=["y"],
    )
    return predictive(rng_key, **inputs, **model_kwargs)


def embedding_diagnostics(params, data, model_kwargs):
    """Compute the embedding gap and a normalized JEPA score once from params.

    Both embeddings are deterministic functions of the fitted network weights
    and the inputs, so no posterior copies are needed.  The two passes differ
    only in ``do_t`` and ``x_tgt``; the gap comes from swapping those inputs,
    not from ``handlers.do``.  Dividing the prediction MSE by the spread of
    the target embeddings makes 1.0 equivalent to predicting the mean and
    prevents a collapsed representation from scoring artificially well.
    """
    N, D = data["x_ctx"].shape
    _, enc_apply = encoder(
        D, model_kwargs["dim_emb"], model_kwargs["dim_z"], model_kwargs["hidden_dim"]
    )
    _, pred_apply = predictor(model_kwargs["dim_emb"], model_kwargs["hidden_dim"])
    enc_params = params["encoder$params"]
    pred_params = params["predictor$params"]

    s_ctx, _ = enc_apply(enc_params, data["x_ctx"])
    s_tgt0, _ = enc_apply(enc_params, data["x_tgt0"])
    s_tgt1, _ = enc_apply(enc_params, data["x_tgt1"])
    s_pred0 = pred_apply(
        pred_params, jnp.concatenate([s_ctx, jnp.zeros((N, 1))], axis=-1)
    )
    s_pred1 = pred_apply(
        pred_params, jnp.concatenate([s_ctx, jnp.ones((N, 1))], axis=-1)
    )

    gap = float(jnp.mean(jnp.linalg.norm(s_pred1 - s_pred0, axis=-1)))
    mse = 0.5 * (
        float(jnp.mean(jnp.sum((s_pred0 - s_tgt0) ** 2, axis=-1)))
        + float(jnp.mean(jnp.sum((s_pred1 - s_tgt1) ** 2, axis=-1)))
    )
    spread = float(jnp.sum(jnp.var(s_tgt1, axis=0))) + 1.0e-8
    return gap, mse / spread


def print_results(tau, pred0, pred1, losses, true_tau, emb_gap, jepa_score):
    header = "=" * 24 + " Bayesian Causal JEPA " + "=" * 24
    print("\n" + header + "\n")
    tau_mean, tau_std = float(jnp.mean(tau)), float(jnp.std(tau))
    lo, hi = np.percentile(np.asarray(tau), [5.0, 95.0])
    y_do0 = jnp.mean(pred0["y"])
    y_do1 = jnp.mean(pred1["y"])
    ate_ppc = float(y_do1 - y_do0)
    print("true ATE (tau)                       : {:.3f}".format(true_tau))
    print(
        "posterior tau    mean / std / 90% CI : {:.3f} / {:.3f} / [{:.3f}, {:.3f}]".format(
            tau_mean, tau_std, lo, hi
        )
    )
    print("E[Y | do(T=1)] - E[Y | do(T=0)]     : {:.3f}".format(ate_ppc))
    print("embedding gap (swapped inputs)      : {:.3f}".format(emb_gap))
    print("JEPA score (norm.; 1.0 = mean pred)  : {:.4f}".format(jepa_score))
    print(
        "final / initial ELBO loss            : {:.2f} / {:.2f}".format(
            float(losses[-1]), float(losses[0])
        )
    )
    print("\n" + "=" * len(header))


def main(args):
    model_kwargs = dict(
        dim_z=args.dim_z,
        dim_emb=args.dim_emb,
        hidden_dim=args.hidden_dim,
        jepa_coeff=args.jepa_coeff,
        var_coeff=args.var_coeff,
    )

    rng_key_data, rng_key_svi, rng_key_post, rng_key_do = random.split(
        random.key(args.seed), 4
    )
    data, true_tau = get_data(rng_key_data, args)
    train_inputs = {k: data[k] for k in ("x_ctx", "x_tgt", "do_t", "t_obs", "y_obs")}

    svi_result, guide = run_inference(model_kwargs, train_inputs, args, rng_key_svi)
    params = svi_result.params
    posterior = posterior_samples(
        guide, params, train_inputs, model_kwargs, args, rng_key_post
    )
    tau = posterior["tau"]

    key_do0, key_do1 = random.split(rng_key_do)
    pred0 = interventional_predictive(posterior, params, data, 0, model_kwargs, key_do0)
    pred1 = interventional_predictive(posterior, params, data, 1, model_kwargs, key_do1)
    emb_gap, jepa_score = embedding_diagnostics(params, data, model_kwargs)
    print_results(tau, pred0, pred1, svi_result.losses, true_tau, emb_gap, jepa_score)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.22.0")
    parser = argparse.ArgumentParser(description="Bayesian Causal JEPA example")
    parser.add_argument("-n", "--num-steps", nargs="?", default=3000, type=int)
    parser.add_argument("--num-data", nargs="?", default=512, type=int)
    parser.add_argument("--num-posterior-samples", nargs="?", default=400, type=int)
    parser.add_argument("--dim-z", nargs="?", default=4, type=int)
    parser.add_argument("--dim-view", nargs="?", default=32, type=int)
    parser.add_argument("--dim-emb", nargs="?", default=16, type=int)
    parser.add_argument("--hidden-dim", nargs="?", default=64, type=int)
    parser.add_argument("--jepa-coeff", nargs="?", default=5.0, type=float)
    parser.add_argument("--var-coeff", nargs="?", default=1.0, type=float)
    parser.add_argument("-lr", "--learning-rate", nargs="?", default=1.0e-3, type=float)
    parser.add_argument("--seed", nargs="?", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
