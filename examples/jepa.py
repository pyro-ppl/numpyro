# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Causal JEPA
============================

Joint Embedding Predictive Architectures (JEPA) predict the *representation*
of a masked target view from a context view, instead of reconstructing raw
observations.  This example asks a causal version of that question: what would
the target representation look like under an intervention ``do(T = t)``?

We simulate a confounded system with an unobserved confounder::

    z       ~ Normal(0, I)                     # latent confounder
    T       ~ Bernoulli(sigmoid(a + b^T z))   # confounded treatment assignment
    Y(t)    = beta^T z + tau * t + eps        # structural outcome, tau = ATE

The learner never sees ``(z, T, Y)`` as a low-dimensional table.  It only sees
two high-dimensional "views" that stand in for image patches::

    x_ctx     = tanh(W_ctx @ z) + noise                  # hides T
    x_tgt(t)  = tanh(W_tgt @ [z, t, Y(t)]) + noise      # target under do(T=t)

For every observation we draw a random intervention ``a ~ Bernoulli(1/2)``
and pair the context with the *counterfactual* target view ``x_tgt(a)``.  This
breaks the ``z -> T`` backdoor path in the prediction task: the predictor must
condition on the intervention token ``a`` rather than on the spurious
correlation between the context and the observed treatment.

The NumPyro model combines three ingredients:

1. **Bayesian structural causal model.**  Priors on ``a, b, beta, tau, sigma``,
   a Bernoulli likelihood for the observed treatment and a Gaussian likelihood
   for the observed outcome, with the latent confounder replaced by a learned
   proxy ``z_hat = enc_z(x_ctx)``.
2. **JEPA energy.**  A predictor maps ``[s_ctx, a]`` to the embedding of the
   target view.  The target branch is wrapped in ``stop_gradient`` (cheap
   stand-in for the EMA target encoder used by I-JEPA / V-JEPA).  The squared
   prediction error is added to the log density with ``numpyro.factor`` inside
   the data plate, so its scale is comparable to the summed likelihood::

       E_i(x, a) = || pred([s_ctx_i, a_i]) - sg(s_tgt_i) ||^2

   Maximizing the ELBO therefore minimizes the embedding prediction error.
3. **Anti-collapse regularizer.**  A VICReg-style variance hinge on the
   context and predicted embeddings (the target branch is stop-gradient)
   keeps the representation from collapsing to a constant.

The confounder proxy is amortized by the encoder; global causal parameters are
given an ``AutoNormal`` variational guide.  After SVI we apply the do-operator
(``numpyro.handlers.do``) to sample from ``p(Y, s_tgt | do(T=0))`` and
``p(Y, s_tgt | do(T=1))``, then report the posterior ATE and the
counterfactual embedding gap.

Usage::

    python causal_jepa.py --num-steps 3000 --device cpu

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
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.scipy.special import expit

import numpyro
from numpyro import handlers, optim
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


# ---------------------------------------------------------------------------
# Networks: a shared encoder (embedding + confounder proxy) and a predictor.
# ---------------------------------------------------------------------------


def encoder(dim_view, dim_emb, dim_z, hidden_dim):
    """Return a stax network mapping a view to ``(embedding, confounder proxy)``."""
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
    """Bayesian SCM + interventional JEPA energy.

    All terms live in one NumPyro trace, so the ELBO jointly trains the
    representation, the predictor and the posterior over causal parameters.
    ``t_obs`` / ``y_obs`` may be ``None`` for posterior predictive sampling.
    """
    N, D = x_ctx.shape
    assert x_tgt.shape == (N, D)
    assert do_t.shape == (N,)

    encode = numpyro.module(
        "encoder", encoder(D, dim_emb, dim_z, hidden_dim), (N, D)
    )
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
    """Simulate a confounded system and expose both counterfactual views."""
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

    # Randomized intervention: the training pair is (x_ctx, do_t, x_tgt(do_t)),
    # NOT (x_ctx, t, x_tgt(t)).  This is the causal part of the JEPA objective.
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
# Inference and counterfactual evaluation
# ---------------------------------------------------------------------------


def run_inference(model_kwargs, data, args, rng_key):
    """Fit the model with SVI and an AutoNormal guide over causal parameters."""
    guide = AutoNormal(model)
    adam = optim.Adam(args.learning_rate)
    svi = SVI(model, guide, adam, Trace_ELBO(), **model_kwargs)

    train = {k: data[k] for k in ("x_ctx", "x_tgt", "do_t", "t_obs", "y_obs")}
    start = time.time()
    svi_result = svi.run(
        rng_key,
        args.num_steps,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        **train,
    )
    print("\nSVI elapsed time: {:.2f} s.".format(time.time() - start))
    assert jnp.isfinite(svi_result.losses).all()
    return svi_result, guide


def posterior_samples(guide, params, data, model_kwargs, args, rng_key):
    """Draw posterior samples of ``(a, b, beta, tau, sigma)`` from the guide."""
    # AutoNormal ignores the extra kwargs, but forwarding them keeps a
    # cold-start prototype trace (which replays the model) valid.
    predictive = Predictive(guide, params=params, num_samples=args.num_posterior_samples)
    return predictive(rng_key, **data, **model_kwargs)


def interventional_predictive(
    posterior, params, data, do_value, model_kwargs, rng_key
):
    """Sample ``p(Y, s_pred, s_tgt | do(T = do_value))``.

    ``handlers.do`` fixes the treatment site to ``do_value`` and re-evaluates
    its descendants under the structural equations, which is exactly the
    counterfactual quantity of interest.
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
        return_sites=["y", "s_pred", "s_tgt"],
        condition_deterministic=True,
    )
    return predictive(rng_key, **inputs, **model_kwargs)


def print_results(tau, pred0, pred1, losses, true_tau):
    header = "=" * 24 + " Bayesian Causal JEPA " + "=" * 24
    print("\n" + header + "\n")
    tau_mean, tau_std = float(jnp.mean(tau)), float(jnp.std(tau))
    lo, hi = np.percentile(np.asarray(tau), [5.0, 95.0])
    y_do0 = jnp.mean(pred0["y"])
    y_do1 = jnp.mean(pred1["y"])
    ate_ppc = float(y_do1 - y_do0)
    emb_gap = float(
        jnp.mean(jnp.linalg.norm(pred1["s_pred"] - pred0["s_pred"], axis=-1))
    )
    jepa_loss = 0.5 * (
        float(jnp.mean(jnp.sum((pred0["s_pred"] - pred0["s_tgt"]) ** 2, axis=-1)))
        + float(jnp.mean(jnp.sum((pred1["s_pred"] - pred1["s_tgt"]) ** 2, axis=-1)))
    )
    print(
        "true ATE (tau)                       : {:.3f}".format(true_tau)
    )
    print(
        "posterior tau    mean / std / 90% CI : {:.3f} / {:.3f} / [{:.3f}, {:.3f}]".format(
            tau_mean, tau_std, lo, hi
        )
    )
    print(
        "E[Y | do(T=1)] - E[Y | do(T=0)]     : {:.3f}".format(ate_ppc)
    )
    print(
        "counterfactual embedding gap         : {:.3f}".format(emb_gap)
    )
    print(
        "JEPA prediction loss (do 0 / do 1)   : {:.4f} (avg)".format(jepa_loss)
    )
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

    rng_key = random.key(args.seed)
    rng_key, rng_key_data, rng_key_svi, rng_key_post, rng_key_do = random.split(
        rng_key, 5
    )
    data, true_tau = get_data(rng_key_data, args)

    svi_result, guide = run_inference(model_kwargs, data, args, rng_key_svi)
    params = svi_result.params
    train_inputs = {k: data[k] for k in ("x_ctx", "x_tgt", "do_t", "t_obs", "y_obs")}
    posterior = posterior_samples(
        guide, params, train_inputs, model_kwargs, args, rng_key_post
    )
    tau = posterior["tau"]

    key_do0, key_do1 = random.split(rng_key_do)
    pred0 = interventional_predictive(
        posterior, params, data, 0, model_kwargs, key_do0
    )
    pred1 = interventional_predictive(
        posterior, params, data, 1, model_kwargs, key_do1
    )
    print_results(tau, pred0, pred1, svi_result.losses, true_tau)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.22.0")
    parser = argparse.ArgumentParser(description="Bayesian Causal JEPA example")
    parser.add_argument("-n", "--num-steps", nargs="?", default=3000, type=int)
    parser.add_argument("--num-data", nargs="?", default=512, type=int)
    parser.add_argument(
        "--num-posterior-samples", nargs="?", default=400, type=int
    )
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
    numpyro.set_host_device_count(1)

    main(args)
