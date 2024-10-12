# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import jax
import jax.numpy as jnp

from numpyro.contrib.stochastic_support.dcc import StochasticSupportInference
from numpyro.handlers import condition
from numpyro.infer import (
    SVI,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceGraph_ELBO,
    TraceMeanField_ELBO,
)
from numpyro.infer.autoguide import AutoNormal

SDVIResult = namedtuple("SDVIResult", ["guides", "slp_weights"])

VALID_ELBOS = (Trace_ELBO, TraceMeanField_ELBO, TraceEnum_ELBO, TraceGraph_ELBO)


class SDVI(StochasticSupportInference):
    """
    Implements the Support Decomposition Variational Inference (SDVI) algorithm for models with
    stochastic support from [1]. This implementation creates a separate guide for each SLP, trains
    the guides separately, and then combines the guides by weighting them proportional to their ELBO
    estimates.

    **References:**

    1. *Rethinking Variational Inference for Probabilistic Programs with Stochastic Support*,
       Tim Reichelt, Luke Ong, Tom Rainforth

    **Example:**

    .. code-block:: python

        def model():
            model1 = numpyro.sample("model1", dist.Bernoulli(0.5), infer={"branching": True})
            if model1 == 0:
                mean = numpyro.sample("a1", dist.Normal(0.0, 1.0))
            else:
                mean = numpyro.sample("a2", dist.Normal(1.0, 1.0))
            numpyro.sample("obs", dist.Normal(mean, 1.0), obs=0.2)

        sdvi = SDVI(model, numpyro.optim.Adam(step_size=0.001))
        sdvi_result = sdvi.run(random.PRNGKey(0))

    :param model: Python callable containing Pyro primitives :mod:`~numpyro.primitives`.
    :param optimizer:  An instance of :class:`~numpyro.optim._NumpyroOptim`, a
        ``jax.example_libraries.optimizers.Optimizer`` or an Optax
        ``GradientTransformation``. Gets passed to :class:`~numpyro.infer.SVI`.
    :param int svi_num_steps: Number of steps to run SVI for each SLP.
    :param int combine_elbo_particles: Number of particles to estimate ELBO for computing
        SLP weights.
    :param guide_init: A constructor for the guide. This should be a callable that returns a
        :class:`~numpyro.infer.autoguide.AutoGuide` instance. Defaults to
        :class:`~numpyro.infer.autoguide.AutoNormal`.
    :param loss: ELBO loss for SVI. Defaults to :class:`~numpyro.infer.Trace_ELBO`.
    :param bool svi_progress_bar: Whether to use a progress bar for SVI.
    :param int num_slp_samples: Number of samples to draw from the prior to discover the
        straight-line programs (SLPs).
    :param int max_slps: Maximum number of SLPs to discover. DCC will not run inference
        on more than `max_slps`.
    """

    def __init__(
        self,
        model,
        optimizer,
        svi_num_steps=1000,
        combine_elbo_particles=1000,
        guide_init=AutoNormal,
        loss=Trace_ELBO(),
        svi_progress_bar=False,
        num_slp_samples=1000,
        max_slps=124,
    ):
        self.guide_init = guide_init
        self.optimizer = optimizer
        self.svi_num_steps = svi_num_steps
        self.svi_progress_bar = svi_progress_bar

        if not isinstance(loss, VALID_ELBOS):
            err_str = ", ".join(x.__name__ for x in VALID_ELBOS)
            raise ValueError(f"loss must be an instance of: ({err_str})")
        self.loss = loss
        self.combine_elbo_particles = combine_elbo_particles

        super().__init__(model, num_slp_samples, max_slps)

    def _run_inference(self, rng_key, branching_trace, *args, **kwargs):
        """
        Run SVI on a given SLP defined by its branching trace.
        """
        slp_model = condition(self.model, branching_trace)
        guide = self.guide_init(slp_model)
        svi = SVI(slp_model, guide, self.optimizer, loss=self.loss)
        svi_result = svi.run(
            rng_key,
            self.svi_num_steps,
            *args,
            progress_bar=self.svi_progress_bar,
            **kwargs,
        )
        return guide, svi_result.params

    def _combine_inferences(self, rng_key, guides, branching_traces, *args, **kwargs):
        """Weight each SLP proportional to its estimated ELBO."""
        elbos = {}
        for bt, (guide, param_map) in guides.items():
            slp_model = condition(self.model, branching_traces[bt])
            elbos[bt] = -Trace_ELBO(num_particles=self.combine_elbo_particles).loss(
                rng_key, param_map, slp_model, guide, *args, **kwargs
            )

        normalizer = jax.scipy.special.logsumexp(jnp.array(list(elbos.values())))
        slp_weights = {k: jnp.exp(v - normalizer) for k, v in elbos.items()}
        return SDVIResult(guides, slp_weights)
