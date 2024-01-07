# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple

import jax
import jax.numpy as jnp
from jax import random

from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import init_to_value, log_density

DCCResult = namedtuple("DCCResult", ["samples", "slp_weights"])


class DCC:
    """
    Implements the Divide, Conquer, and Combine (DCC) algorithm from [1].

    **References:**
    1. *Divide, Conquer, and Combine: a New Inference Strategy for Probabilistic Programs with Stochastic Support*,
       Yuan Zhou, Hongseok Yang, Yee Whye Teh, Tom Rainforth
    """

    def __init__(
        self,
        model,
        mcmc_kwargs,
        num_slp_samples=1000,
        max_slps=124,
    ):
        self.model = model
        self.mcmc_kwargs = mcmc_kwargs

        self.num_slp_samples = num_slp_samples
        self.max_slps = max_slps

    def _find_slps(self, rng_key, *args, **kwargs):
        """
        Discover the straight-line programs (SLPs) in the model by sampling from the prior.
        This implementation assumes that all branching is done via discrete sampling sites
        that are annotated with `infer={"branching": True}`.
        """
        branching_traces = {}
        for _ in range(self.num_slp_samples):
            rng_key, subkey = random.split(rng_key)
            tr = trace(seed(self.model, subkey)).get_trace(*args, **kwargs)
            btr = self._get_branching_trace(tr)
            btr_str = ",".join(str(x) for x in btr.values())
            if btr_str not in branching_traces:
                branching_traces[btr_str] = btr
                if len(branching_traces) >= self.max_slps:
                    break

        return branching_traces

    def _get_branching_trace(self, tr):
        """
        Extract the sites from the trace that are annotated with `infer={"branching": True}`.
        """
        branching_trace = OrderedDict()
        for site in tr.values():
            if site["type"] == "sample" and site["infer"].get("branching", False):
                # TODO: Assert that this is a discrete sampling site and univariate distribution.
                # It is essential that we convert the value to a Python int. If it remains
                # a JAX Array, then during JIT compilation it will be treated as an AbstractArray
                # and we are not able to branch based on this value.
                branching_trace[site["name"]] = int(site["value"])
        return branching_trace

    def _run_mcmc(self, rng_key, branching_trace, *args, **kwargs):
        """
        Run MCMC on the model conditioned on the given branching trace.
        """
        slp_model = condition(self.model, data=branching_trace)
        kernel = NUTS(slp_model)
        mcmc = MCMC(kernel, **self.mcmc_kwargs)
        mcmc.run(rng_key, *args, **kwargs)

        return mcmc.get_samples()

    def _combine_samples(self, rng_key, samples, branching_traces, *args, **kwargs):
        """
        Weight each SLP proportional to its estimated normalization constant.
        The normalization constants are estimated using importance sampling with
        the proposal centered on the MCMC samples.
        """

        def log_weight(rng_key, i, slp_model, slp_samples):
            trace = {k: v[i] for k, v in slp_samples.items()}
            guide = AutoNormal(
                slp_model,
                init_loc_fn=init_to_value(values=trace),
                init_scale=1.0,
            )
            rng_key, subkey = random.split(rng_key)
            guide_trace = seed(guide, subkey)(*args, **kwargs)
            guide_log_density, _ = log_density(guide, args, kwargs, guide_trace)
            model_log_density, _ = log_density(slp_model, args, kwargs, guide_trace)
            return model_log_density - guide_log_density

        log_weights = jax.vmap(log_weight, in_axes=(None, 0, None, None))

        log_Zs = {}
        for bt, slp_samples in samples.items():
            num_samples = slp_samples[next(iter(slp_samples))].shape[0]
            slp_model = condition(self.model, data=branching_traces[bt])
            lws = log_weights(rng_key, jnp.arange(num_samples), slp_model, slp_samples)
            log_Zs[bt] = jax.scipy.special.logsumexp(lws) - jnp.log(num_samples)

        normalizer = jax.scipy.special.logsumexp(jnp.array(list(log_Zs.values())))
        slp_weights = {k: jnp.exp(v - normalizer) for k, v in log_Zs.items()}
        return DCCResult(samples, slp_weights)

    def run(self, rng_key, *args, **kwargs):
        rng_key, subkey = random.split(rng_key)
        branching_traces = self._find_slps(subkey, *args, **kwargs)

        samples = dict()
        for key, bt in branching_traces.items():
            rng_key, subkey = random.split(rng_key)
            samples[key] = self._run_mcmc(subkey, bt, *args, **kwargs)

        rng_key, subkey = random.split(rng_key)
        return self._combine_samples(subkey, samples, branching_traces, *args, **kwargs)
