# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple

import jax
from jax import random
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import init_to_value, log_density

DCCResult = namedtuple("DCCResult", ["samples", "slp_weights"])


class StochasticSupportInference(ABC):
    """
    Base class for running inference in programs with stochastic support. Each subclass
    decomposes the input model into so called straight-line programs (SLPs) which are
    the different control-flow paths in the model. Inference is then run in each SLP
    separately and the results are combined to produce an overall posterior.

    .. note:: This implementation assumes that all stochastic branching is done based on the
       outcomes of discrete sampling sites that are annotated with ``infer={"branching": True}``.
       For example,

       .. code-block:: python

            def model():
                model1 = numpyro.sample("model1", dist.Bernoulli(0.5), infer={"branching": True})
                if model1 == 0:
                    mean = numpyro.sample("a1", dist.Normal(0.0, 1.0))
                else:
                    mean = numpyro.sample("a2", dist.Normal(1.0, 1.0))
                numpyro.sample("obs", dist.Normal(mean, 1.0), obs=0.2)

    :param model: Python callable containing Pyro primitives :mod:`~numpyro.primitives`.
        local inference. Defaults to :class:`~numpyro.infer.NUTS`.
    :param int num_slp_samples: Number of samples to draw from the prior to discover the
        straight-line programs (SLPs).
    :param int max_slps: Maximum number of SLPs to discover. DCC will not run inference
        on more than `max_slps`.
    """

    def __init__(self, model, num_slp_samples, max_slps):
        self.model = model
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
                if (
                    not isinstance(site["fn"], dist.Distribution)
                    or not site["fn"].support.is_discrete
                ):
                    raise RuntimeError(
                        "Branching is only supported for discrete sampling sites."
                    )
                # It is essential that we convert the value to a Python int. If it remains
                # a JAX Array, then during JIT compilation it will be treated as an AbstractArray
                # which means branching will raise in an error.
                # Reference: (https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-jit)
                branching_trace[site["name"]] = int(site["value"])
        return branching_trace

    @abstractmethod
    def _run_inference(self, rng_key, branching_trace, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _combine_inferences(
        self, rng_key, inferences, branching_traces, *args, **kwargs
    ):
        raise NotImplementedError

    def run(self, rng_key, *args, **kwargs):
        """
        Run inference on each SLP separately and combine the results.

        :param jax.random.PRNGKey rng_key: Random number generator key.
        :param args: Arguments to the model.
        :param kwargs: Keyword arguments to the model.
        """
        rng_key, subkey = random.split(rng_key)
        branching_traces = self._find_slps(subkey, *args, **kwargs)

        inferences = dict()
        for key, bt in branching_traces.items():
            rng_key, subkey = random.split(rng_key)
            inferences[key] = self._run_inference(subkey, bt, *args, **kwargs)

        rng_key, subkey = random.split(rng_key)
        return self._combine_inferences(
            subkey, inferences, branching_traces, *args, **kwargs
        )


class DCC(StochasticSupportInference):
    """
    Implements the Divide, Conquer, and Combine (DCC) algorithm for models with
    stochastic support from [1].

    **References:**

    1. *Divide, Conquer, and Combine: a New Inference Strategy for Probabilistic Programs with Stochastic Support*,
       Yuan Zhou, Hongseok Yang, Yee Whye Teh, Tom Rainforth

    **Example:**

    .. code-block:: python

        def model():
            model1 = numpyro.sample("model1", dist.Bernoulli(0.5), infer={"branching": True})
            if model1 == 0:
                mean = numpyro.sample("a1", dist.Normal(0.0, 1.0))
            else:
                mean = numpyro.sample("a2", dist.Normal(1.0, 1.0))
            numpyro.sample("obs", dist.Normal(mean, 1.0), obs=0.2)

        mcmc_kwargs = dict(
            num_warmup=500, num_samples=1000
        )
        dcc = DCC(model, mcmc_kwargs=mcmc_kwargs)
        dcc_result = dcc.run(random.PRNGKey(0))

    :param model: Python callable containing Pyro primitives :mod:`~numpyro.primitives`.
    :param dict mcmc_kwargs: Dictionary of arguments passed to :data:`~numpyro.infer.MCMC`.
    :param numpyro.infer.mcmc.MCMCKernel kernel_cls: MCMC kernel class that is used for
        local inference. Defaults to :class:`~numpyro.infer.NUTS`.
    :param int num_slp_samples: Number of samples to draw from the prior to discover the
        straight-line programs (SLPs).
    :param int max_slps: Maximum number of SLPs to discover. DCC will not run inference
        on more than `max_slps`.
    :param float proposal_scale: Scale parameter for the proposal distribution for
        estimating the normalization constant of an SLP.
    """

    def __init__(
        self,
        model,
        mcmc_kwargs,
        kernel_cls=NUTS,
        num_slp_samples=1000,
        max_slps=124,
        proposal_scale=1.0,
    ):
        self.kernel_cls = kernel_cls
        self.mcmc_kwargs = mcmc_kwargs

        self.proposal_scale = proposal_scale

        super().__init__(model, num_slp_samples, max_slps)

    def _run_inference(self, rng_key, branching_trace, *args, **kwargs):
        """
        Run MCMC on the model conditioned on the given branching trace.
        """
        slp_model = condition(self.model, data=branching_trace)
        kernel = self.kernel_cls(slp_model)
        mcmc = MCMC(kernel, **self.mcmc_kwargs)
        mcmc.run(rng_key, *args, **kwargs)

        return mcmc.get_samples()

    def _combine_inferences(self, rng_key, samples, branching_traces, *args, **kwargs):
        """
        Weight each SLP proportional to its estimated normalization constant.
        The normalization constants are estimated using importance sampling with
        the proposal centred on the MCMC samples. This is a special case of the
        layered adaptive importance sampling algorithm from [1].

        **References:**
        1. *Layered adaptive importance sampling*,
            Luca Martino, Victor Elvira, David Luengo, and Jukka Corander.
        """

        def log_weight(rng_key, i, slp_model, slp_samples):
            trace = {k: v[i] for k, v in slp_samples.items()}
            guide = AutoNormal(
                slp_model,
                init_loc_fn=init_to_value(values=trace),
                init_scale=self.proposal_scale,
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
