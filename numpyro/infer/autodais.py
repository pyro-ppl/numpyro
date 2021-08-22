# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import jax.numpy as jnp
from jax import lax, tree_map

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.distributions.util import periodic_repeat
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import helpful_support_errors
from .autoguide.py import AutoGuide


class AutoDAIS(AutoGuide):
    """
    This implementation of :class:`AutoDais` uses DAIS [1, 2] to construct
    a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    References:
    [1] "MCMC Variational Inference via Uncorrected Hamiltonian Annealing,"
        Tomas Geffner, Justin Domke.
    [2] "Differentiable Annealed Importance Sampling and the Perils of Gradient Noise,"
        Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger Grosse.

    Usage::

        guide = AutoDAIS(model)
        svi = SVI(model, guide, ...)

    :param callable model: A NumPyro model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`numpyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.positive

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
        create_plates=None,
    ):
        self._init_scale = init_scale
        self._event_dims = {}
        super().__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates
        )

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = (
                site["fn"].event_dim
                + jnp.ndim(self._init_locs[name])
                - jnp.ndim(site["value"])
            )
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = self._prototype_frame_full_sizes[frame.name]
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    self._init_locs[name] = periodic_repeat(
                        self._init_locs[name], full_size, dim
                    )

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = self._event_dims[name]
            init_loc = self._init_locs[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                site_loc = numpyro.param(
                    "{}_{}_loc".format(name, self.prefix), init_loc, event_dim=event_dim
                )
                site_scale = numpyro.param(
                    "{}_{}_scale".format(name, self.prefix),
                    jnp.full(jnp.shape(init_loc), self._init_scale),
                    constraint=self.scale_constraint,
                    event_dim=event_dim,
                )

                site_fn = dist.Normal(site_loc, site_scale).to_event(event_dim)
                if site["fn"].support is constraints.real or (
                    isinstance(site["fn"].support, constraints.independent)
                    and site["fn"].support is constraints.real
                ):
                    result[name] = numpyro.sample(name, site_fn)
                else:
                    with helpful_support_errors(site):
                        transform = biject_to(site["fn"].support)
                    guide_dist = dist.TransformedDistribution(site_fn, transform)
                    result[name] = numpyro.sample(name, guide_dist)

        return result

    def _constrain(self, latent_samples):
        name = list(latent_samples)[0]
        sample_shape = jnp.shape(latent_samples[name])[
            : jnp.ndim(latent_samples[name]) - jnp.ndim(self._init_locs[name])
        ]
        if sample_shape:
            flatten_samples = tree_map(
                lambda x: jnp.reshape(x, (-1,) + jnp.shape(x)[len(sample_shape) :]),
                latent_samples,
            )
            contrained_samples = lax.map(self._postprocess_fn, flatten_samples)
            return tree_map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                contrained_samples,
            )
        else:
            return self._postprocess_fn(latent_samples)

    def sample_posterior(self, rng_key, params, sample_shape=()):
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        scales = {k: params["{}_{}_scale".format(k, self.prefix)] for k in locs}
        with handlers.seed(rng_seed=rng_key):
            latent_samples = {}
            for k in locs:
                latent_samples[k] = numpyro.sample(
                    k, dist.Normal(locs[k], scales[k]).expand_by(sample_shape)
                )
        return self._constrain(latent_samples)
