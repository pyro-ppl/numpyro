# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
import warnings

import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import (
    ComposeTransform,
    IdentityTransform,
    biject_to,
)


def init_to_median(site=None, reinit_param=lambda site: False, num_samples=15):
    """
    Initialize to the prior median. For priors with no `.sample` method implemented,
    we defer to the :func:`init_to_uniform` strategy.

    :param int num_samples: number of prior points to calculate median.
    """
    if site is None:
        return partial(
            init_to_median, num_samples=num_samples, reinit_param=reinit_param
        )

    if (
            site["type"] == "sample"
            and not site["is_observed"]
            and not site["fn"].support.is_discrete
    ):
        if site["value"] is not None:
            warnings.warn(
                f"init_to_median() skipping initialization of site '{site['name']}'"
                " which already stores a value."
            )
            return site["value"]

        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape")
        try:
            samples = site["fn"](
                sample_shape=(num_samples,) + sample_shape, rng_key=rng_key
            )
            return jnp.median(samples, axis=0)
        except NotImplementedError:
            return init_to_uniform(site)

    if site["type"] == "param" and reinit_param(site):
        return site["args"][0]


def init_to_sample(site=None, reinit_param=lambda site: False):
    """
    Initialize to a prior sample. For priors with no `.sample` method implemented,
    we defer to the :func:`init_to_uniform` strategy.
    """
    return init_to_median(site, num_samples=1, reinit_param=reinit_param)


def init_to_uniform(site=None, radius=2, reinit_param=lambda site: False):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.
    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    if site is None:
        return partial(init_to_uniform, radius=radius, reinit_param=reinit_param)

    if (
            site["type"] == "sample"
            and not site["is_observed"]
            and not site["fn"].support.is_discrete
    ) or (site["type"] == "param" and reinit_param(site)):
        if site["value"] is not None:
            warnings.warn(
                f"init_to_uniform() skipping initialization of site '{site['name']}'"
                " which already stores a value."
            )
            return site["value"]

        # XXX: we import here to avoid circular import
        from numpyro.infer.util import helpful_support_errors

        if site["type"] == "sample":
            with helpful_support_errors(site):
                prototype_shape = site["fn"].shape()
                transform = biject_to(site["fn"].support)
        elif site["type"] == "param":
            prototype_shape = jnp.shape(site["args"][0])
            transform = get_parameter_transform(site)

        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape", ())
        unconstrained_shape = transform.inverse_shape(prototype_shape)
        unconstrained_samples = dist.Uniform(-radius, radius)(
            rng_key=rng_key, sample_shape=sample_shape + unconstrained_shape
        )
        return transform(unconstrained_samples)


def init_to_feasible(site=None, reinit_param=lambda site: False):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    return init_to_uniform(site, radius=0, reinit_param=reinit_param)


def init_to_value(site=None, values=None, reinit_param=lambda site: False):
    """
    Initialize to the value specified in `values`. We defer to
    :func:`init_to_uniform` strategy for sites which do not appear in `values`.

    :param dict values: dictionary of initial values keyed by site name.
    """
    if values is None:
        values = {}
    if site is None:
        return partial(init_to_value, values=values, reinit_param=reinit_param)

    if (site["type"] == "sample" and not site["is_observed"]) or (
            site["type"] == "param" and reinit_param(site)
    ):
        if site["name"] in values:
            return values[site["name"]]
        else:  # defer to default strategy
            return init_to_uniform(site, reinit_param=reinit_param)


def init_with_noise(
        init_strategy, site=None, noise_scale=1.0, reinit_param=lambda site: False
):
    if site is None:
        return partial(
            init_with_noise,
            init_strategy,
            noise_scale=noise_scale,
            reinit_param=reinit_param,
        )
    vals = init_strategy(site, reinit_param=reinit_param)
    if isinstance(site["fn"], dist.TransformedDistribution):
        fn = site["fn"].base_dist
    else:
        fn = site["fn"]
    if vals is not None:
        if site["type"] == "param":
            base_transform = get_parameter_transform(site)
        elif site["type"] == "sample":
            base_transform = biject_to(fn.support)
        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape", ())
        unconstrained_init = dist.Normal(
            loc=base_transform.inv(vals), scale=noise_scale
        ).sample(rng_key, sample_shape)
        return base_transform(unconstrained_init)
    else:
        return None


def get_parameter_transform(site):
    constraint = site["kwargs"].get("constraint", real)
    transform = site["kwargs"].get("particle_transform", IdentityTransform())
    return ComposeTransform([transform, biject_to(constraint)])
