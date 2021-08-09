# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from jax import random
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.distributions import biject_to


def init_to_median(site=None, num_samples=15):
    """
    Initialize to the prior median. For priors with no `.sample` method implemented,
    we defer to the :func:`init_to_uniform` strategy.

    :param int num_samples: number of prior points to calculate median.
    """
    if site is None:
        return partial(init_to_median, num_samples=num_samples)

    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and not site["fn"].is_discrete
    ):
        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape")
        try:
            samples = site["fn"](
                sample_shape=(num_samples,) + sample_shape, rng_key=rng_key
            )
            return jnp.median(samples, axis=0)
        except NotImplementedError:
            return init_to_uniform(site)


def init_to_sample(site=None):
    """
    Initialize to a prior sample. For priors with no `.sample` method implemented,
    we defer to the :func:`init_to_uniform` strategy.
    """
    return init_to_median(site, num_samples=1)


def init_to_uniform(site=None, radius=2):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    if site is None:
        return partial(init_to_uniform, radius=radius)

    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and not site["fn"].is_discrete
    ):
        # XXX: we import here to avoid circular import
        from numpyro.infer.util import helpful_support_errors

        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape")
        rng_key, subkey = random.split(rng_key)

        with helpful_support_errors(site):
            transform = biject_to(site["fn"].support)
        unconstrained_shape = transform.inverse_shape(site["fn"].shape())
        unconstrained_samples = dist.Uniform(-radius, radius)(
            rng_key=rng_key, sample_shape=sample_shape + unconstrained_shape
        )
        return transform(unconstrained_samples)


def init_to_feasible(site=None):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    return init_to_uniform(site, radius=0)


def init_to_value(site=None, values={}):
    """
    Initialize to the value specified in `values`. We defer to
    :func:`init_to_uniform` strategy for sites which do not appear in `values`.

    :param dict values: dictionary of initial values keyed by site name.
    """
    if site is None:
        return partial(init_to_value, values=values)

    if site["type"] == "sample" and not site["is_observed"]:
        if site["name"] in values:
            return values[site["name"]]
        else:  # defer to default strategy
            return init_to_uniform(site)
