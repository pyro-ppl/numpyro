# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from jax import random
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions import biject_to


def init_to_median(site=None, num_samples=15):
    """
    Initialize to the prior median.

    :param int num_samples: number of prior points to calculate median.
    """
    if site is None:
        return partial(init_to_median, num_samples=num_samples)

    if site['type'] == 'sample' and not site['is_observed'] and not site['fn'].is_discrete:
        rng_key = site['kwargs'].get('rng_key')
        sample_shape = site['kwargs'].get('sample_shape')
        samples = site['fn'].sample(rng_key, sample_shape=(num_samples,) + sample_shape)
        return np.median(samples, axis=0)


def init_to_prior(site=None):
    """
    Initialize to a prior sample.
    """
    return init_to_median(site, num_samples=1)


def init_to_uniform(site=None, radius=2):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    if site is None:
        return partial(init_to_uniform, radius=radius)

    if site['type'] == 'sample' and not site['is_observed'] and not site['fn'].is_discrete:
        rng_key = site['kwargs'].get('rng_key')
        sample_shape = site['kwargs'].get('sample_shape')
        rng_key, subkey = random.split(rng_key)
        # this is used to interpret the changes of event_shape in
        # domain and codomain spaces
        prototype_value = site['fn'].sample(subkey, sample_shape=())
        transform = biject_to(site['fn'].support)
        unconstrained_event_shape = np.shape(transform.inv(prototype_value))
        unconstrained_samples = dist.Uniform(-radius, radius).sample(
            rng_key, sample_shape=sample_shape + unconstrained_event_shape)
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

    if site['type'] == 'sample' and not site['is_observed']:
        if site['name'] in values:
            return values[site['name']]
        else:  # defer to default strategy
            return init_to_uniform(site)
