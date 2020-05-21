# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import jax.numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import biject_to
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import ComposeTransform


def _init_to_median(site, num_samples=15, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        samples = numpyro.sample('_init', fn,
                                 sample_shape=(num_samples,) + site['kwargs']['sample_shape'])
        return np.median(samples, axis=0)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
            value = base_transform(transform.inv(value))
        return value


def init_to_median(num_samples=15):
    """
    Initialize to the prior median.

    :param int num_samples: number of prior points to calculate median.
    """
    return partial(_init_to_median, num_samples=num_samples)


def init_to_prior():
    """
    Initialize to a prior sample.
    """
    return init_to_median(num_samples=1)


def _init_to_uniform(site, radius=2, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        value = numpyro.sample('_init', fn, sample_shape=site['kwargs']['sample_shape'])
        base_transform = biject_to(fn.support)
        unconstrained_value = numpyro.sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                             sample_shape=np.shape(base_transform.inv(value)))
        return base_transform(unconstrained_value)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        unconstrained_value = numpyro.sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                             sample_shape=np.shape(transform.inv(value)))
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
        else:
            base_transform = transform
        return base_transform(unconstrained_value)


def init_to_uniform(site=None, radius=2):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    if site is None:
        return partial(init_to_uniform, radius=radius)


def init_to_feasible():
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    return init_to_uniform(radius=0)


def _init_to_value(site, values={}, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if site['name'] not in values:
            return _init_to_uniform(site, skip_param=skip_param)

        value = values[site['name']]
        if isinstance(site['fn'], dist.TransformedDistribution):
            value = ComposeTransform(site['fn'].transforms).inv(value)
        return value

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
            value = base_transform(transform.inv(value))
        return value


def init_to_value(values):
    """
    Initialize to the value specified in `values`. We defer to
    :func:`init_to_uniform` strategy for sites which do not appear in `values`.

    :param dict values: dictionary of initial values keyed by site name.
    """
    return partial(_init_to_value, values=values)
