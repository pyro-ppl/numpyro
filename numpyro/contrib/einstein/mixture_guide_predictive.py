# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp, random, vmap

from numpyro.handlers import substitute
from numpyro.infer import Predictive
from numpyro.infer.util import _predictive


class MixtureGuidePredictive:
    """(EXPERIMENTAL INTERFACE) This class constructs the predictive distribution for
    :class:`numpyro.contrib.einstein.steinvi.SteinVi`

    .. Note:: For single mixture component use numpyro.infer.Predictive.

    .. warning::
        The `MixtureGuidePredictive` is experimental and will likely be replaced by
        :class:`numpyro.infer.util.Predictive` in the future.

    :param Callable model: Python callable containing Pyro primitives.
    :param Callable guide: Python callable containing Pyro primitives to get posterior samples of sites.
    :param dict params:  Dictionary of values for param sites of model/guide
    :param Sequence guide_sites: Names of sites that contribute to the Stein mixture.
    :param Optional[int] num_samples:
    :param Optional[Sequence[str]] return_sites: Sites to return. By default, only sample sites not present
        in the guide are returned.
    :param str mixture_assignment_sitename: Name of site for mixture component assignment for sites not in the Stein
        mixture.
    """

    def __init__(
        self,
        model: Callable,
        guide: Callable,
        params: dict,
        guide_sites: Sequence,
        num_samples: Optional[int] = None,
        return_sites: Optional[Sequence[str]] = None,
        mixture_assignment_sitename="mixture_assignments",
    ):
        self.model_predictive = partial(
            Predictive,
            model=model,
            params={
                name: param for name, param in params.items() if name not in guide_sites
            },
            num_samples=num_samples,
            return_sites=return_sites,
            infer_discrete=False,
            parallel=False,
        )
        self._batch_shape = (num_samples,)
        self.parallel = False
        self.guide_params = {
            name: param for name, param in params.items() if name in guide_sites
        }

        self.guide = guide
        self.return_sites = return_sites
        self.num_mixture_components = jnp.shape(jax.tree.flatten(params)[0][0])[0]
        self.mixture_assignment_sitename = mixture_assignment_sitename

    def _call_with_params(self, rng_key, params, args, kwargs):
        rng_key, guide_rng_key = random.split(rng_key)
        # use return_sites='' as a special signal to return all sites
        guide = substitute(self.guide, params)
        samples = _predictive(
            guide_rng_key,
            guide,
            {},
            self._batch_shape,
            return_sites="",
            parallel=self.parallel,
            model_args=args,
            model_kwargs=kwargs,
        )
        return samples

    def __call__(self, rng_key, *args, **kwargs):
        guide_key, assign_key, model_key = random.split(rng_key, 3)

        samples_guide = vmap(
            lambda key, params: self._call_with_params(
                key, params=params, args=args, kwargs=kwargs
            ),
            in_axes=0,
            out_axes=1,
        )(random.split(guide_key, self.num_mixture_components), self.guide_params)

        assigns = random.randint(
            assign_key,
            shape=self._batch_shape,
            minval=0,
            maxval=self.num_mixture_components,
        )
        predictive_assign = jax.tree.map(
            lambda arr: vmap(lambda i, assign: arr[i, assign])(
                jnp.arange(self._batch_shape[0]), assigns
            ),
            samples_guide,
        )
        predictive_model = self.model_predictive(posterior_samples=predictive_assign)
        samples_model = predictive_model(model_key, *args, **kwargs)
        if self.return_sites is not None:
            samples_guide = {
                name: value
                for name, value in samples_guide.items()
                if name in self.return_sites
            }
        else:
            samples_guide = {}

        return {
            self.mixture_assignment_sitename: assigns,
            **samples_guide,
            **samples_model,  # use samples from model if site in model and guide
        }
