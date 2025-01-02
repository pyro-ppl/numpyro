# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import math
from typing import Iterable

import numpy as np

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import biject_to, constraints
from numpyro.distributions.util import is_identically_one, safe_normalize, sum_rightmost
from numpyro.infer.autoguide import AutoContinuous
from numpyro.util import not_jax_tracer


class Reparam(ABC):
    """
    Base class for reparameterizers.
    """

    @abstractmethod
    def __call__(self, name, fn, obs):
        """
        :param str name: A sample site name.
        :param ~numpyro.distributions.Distribution fn: A distribution.
        :param numpy.ndarray obs: Observed value or None.
        :return: A pair (``new_fn``, ``value``).
        """
        return fn, obs

    def _unwrap(self, fn):
        """
        Unwrap Independent(...) and ExpandedDistribution(...) distributions.
        We can recover the input `fn` from the result triple `(fn, expand_shape, event_dim)`
        with `fn.expand(expand_shape).to_event(event_dim - fn.event_dim)`.
        """
        shape = fn.shape()
        event_dim = fn.event_dim
        while isinstance(fn, (dist.Independent, dist.ExpandedDistribution)):
            fn = fn.base_dist
        expand_shape = shape[: len(shape) - fn.event_dim]
        return fn, expand_shape, event_dim

    def _wrap(self, fn, expand_shape, event_dim):
        """
        Wrap in Independent and ExpandedDistribution distributions.
        """
        # Match batch_shape.
        assert fn.event_dim <= event_dim
        fn = fn.expand(expand_shape)  # no-op if expand_shape == fn.batch_shape

        # Match event_dim.
        if fn.event_dim < event_dim:
            fn = fn.to_event(event_dim - fn.event_dim)
        assert fn.event_dim == event_dim
        return fn


class LocScaleReparam(Reparam):
    """
    Generic decentering reparameterizer [1] for latent variables parameterized
    by ``loc`` and ``scale`` (and possibly additional ``shape_params``).

    This reparameterization works only for latent variables, not likelihoods.

    **References:**

    1. *Automatic Reparameterisation of Probabilistic Programs*,
       Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)

    :param float centered: optional centered parameter. If None (default) learn
        a per-site per-element centering parameter in ``[0,1]`` initialized at value 0.5.
        To sample the parameter, consider using :class:`~numpyro.handlers.lift` handler with a
        prior like ``Uniform(0, 1)`` to cast the parameter to a latent variable. If 0, fully
        decenter the distribution; if 1, preserve the centered distribution
        unchanged.
    :param shape_params: list of additional parameter names to copy unchanged from
        the centered to decentered distribution.
    :type shape_params: tuple or list
    """

    def __init__(self, centered=None, shape_params=()):
        assert centered is None or isinstance(
            centered, (int, float, np.generic, np.ndarray, jnp.ndarray, jax.core.Tracer)
        )
        assert isinstance(shape_params, (tuple, list))
        assert all(isinstance(name, str) for name in shape_params)
        if centered is not None:
            is_valid = constraints.unit_interval.check(centered)
            if not_jax_tracer(is_valid):
                if not np.all(is_valid):
                    raise ValueError(
                        "`centered` argument does not satisfy `0 <= centered <= 1`."
                    )

        self.centered = centered
        self.shape_params = shape_params

    def __call__(self, name, fn, obs):
        assert obs is None, "LocScaleReparam does not support observe statements"
        support = fn.support
        if isinstance(support, constraints.independent):
            support = fn.support.base_constraint
        if support is not constraints.real:
            raise ValueError(
                "LocScaleReparam only supports distributions with real "
                f"support, but got {support} support at site {name}."
            )

        centered = self.centered
        if is_identically_one(centered):
            return fn, obs
        event_shape = fn.event_shape
        fn, expand_shape, event_dim = self._unwrap(fn)

        # Apply a partial decentering transform.
        params = {key: getattr(fn, key) for key in self.shape_params}
        if self.centered is None:
            centered = numpyro.param(
                "{}_centered".format(name),
                jnp.full(event_shape, 0.5),
                constraint=constraints.unit_interval,
            )
        if isinstance(centered, (int, float, np.generic)) and centered == 0.0:
            params["loc"] = jnp.zeros_like(fn.loc)
            params["scale"] = jnp.ones_like(fn.scale)
        else:
            params["loc"] = fn.loc * centered
            params["scale"] = fn.scale**centered
        decentered_fn = self._wrap(type(fn)(**params), expand_shape, event_dim)

        # Draw decentered noise.
        decentered_value = numpyro.sample("{}_decentered".format(name), decentered_fn)

        # Differentiably transform.
        delta = decentered_value - centered * fn.loc
        value = fn.loc + jnp.power(fn.scale, 1 - centered) * delta

        # Simulate a pyro.deterministic() site.
        return None, value


class TransformReparam(Reparam):
    """
    Reparameterizer for
    :class:`~numpyro.distributions.TransformedDistribution` latent variables.

    This is useful for transformed distributions with complex,
    geometry-changing transforms, where the posterior has simple shape in
    the space of ``base_dist``.

    This reparameterization works only for latent variables, not likelihoods.
    """

    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        fn, expand_shape, event_dim = self._unwrap(fn)
        if not isinstance(fn, dist.TransformedDistribution):
            raise ValueError(
                "TransformReparam does not automatically work with {}"
                " distribution anymore. Please explicitly using"
                " TransformedDistribution(base_dist, AffineTransform(...)) pattern"
                " with TransformReparam.".format(type(fn).__name__)
            )

        # Draw noise from the base distribution.
        base_event_dim = event_dim
        for t in reversed(fn.transforms):
            base_event_dim += t.domain.event_dim - t.codomain.event_dim
        x = numpyro.sample(
            "{}_base".format(name),
            self._wrap(fn.base_dist, expand_shape, base_event_dim),
        )

        # Differentiably transform.
        for t in fn.transforms:
            x = t(x)

        # Simulate a pyro.deterministic() site.
        return None, x


class ProjectedNormalReparam(Reparam):
    """
    Reparametrizer for :class:`~numpyro.distributions.ProjectedNormal` latent
    variables.

    This reparameterization works only for latent variables, not likelihoods.
    """

    def __call__(self, name, fn, obs):
        assert obs is None, "ProjectedNormalReparam does not support observe statements"
        fn, expand_shape, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.ProjectedNormal)

        # Draw parameter-free noise.
        new_fn = dist.Normal(jnp.zeros(fn.concentration.shape), 1).to_event(1)
        x = numpyro.sample(
            "{}_normal".format(name), self._wrap(new_fn, expand_shape, event_dim)
        )

        # Differentiably transform.
        value = safe_normalize(x + fn.concentration)

        # Simulate a pyro.deterministic() site.
        return None, value


class NeuTraReparam(Reparam):
    """
    Neural Transport reparameterizer [1] of multiple latent variables.

    This uses a trained :class:`~numpyro.infer.autoguide.AutoContinuous`
    guide to alter the geometry of a model, typically for use e.g. in MCMC.
    Example usage::

        # Step 1. Train a guide
        guide = AutoIAFNormal(model)
        svi = SVI(model, guide, ...)
        # ...train the guide...

        # Step 2. Use trained guide in NeuTra MCMC
        neutra = NeuTraReparam(guide)
        model = neutra.reparam(model)
        nuts = NUTS(model)
        # ...now use the model in HMC or NUTS...

    This reparameterization works only for latent variables, not likelihoods.
    Note that all sites must share a single common :class:`NeuTraReparam`
    instance, and that the model must have static structure.

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~numpyro.infer.autoguide.AutoContinuous guide: A guide.
    :param params: trained parameters of the guide.
    """

    def __init__(self, guide, params):
        if not isinstance(guide, AutoContinuous):
            raise TypeError(
                "NeuTraReparam expected an AutoContinuous guide, but got {}".format(
                    type(guide)
                )
            )
        self.guide = guide
        self.params = params
        try:
            self.transform = self.guide.get_transform(params)
        except (NotImplementedError, TypeError) as e:
            raise ValueError(
                "NeuTraReparam only supports guides that implement "
                "`get_transform` method that does not depend on the "
                "model's `*args, **kwargs`"
            ) from e
        self._x_unconstrained = {}

    def _reparam_config(self, site):
        if site["name"] in self.guide.prototype_trace:
            # We only reparam if this is an unobserved site in the guide
            # prototype trace.
            guide_site = self.guide.prototype_trace[site["name"]]
            if not guide_site.get("is_observed", False):
                return self

    def reparam(self, fn=None):
        return numpyro.handlers.reparam(fn, config=self._reparam_config)

    def __call__(self, name, fn, obs):
        if name not in self.guide.prototype_trace:
            return fn, obs
        assert obs is None, "NeuTraReparam does not support observe statements"

        log_density = 0.0
        compute_density = numpyro.get_mask() is not False
        if not self._x_unconstrained:  # On first sample site.
            # Sample a shared latent.
            model_plates = {
                msg["name"]
                for msg in self.guide.prototype_trace.values()
                if msg["type"] == "plate"
            }
            z_unconstrained = numpyro.sample(
                "{}_shared_latent".format(self.guide.prefix),
                self.guide.get_base_dist().mask(False),
                infer={"block_plates": model_plates},
            )

            # Differentiably transform.
            x_unconstrained = self.transform(z_unconstrained)
            if compute_density:
                log_density = self.transform.log_abs_det_jacobian(
                    z_unconstrained, x_unconstrained
                )
            self._x_unconstrained = self.guide._unpack_latent(x_unconstrained)

        # Extract a single site's value from the shared latent.
        unconstrained_value = self._x_unconstrained.pop(name)
        transform = biject_to(fn.support)
        value = transform(unconstrained_value)
        if compute_density:
            logdet = transform.log_abs_det_jacobian(unconstrained_value, value)
            logdet = sum_rightmost(
                logdet, jnp.ndim(logdet) - jnp.ndim(value) + len(fn.event_shape)
            )
            log_density = log_density + fn.log_prob(value) + logdet
        numpyro.factor("_{}_log_prob".format(name), log_density)
        return None, value

    def transform_sample(self, latent):
        """
        Given latent samples from the warped posterior (with possible batch dimensions),
        return a `dict` of samples from the latent sites in the model.

        :param latent: sample from the warped posterior (possibly batched).
        :return: a `dict` of samples keyed by latent sites in the model.
        :rtype: dict
        """
        x_unconstrained = self.transform(latent)
        return self.guide._unpack_and_constrain(x_unconstrained, self.params)


class CircularReparam(Reparam):
    """
    Reparametrizer for :class:`~numpyro.distributions.VonMises` latent
    variables.
    """

    def __call__(self, name, fn, obs):
        # Support must be circular
        support = fn.support
        if isinstance(support, constraints.independent):
            support = fn.support.base_constraint
        assert support is constraints.circular
        assert obs is None, "CircularReparam does not support observe statements"

        # Draw parameter-free noise.
        new_fn = dist.ImproperUniform(constraints.real, fn.batch_shape, fn.event_shape)
        value = numpyro.sample(
            f"{name}_unwrapped",
            new_fn,
            obs=obs,
        )

        # Differentiably transform.
        value = jnp.remainder(value + math.pi, 2 * math.pi) - math.pi

        # Simulate a pyro.deterministic() site.
        numpyro.factor(f"{name}_factor", fn.log_prob(value))
        return None, value


class ExplicitReparam(Reparam):
    """
    Explicit reparametrizer of a latent variable :code:`x` to a transformed space
    :code:`y = transform(x)` with more amenable geometry. This reparametrizer is similar
    to :class:`.TransformReparam` but allows reparametrizations to be decoupled from the
    model declaration.

    :param transform: Bijective transform to the reparameterized space.

    **Example:**

    .. doctest::

        >>> from jax import random
        >>> from jax import numpy as jnp
        >>> import numpyro
        >>> from numpyro import handlers, distributions as dist
        >>> from numpyro.infer import MCMC, NUTS
        >>> from numpyro.infer.reparam import ExplicitReparam
        >>>
        >>> def model():
        ...    numpyro.sample("x", dist.Gamma(4, 4))
        >>>
        >>> # Sample in unconstrained space using a soft-plus instead of exp transform.
        >>> reparam = ExplicitReparam(dist.transforms.SoftplusTransform().inv)
        >>> reparametrized = handlers.reparam(model, {"x": reparam})
        >>> kernel = NUTS(model=reparametrized)
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
        >>> mcmc.run(random.PRNGKey(2))  # doctest: +SKIP
        sample: 100%|██████████| 2000/2000 [00:00<00:00, 2306.47it/s, 3 steps of size 9.65e-01. acc. prob=0.93]
    """

    def __init__(self, transform):
        if isinstance(transform, Iterable) and all(
            isinstance(t, dist.transforms.Transform) for t in transform
        ):
            transform = dist.transforms.ComposeTransform(transform)
        self.transform = transform

    def __call__(self, name, fn, obs):
        assert obs is None, "ExplicitReparam does not support observe statements"
        transformed = dist.TransformedDistribution(fn, self.transform)
        x = numpyro.sample(f"{name}_base", transformed)
        return None, self.transform.inv(x)
