# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Adapted from pyro.infer.autoguide
from abc import ABC, abstractmethod
from contextlib import ExitStack
from functools import partial
import math
import warnings

import numpy as np

import jax
from jax import grad, hessian, lax, random
from jax.example_libraries import stax
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.flows import (
    BlockNeuralAutoregressiveTransform,
    InverseAutoregressiveTransform,
)
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    IndependentTransform,
    LowerCholeskyAffine,
    PermuteTransform,
    ReshapeTransform,
    UnpackTransform,
    biject_to,
)
from numpyro.distributions.util import (
    cholesky_of_inverse,
    periodic_repeat,
    sum_rightmost,
)
from numpyro.infer import Predictive
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.infer.util import (
    helpful_support_errors,
    initialize_model,
    potential_energy,
)
from numpyro.nn.auto_reg_nn import AutoregressiveNN
from numpyro.nn.block_neural_arn import BlockNeuralAutoregressiveNN
from numpyro.util import find_stack_level, not_jax_tracer

__all__ = [
    "AutoBatchedLowRankMultivariateNormal",
    "AutoBatchedMultivariateNormal",
    "AutoContinuous",
    "AutoGuide",
    "AutoGuideList",
    "AutoDAIS",
    "AutoDiagonalNormal",
    "AutoLaplaceApproximation",
    "AutoLowRankMultivariateNormal",
    "AutoNormal",
    "AutoMultivariateNormal",
    "AutoBNAFNormal",
    "AutoIAFNormal",
    "AutoDelta",
    "AutoSemiDAIS",
    "AutoSurrogateLikelihoodDAIS",
]


class AutoGuide(ABC):
    """
    Base class for automatic guides.

    Derived classes must implement the :meth:`__call__` method.

    :param callable model: a pyro model
    :param str prefix: a prefix that will be prefixed to all param internal sites
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`numpyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    def __init__(
        self, model, *, prefix="auto", init_loc_fn=init_to_uniform, create_plates=None
    ):
        self.model = model
        self.prefix = prefix
        self.init_loc_fn = init_loc_fn
        self.create_plates = create_plates
        self.prototype_trace = None
        self._prototype_frames = {}
        self._prototype_frame_full_sizes = {}

    def _create_plates(self, *args, **kwargs):
        if self.create_plates is None:
            self.plates = {}
        else:
            plates = self.create_plates(*args, **kwargs)
            if isinstance(plates, numpyro.plate):
                plates = [plates]
            assert all(
                isinstance(p, numpyro.plate) for p in plates
            ), "create_plates() returned a non-plate"
            self.plates = {p.name: p for p in plates}
        for name, frame in sorted(self._prototype_frames.items()):
            if name not in self.plates:
                full_size = self._prototype_frame_full_sizes[name]
                self.plates[name] = numpyro.plate(
                    name, full_size, dim=frame.dim, subsample_size=frame.size
                )
        return self.plates

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("plates", None)
        return state

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        A guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def sample_posterior(self, rng_key, params, *, sample_shape=()):
        """
        Generate samples from the approximate posterior over the latent
        sites in the model.

        :param jax.random.PRNGKey rng_key: random key to be used draw samples.
        :param dict params: Current parameters of model and autoguide.
            The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
            method from :class:`~numpyro.infer.svi.SVI`.
        :param tuple sample_shape: sample shape of each latent site, defaults to ().
        :return: a dict containing samples drawn the this guide.
        :rtype: dict
        """
        raise NotImplementedError

    def _setup_prototype(self, *args, **kwargs):
        rng_key = numpyro.prng_key()
        with handlers.block():
            (
                init_params,
                self._potential_fn_gen,
                postprocess_fn_gen,
                self.prototype_trace,
            ) = initialize_model(
                rng_key,
                self.model,
                init_strategy=self.init_loc_fn,
                dynamic_args=True,
                model_args=args,
                model_kwargs=kwargs,
            )
        self._potential_fn = self._potential_fn_gen(*args, **kwargs)
        postprocess_fn = postprocess_fn_gen(*args, **kwargs)
        # We apply a fixed seed just in case postprocess_fn requires
        # a random key to generate subsample indices. It does not matter
        # because we only collect deterministic sites.
        self._postprocess_fn = handlers.seed(postprocess_fn, rng_seed=0)
        self._init_locs = init_params[0]

        self._prototype_frames = {}
        self._prototype_plate_sizes = {}
        for name, site in self.prototype_trace.items():
            if site["type"] == "sample":
                if not site["is_observed"] and site["fn"].support.is_discrete:
                    # raise support errors early for discrete sites
                    with helpful_support_errors(site):
                        biject_to(site["fn"].support)
                for frame in site["cond_indep_stack"]:
                    if frame.name in self._prototype_frames:
                        assert (
                            frame == self._prototype_frames[frame.name]
                        ), f"The plate {frame.name} has inconsistent dim or size. Please check your model again."
                    else:
                        self._prototype_frames[frame.name] = frame
            elif site["type"] == "plate":
                self._prototype_frame_full_sizes[name] = site["args"][0]

    def median(self, params):
        """
        Returns the posterior median value of each latent variable.

        :param dict params: A dict containing parameter values.
            The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
            method from :class:`~numpyro.infer.svi.SVI`.
        :return: A dict mapping sample site name to median value.
        :rtype: dict
        """
        raise NotImplementedError

    def quantiles(self, params, quantiles):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles(params, [0.05, 0.5, 0.95]))

        :param dict params: A dict containing parameter values.
            The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
            method from :class:`~numpyro.infer.svi.SVI`.
        :param list quantiles: A list of requested quantiles between 0 and 1.
        :return: A dict mapping sample site name to an array of quantile values.
        :rtype: dict
        """
        raise NotImplementedError


class AutoGuideList(AutoGuide):
    """
    Container class to combine multiple automatic guides.

    Example usage::

        rng_key_init = random.PRNGKey(0)
        guide = AutoGuideList(my_model)
        guide.append(
            AutoNormal(
                numpyro.handlers.block(numpyro.handlers.seed(model, rng_seed=0), hide=["coefs"])
            )
        )
        guide.append(
            AutoDelta(
                numpyro.handlers.block(numpyro.handlers.seed(model, rng_seed=1), expose=["coefs"])
            )
        )
        svi = SVI(model, guide, optim, Trace_ELBO())
        svi_state = svi.init(rng_key_init, data, labels)
        params = svi.get_params(svi_state)

    :param callable model: a NumPyro model
    """

    def __init__(
        self, model, *, prefix="auto", init_loc_fn=init_to_uniform, create_plates=None
    ):
        self._guides = []
        super().__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates
        )

    def append(self, part):
        """
        Add an automatic or custom guide for part of the model. The guide should
        have been created by blocking the model to restrict to a subset of
        sample sites. No two parts should operate on any one sample site.

        :param part: a partial guide to add
        :type part: AutoGuide
        """
        if (
            isinstance(part, numpyro.infer.autoguide.AutoDAIS)
            or isinstance(part, numpyro.infer.autoguide.AutoSemiDAIS)
            or isinstance(part, numpyro.infer.autoguide.AutoSurrogateLikelihoodDAIS)
        ):
            raise ValueError(
                "AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported."
            )
        self._guides.append(part)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)
            check_deterministic_sites = True
        else:
            check_deterministic_sites = False

        # create all plates
        self._create_plates(*args, **kwargs)

        # run sub-guides
        result = {}
        for part in self._guides:
            result.update(part(*args, **kwargs))

        # Check deterministic sites after calling sub-guides because they are not
        # initialized prior to the first call. We do not check guides that do not have
        # a prototype_trace attribute, e.g., custom guides.
        if check_deterministic_sites:
            for i, part in enumerate(self._guides):
                prototype_trace = getattr(part, "prototype_trace", None)
                if prototype_trace:
                    for key, value in prototype_trace.items():
                        if value["type"] == "deterministic":
                            raise ValueError(
                                f"deterministic site '{key}' in sub-guide at position "
                                f"{i} should not be exposed"
                            )

        return result

    def __getitem__(self, key):
        return self._guides[key]

    def __len__(self):
        return len(self._guides)

    def __iter__(self):
        yield from self._guides

    def sample_posterior(self, rng_key, params, *args, sample_shape=(), **kwargs):
        result = {}
        for part in self._guides:
            # TODO: remove this when sample_posterior() signatures are consistent
            # we know part is not AutoDAIS, AutoSemiDAIS, or AutoSurrogateLikelihoodDAIS
            if isinstance(part, numpyro.infer.autoguide.AutoDelta):
                result.update(
                    part.sample_posterior(
                        rng_key, params, *args, sample_shape=sample_shape, **kwargs
                    )
                )
            else:
                result.update(
                    part.sample_posterior(rng_key, params, sample_shape=sample_shape)
                )
        return result

    def median(self, params):
        result = {}
        for part in self._guides:
            result.update(part.median(params))
        return result

    def quantiles(self, params, quantiles):
        result = {}
        for part in self._guides:
            result.update(part.quantiles(params, quantiles))
        return result


class AutoNormal(AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Normal distributions
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    This should be equivalent to :class:`AutoDiagonalNormal` , but with
    more convenient site names and with better support for mean field ELBO.

    Usage::

        guide = AutoNormal(model)
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

    scale_constraint = constraints.softplus_positive

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
                    and site["fn"].support.base_constraint is constraints.real
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
            flatten_samples = jax.tree.map(
                lambda x: jnp.reshape(x, (-1,) + jnp.shape(x)[len(sample_shape) :]),
                latent_samples,
            )
            contrained_samples = lax.map(self._postprocess_fn, flatten_samples)
            return jax.tree.map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                contrained_samples,
            )
        else:
            return self._postprocess_fn(latent_samples)

    def sample_posterior(self, rng_key, params, *, sample_shape=()):
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        scales = {k: params["{}_{}_scale".format(k, self.prefix)] for k in locs}
        with handlers.seed(rng_seed=rng_key):
            latent_samples = {}
            for k in locs:
                latent_samples[k] = numpyro.sample(
                    k, dist.Normal(locs[k], scales[k]).expand_by(sample_shape)
                )
        return self._constrain(latent_samples)

    def median(self, params):
        locs = {
            k: params["{}_{}_loc".format(k, self.prefix)]
            for k, v in self._init_locs.items()
        }
        return self._constrain(locs)

    def quantiles(self, params, quantiles):
        quantiles = jnp.array(quantiles)
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        scales = {k: params["{}_{}_scale".format(k, self.prefix)] for k in locs}
        latent = {
            k: dist.Normal(locs[k], scales[k]).icdf(
                quantiles.reshape((-1,) + (1,) * jnp.ndim(locs[k]))
            )
            for k in locs
        }
        return self._constrain(latent)


class AutoDelta(AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Delta distributions to
    construct a MAP guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    .. note:: This class does MAP inference in constrained space.

    Usage::

        guide = AutoDelta(model)
        svi = SVI(model, guide, ...)

    :param callable model: A NumPyro model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`numpyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    def __init__(
        self, model, *, prefix="auto", init_loc_fn=init_to_median, create_plates=None
    ):
        self._event_dims = {}
        super().__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates
        )

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        with numpyro.handlers.block():
            self._init_locs = {
                k: v
                for k, v in self._postprocess_fn(self._init_locs).items()
                if k in self._init_locs
            }
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = site["fn"].event_dim
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
                    "{}_{}_loc".format(name, self.prefix),
                    init_loc,
                    constraint=site["fn"].support,
                    event_dim=event_dim,
                )

                site_fn = dist.Delta(site_loc).to_event(event_dim)
                result[name] = numpyro.sample(name, site_fn)

        return result

    def sample_posterior(self, rng_key, params, *args, sample_shape=(), **kwargs):
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        latent_samples = {
            k: jnp.broadcast_to(v, sample_shape + jnp.shape(v)) for k, v in locs.items()
        }
        deterministic_vars = [
            k for k, v in self.prototype_trace.items() if v["type"] == "deterministic"
        ]
        if not deterministic_vars:
            return latent_samples
        else:
            predictive = Predictive(
                model=self.model,
                posterior_samples=latent_samples,
                return_sites=deterministic_vars,
                batch_ndims=len(sample_shape),
            )
            deterministic_samples = predictive(rng_key, *args, **kwargs)
            return {**latent_samples, **deterministic_samples}

    def median(self, params):
        locs = {k: params["{}_{}_loc".format(k, self.prefix)] for k in self._init_locs}
        return locs


def _unravel_dict(x_flat, shape_dict):
    """Return `x` from the flatten version `x_flat`. Shape information
    of each item in `x` is defined in `shape_dict`.
    """
    assert jnp.ndim(x_flat) == 1
    assert isinstance(shape_dict, dict)
    x = {}
    curr_pos = next_pos = 0
    for name, shape in shape_dict.items():
        next_pos = curr_pos + int(np.prod(shape))
        x[name] = x_flat[curr_pos:next_pos].reshape(shape)
        curr_pos = next_pos
    assert next_pos == x_flat.shape[0]
    return x


def _ravel_dict(x):
    """Return the flatten version of `x` and shapes of each item in `x`."""
    assert isinstance(x, dict)
    shape_dict = {name: jnp.shape(value) for name, value in x.items()}
    x_flat = _ravel_dict_with_shape_dict(x, shape_dict)
    return x_flat, shape_dict


def _ravel_dict_with_shape_dict(x, shape_dict):
    assert set(x.keys()) == set(shape_dict.keys())
    x_flat = []
    for name, shape in shape_dict.items():
        value = x[name]
        assert shape == jnp.shape(value)
        x_flat.append(jnp.reshape(value, -1))
    x_flat = jnp.concatenate(x_flat) if x_flat else jnp.zeros((0,))
    return x_flat


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].

    Each derived class implements its own :meth:`_get_posterior` method.

    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.

    **Reference:**

    1. *Automatic Differentiation Variational Inference*,
       Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M.
       Blei

    :param callable model: A NumPyro model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self._init_latent, shape_dict = _ravel_dict(self._init_locs)
        unpack_latent = partial(_unravel_dict, shape_dict=shape_dict)
        # this is to match the behavior of Pyro, where we can apply
        # unpack_latent for a batch of samples
        self._unpack_latent = UnpackTransform(
            unpack_latent, _ravel_dict_with_shape_dict
        )
        self.latent_dim = jnp.size(self._init_latent)
        if self.latent_dim == 0:
            raise RuntimeError(
                "{} found no latent variables; Use an empty guide instead".format(
                    type(self).__name__
                )
            )
        for site in self.prototype_trace.values():
            if site["type"] == "sample" and not site["is_observed"]:
                for frame in site["cond_indep_stack"]:
                    if frame.size != self._prototype_frame_full_sizes[frame.name]:
                        raise ValueError(
                            "AutoContinuous guide does not support"
                            " local latent variables."
                        )

    @abstractmethod
    def _get_posterior(self):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        sample_shape = kwargs.pop("sample_shape", ())
        posterior = self._get_posterior()
        return numpyro.sample(
            "_{}_latent".format(self.prefix),
            posterior.expand_by(sample_shape),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        latent = self._sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = {}

        for name, unconstrained_value in self._unpack_latent(latent).items():
            site = self.prototype_trace[name]
            with helpful_support_errors(site):
                transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            event_ndim = site["fn"].event_dim
            if numpyro.get_mask() is False:
                log_density = 0.0
            else:
                log_density = -transform.log_abs_det_jacobian(
                    unconstrained_value, value
                )
                log_density = sum_rightmost(
                    log_density, jnp.ndim(log_density) - jnp.ndim(value) + event_ndim
                )
            delta_dist = dist.Delta(
                value, log_density=log_density, event_dim=event_ndim
            )
            result[name] = numpyro.sample(name, delta_dist)

        return result

    def _unpack_and_constrain(self, latent_sample, params):
        def unpack_single_latent(latent):
            unpacked_samples = self._unpack_latent(latent)
            # XXX: we need to add param here to be able to replay model
            unpacked_samples.update(
                {
                    k: v
                    for k, v in params.items()
                    if k in self.prototype_trace
                    and self.prototype_trace[k]["type"] == "param"
                }
            )
            samples = self._postprocess_fn(unpacked_samples)
            # filter out param sites
            return {
                k: v
                for k, v in samples.items()
                if k in self.prototype_trace
                and self.prototype_trace[k]["type"] != "param"
            }

        sample_shape = jnp.shape(latent_sample)[:-1]
        if sample_shape:
            latent_sample = jnp.reshape(
                latent_sample, (-1, jnp.shape(latent_sample)[-1])
            )
            unpacked_samples = lax.map(unpack_single_latent, latent_sample)
            return jax.tree.map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                unpacked_samples,
            )
        else:
            return unpack_single_latent(latent_sample)

    def get_base_dist(self):
        """
        Returns the base distribution of the posterior when reparameterized
        as a :class:`~numpyro.distributions.distribution.TransformedDistribution`. This
        should not depend on the model's `*args, **kwargs`.
        """
        raise NotImplementedError

    def get_transform(self, params):
        """
        Returns the transformation learned by the guide to generate samples from the unconstrained
        (approximate) posterior.

        :param dict params: Current parameters of model and autoguide.
            The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
            method from :class:`~numpyro.infer.svi.SVI`.
        :return: the transform of posterior distribution
        :rtype: :class:`~numpyro.distributions.transforms.Transform`
        """
        posterior = handlers.substitute(self._get_posterior, params)()
        assert isinstance(
            posterior, dist.TransformedDistribution
        ), "posterior is not a transformed distribution"
        if len(posterior.transforms) > 0:
            return ComposeTransform(posterior.transforms)
        else:
            return posterior.transforms[0]

    def get_posterior(self, params):
        """
        Returns the posterior distribution.

        :param dict params: Current parameters of model and autoguide.
            The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
            method from :class:`~numpyro.infer.svi.SVI`.
        """
        base_dist = self.get_base_dist()
        transform = self.get_transform(params)
        return dist.TransformedDistribution(base_dist, transform)

    def sample_posterior(self, rng_key, params, *, sample_shape=()):
        latent_sample = handlers.substitute(
            handlers.seed(self._sample_latent, rng_key), params
        )(sample_shape=sample_shape)
        return self._unpack_and_constrain(latent_sample, params)


class AutoDAIS(AutoContinuous):
    """
    This implementation of :class:`AutoDAIS` uses Differentiable Annealed
    Importance Sampling (DAIS) [1, 2] to construct a guide over the entire
    latent space. Samples from the variational distribution (i.e. guide)
    are generated using a combination of (uncorrected) Hamiltonian Monte Carlo
    and Annealed Importance Sampling. The same algorithm is called Uncorrected
    Hamiltonian Annealing in [1].

    Note that AutoDAIS cannot be used in conjunction with data subsampling.

    **Reference:**

    1. *MCMC Variational Inference via Uncorrected Hamiltonian Annealing*,
       Tomas Geffner, Justin Domke
    2. *Differentiable Annealed Importance Sampling and the Perils of Gradient Noise*,
       Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger Grosse

    Usage::

        guide = AutoDAIS(model)
        svi = SVI(model, guide, ...)

    :param callable model: A NumPyro model.
    :param str prefix: A prefix that will be prefixed to all param internal sites.
    :param int K: A positive integer that controls the number of HMC steps used.
        Defaults to 4.
    :param str base_dist: Controls whether the base Normal variational distribution
       is parameterized by a "diagonal" covariance matrix or a full-rank covariance
       matrix parameterized by a lower-triangular "cholesky" factor. Defaults to "diagonal".
    :param float eta_init: The initial value of the step size used in HMC. Defaults
        to 0.01.
    :param float eta_max: The maximum value of the learnable step size used in HMC.
        Defaults to 0.1.
    :param float gamma_init: The initial value of the learnable damping factor used
        during partial momentum refreshments in HMC. Defaults to 0.9.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of
        the base variational distribution for each (unconstrained transformed)
        latent variable. Defaults to 0.1.
    """

    def __init__(
        self,
        model,
        *,
        K=4,
        base_dist="diagonal",
        eta_init=0.01,
        eta_max=0.1,
        gamma_init=0.9,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
    ):
        if K < 1:
            raise ValueError("K must satisfy K >= 1 (got K = {})".format(K))
        if base_dist not in ["diagonal", "cholesky"]:
            raise ValueError('base_dist must be one of "diagonal" or "cholesky".')
        if eta_init <= 0.0 or eta_init >= eta_max:
            raise ValueError(
                "eta_init must be positive and satisfy eta_init < eta_max."
            )
        if eta_max <= 0.0:
            raise ValueError("eta_max must be positive.")
        if gamma_init <= 0.0 or gamma_init >= 1.0:
            raise ValueError("gamma_init must be in the open interval (0, 1).")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")

        self.eta_init = eta_init
        self.eta_max = eta_max
        self.gamma_init = gamma_init
        self.K = K
        self.base_dist = base_dist
        self._init_scale = init_scale
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.items():
            if (
                site["type"] == "plate"
                and isinstance(site["args"][1], int)
                and site["args"][0] > site["args"][1]
            ):
                raise NotImplementedError(
                    "AutoDAIS cannot be used in conjunction with data subsampling."
                )

    def _get_posterior(self):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        def log_density(x):
            x_unpack = self._unpack_latent(x)
            with numpyro.handlers.block():
                return -self._potential_fn(x_unpack)

        eta0 = numpyro.param(
            "{}_eta0".format(self.prefix),
            self.eta_init,
            constraint=constraints.interval(0, self.eta_max),
        )
        eta_coeff = numpyro.param("{}_eta_coeff".format(self.prefix), 0.00)

        gamma = numpyro.param(
            "{}_gamma".format(self.prefix),
            self.gamma_init,
            constraint=constraints.interval(0, 1),
        )
        betas = numpyro.param(
            "{}_beta_increments".format(self.prefix),
            jnp.ones(self.K),
            constraint=constraints.positive,
        )
        betas = jnp.cumsum(betas)
        betas = betas / betas[-1]  # K-dimensional with betas[-1] = 1

        mass_matrix = numpyro.param(
            "{}_mass_matrix".format(self.prefix),
            jnp.ones(self.latent_dim),
            constraint=constraints.positive,
        )
        inv_mass_matrix = 0.5 / mass_matrix

        init_z_loc = numpyro.param(
            "{}_z_0_loc".format(self.prefix),
            self._init_latent,
        )

        if self.base_dist == "diagonal":
            init_z_scale = numpyro.param(
                "{}_z_0_scale".format(self.prefix),
                jnp.full(self.latent_dim, self._init_scale),
                constraint=constraints.positive,
            )
            base_z_dist = dist.Normal(init_z_loc, init_z_scale).to_event()
        elif self.base_dist == "cholesky":
            scale_tril = numpyro.param(
                "{}_z_0_scale_tril".format(self.prefix),
                jnp.identity(self.latent_dim) * self._init_scale,
                constraint=constraints.scaled_unit_lower_cholesky,
            )
            base_z_dist = dist.MultivariateNormal(init_z_loc, scale_tril=scale_tril)

        z_0 = numpyro.sample(
            "{}_z_0".format(self.prefix),
            base_z_dist,
            infer={"is_auxiliary": True},
        )
        momentum_dist = dist.Normal(0, mass_matrix).to_event()
        eps = numpyro.sample(
            "{}_momentum".format(self.prefix),
            momentum_dist.expand((self.K,)).to_event().mask(False),
            infer={"is_auxiliary": True},
        )

        def scan_body(carry, eps_beta):
            eps, beta = eps_beta
            eta = eta0 + eta_coeff * beta
            eta = jnp.clip(eta, 0.0, self.eta_max)
            z_prev, v_prev, log_factor = carry
            z_half = z_prev + v_prev * eta * inv_mass_matrix
            q_grad = (1.0 - beta) * grad(base_z_dist.log_prob)(z_half)
            p_grad = beta * grad(log_density)(z_half)
            v_hat = v_prev + eta * (q_grad + p_grad)
            z = z_half + v_hat * eta * inv_mass_matrix
            v = gamma * v_hat + jnp.sqrt(1 - gamma**2) * eps
            delta_ke = momentum_dist.log_prob(v_prev) - momentum_dist.log_prob(v_hat)
            log_factor = log_factor + delta_ke
            return (z, v, log_factor), None

        v_0 = eps[-1]  # note the return value of scan doesn't depend on eps[-1]
        (z, _, log_factor), _ = jax.lax.scan(scan_body, (z_0, v_0, 0.0), (eps, betas))

        numpyro.factor("{}_factor".format(self.prefix), log_factor)

        return z

    def sample_posterior(self, rng_key, params, *, sample_shape=()):
        def _single_sample(_rng_key):
            latent_sample = handlers.substitute(
                handlers.seed(self._sample_latent, _rng_key), params
            )(sample_shape=())
            return self._unpack_and_constrain(latent_sample, params)

        if sample_shape:
            rng_key = random.split(rng_key, int(np.prod(sample_shape)))
            samples = lax.map(_single_sample, rng_key)
            return jax.tree.map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                samples,
            )
        else:
            return _single_sample(rng_key)


class AutoSurrogateLikelihoodDAIS(AutoDAIS):
    """
    This implementation of :class:`AutoSurrogateLikelihoodDAIS` provides a
    mini-batchable family of variational distributions as described in [1].
    It combines a user-provided surrogate likelihood with Differentiable Annealed
    Importance Sampling (DAIS) [2, 3]. It is not applicable to models with local
    latent variables (see :class:`AutoSemiDAIS`), but unlike :class:`AutoDAIS`, it
    *can* be used in conjunction with data subsampling.

    **Reference:**

    1. *Surrogate likelihoods for variational annealed importance sampling*,
       Martin Jankowiak, Du Phan
    2. *MCMC Variational Inference via Uncorrected Hamiltonian Annealing*,
       Tomas Geffner, Justin Domke
    3. *Differentiable Annealed Importance Sampling and the Perils of Gradient Noise*,
       Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger Grosse

    Usage::

        # logistic regression model for data {X, Y}
        def model(X, Y):
            theta = numpyro.sample(
                "theta", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
            )
            with numpyro.plate("N", 100, subsample_size=10):
                X_batch = numpyro.subsample(X, event_dim=1)
                Y_batch = numpyro.subsample(Y, event_dim=0)
                numpyro.sample("obs", dist.Bernoulli(logits=theta @ X_batch.T), obs=Y_batch)

        # surrogate model defined by prior and surrogate likelihood.
        # a convenient choice for specifying the latter is to compute the likelihood on
        # a randomly chosen data subset (here {X_surr, Y_surr} of size 20) and then use
        # handlers.scale to scale the log likelihood by a vector of learnable weights.
        def surrogate_model(X_surr, Y_surr):
            theta = numpyro.sample(
                "theta", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
            )
            omegas = numpyro.param(
                "omegas", 5.0 * jnp.ones(20), constraint=dist.constraints.positive
            )
            with numpyro.plate("N", 20), numpyro.handlers.scale(scale=omegas):
                numpyro.sample("obs", dist.Bernoulli(logits=theta @ X_surr.T), obs=Y_surr)

        guide = AutoSurrogateLikelihoodDAIS(model, surrogate_model)
        svi = SVI(model, guide, ...)

    :param callable model: A NumPyro model.
    :param callable surrogate_model: A NumPyro model that is used as a surrogate model
        for guiding the HMC dynamics that define the variational distribution. In particular
        `surrogate_model` should contain the same prior as `model` but should contain a
        cheap-to-evaluate parametric ansatz for the likelihood. A simple ansatz for the latter
        involves computing the likelihood for a fixed subset of the data and scaling the resulting
        log likelihood by a learnable vector of positive weights. See the usage example above.
    :param str prefix: A prefix that will be prefixed to all param internal sites.
    :param int K: A positive integer that controls the number of HMC steps used.
        Defaults to 4.
    :param str base_dist: Controls whether the base Normal variational distribution
       is parameterized by a "diagonal" covariance matrix or a full-rank covariance
       matrix parameterized by a lower-triangular "cholesky" factor. Defaults to "diagonal".
    :param float eta_init: The initial value of the step size used in HMC. Defaults
        to 0.01.
    :param float eta_max: The maximum value of the learnable step size used in HMC.
        Defaults to 0.1.
    :param float gamma_init: The initial value of the learnable damping factor used
        during partial momentum refreshments in HMC. Defaults to 0.9.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of
        the base variational distribution for each (unconstrained transformed)
        latent variable. Defaults to 0.1.
    """

    def __init__(
        self,
        model,
        surrogate_model,
        *,
        K=4,
        eta_init=0.01,
        eta_max=0.1,
        gamma_init=0.9,
        prefix="auto",
        base_dist="diagonal",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
    ):
        super().__init__(
            model,
            K=K,
            eta_init=eta_init,
            eta_max=eta_max,
            gamma_init=gamma_init,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
            base_dist=base_dist,
        )

        self.surrogate_model = surrogate_model

    def _setup_prototype(self, *args, **kwargs):
        AutoContinuous._setup_prototype(self, *args, **kwargs)

        rng_key = numpyro.prng_key()

        with numpyro.handlers.block():
            (_, self._surrogate_potential_fn, _, _) = initialize_model(
                rng_key,
                self.surrogate_model,
                init_strategy=self.init_loc_fn,
                dynamic_args=False,
                model_args=(),
                model_kwargs={},
            )

    def _sample_latent(self, *args, **kwargs):
        def blocked_surrogate_model(x):
            x_unpack = self._unpack_latent(x)
            with numpyro.handlers.block(expose_types=["param"]):
                return -self._surrogate_potential_fn(x_unpack)

        eta0 = numpyro.param(
            "{}_eta0".format(self.prefix),
            self.eta_init,
            constraint=constraints.interval(0, self.eta_max),
        )
        eta_coeff = numpyro.param("{}_eta_coeff".format(self.prefix), 0.0)

        gamma = numpyro.param(
            "{}_gamma".format(self.prefix),
            self.gamma_init,
            constraint=constraints.interval(0, 1),
        )
        betas = numpyro.param(
            "{}_beta_increments".format(self.prefix),
            jnp.ones(self.K),
            constraint=constraints.positive,
        )
        betas = jnp.cumsum(betas)
        betas = betas / betas[-1]  # K-dimensional with betas[-1] = 1

        mass_matrix = numpyro.param(
            "{}_mass_matrix".format(self.prefix),
            jnp.ones(self.latent_dim),
            constraint=constraints.positive,
        )
        inv_mass_matrix = 0.5 / mass_matrix

        init_z_loc = numpyro.param("{}_z_0_loc".format(self.prefix), self._init_latent)

        if self.base_dist == "diagonal":
            init_z_scale = numpyro.param(
                "{}_z_0_scale".format(self.prefix),
                jnp.full(self.latent_dim, self._init_scale),
                constraint=constraints.positive,
            )
            base_z_dist = dist.Normal(init_z_loc, init_z_scale).to_event()
        else:
            scale_tril = numpyro.param(
                "{}_scale_tril".format(self.prefix),
                jnp.identity(self.latent_dim) * self._init_scale,
                constraint=constraints.scaled_unit_lower_cholesky,
            )
            base_z_dist = dist.MultivariateNormal(init_z_loc, scale_tril=scale_tril)

        z_0 = numpyro.sample(
            "{}_z_0".format(self.prefix), base_z_dist, infer={"is_auxiliary": True}
        )

        base_z_dist_log_prob = base_z_dist.log_prob

        momentum_dist = dist.Normal(0, mass_matrix).to_event()
        eps = numpyro.sample(
            "{}_momentum".format(self.prefix),
            momentum_dist.expand((self.K,)).to_event().mask(False),
            infer={"is_auxiliary": True},
        )

        def scan_body(carry, eps_beta):
            eps, beta = eps_beta
            eta = eta0 + eta_coeff * beta
            eta = jnp.clip(eta, 0.0, self.eta_max)
            z_prev, v_prev, log_factor = carry
            z_half = z_prev + v_prev * eta * inv_mass_matrix
            q_grad = (1.0 - beta) * grad(base_z_dist_log_prob)(z_half)
            p_grad = beta * grad(blocked_surrogate_model)(z_half)
            v_hat = v_prev + eta * (q_grad + p_grad)
            z = z_half + v_hat * eta * inv_mass_matrix
            v = gamma * v_hat + jnp.sqrt(1 - gamma**2) * eps
            delta_ke = momentum_dist.log_prob(v_prev) - momentum_dist.log_prob(v_hat)
            log_factor = log_factor + delta_ke
            return (z, v, log_factor), None

        v_0 = eps[-1]  # note the return value of scan doesn't depend on eps[-1]
        (z, _, log_factor), _ = jax.lax.scan(scan_body, (z_0, v_0, 0.0), (eps, betas))

        numpyro.factor("{}_factor".format(self.prefix), log_factor)

        return z


def _subsample_model(model, *args, **kwargs):
    data = kwargs.pop("_subsample_idx", {})
    with handlers.substitute(data=data):
        return model(*args, **kwargs)


class AutoSemiDAIS(AutoGuide):
    r"""
    This implementation of :class:`AutoSemiDAIS` [1] combines a parametric
    variational distribution over global latent variables with Differentiable
    Annealed Importance Sampling (DAIS) [2, 3] to infer local latent variables.
    Unlike :class:`AutoDAIS` this guide can be used in conjunction with data subsampling.
    Note that the resulting ELBO can be understood as a particular realization of a
    'locally enhanced bound' as described in reference [4].

    **References:**

    1. *Surrogate Likelihoods for Variational Annealed Importance Sampling*,
       Martin Jankowiak, Du Phan
    2. *MCMC Variational Inference via Uncorrected Hamiltonian Annealing*,
       Tomas Geffner, Justin Domke
    3. *Differentiable Annealed Importance Sampling and the Perils of Gradient Noise*,
       Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger Grosse
    4. *Variational Inference with Locally Enhanced Bounds for Hierarchical Models*,
       Tomas Geffner, Justin Domke

    Usage::

        def global_model():
            return numpyro.sample("theta", dist.Normal(0, 1))

        def local_model(theta):
            with numpyro.plate("data", 8, subsample_size=2):
                tau = numpyro.sample("tau", dist.Gamma(5.0, 5.0))
                numpyro.sample("obs", dist.Normal(0.0, tau), obs=jnp.ones(2))

        model = lambda: local_model(global_model())
        global_guide = AutoNormal(global_model)
        guide = AutoSemiDAIS(model, local_model, global_guide, K=4)
        svi = SVI(model, guide, ...)

        # sample posterior for particular data subset {3, 7}
        with handlers.substitute(data={"data": jnp.array([3, 7])}):
            samples = guide.sample_posterior(random.PRNGKey(1), params)

    :param callable model: A NumPyro model with global and local latent variables.
    :param callable local_model: The portion of `model` that includes the local latent variables only.
        The signature of `local_model` should be the return type of the global model with global latent
        variables only.
    :param callable global_guide: A guide for the global latent variables, e.g. an autoguide.
        The return type should be a dictionary of latent sample sites names and corresponding samples.
        If there is no global variable in the model, we can set this to None.
    :param callable local_guide: An optional guide for specifying the DAIS base distribution for
        local latent variables.
    :param str prefix: A prefix that will be prefixed to all internal sites.
    :param int K: A positive integer that controls the number of HMC steps used.
        Defaults to 4.
    :param float eta_init: The initial value of the step size used in HMC. Defaults
        to 0.01.
    :param float eta_max: The maximum value of the learnable step size used in HMC.
        Defaults to 0.1.
    :param float gamma_init: The initial value of the learnable damping factor used
        during partial momentum refreshments in HMC. Defaults to 0.9.
    :param float init_scale: Initial scale for the standard deviation of the variational
        distribution for each (unconstrained transformed) local latent variable. Defaults to 0.1.
    :param str subsample_plate: Optional name of the subsample plate site. This is required
        when the model has a subsample plate without `subsample_size` specified or
        the model has a subsample plate with `subsample_size` equal to the plate size.
    :param bool use_global_dais_params: Whether parameters controlling DAIS dynamic
        (HMC step size, HMC mass matrix, etc.) should be global (i.e. common to all
        data points in the subsample plate) or local (i.e. each data point in the
        subsample plate has individual parameters). Note that we do not use global
        parameters for the base distribution.
    """

    def __init__(
        self,
        model,
        local_model,
        global_guide=None,
        local_guide=None,
        *,
        prefix="auto",
        K=4,
        eta_init=0.01,
        eta_max=0.1,
        gamma_init=0.9,
        init_scale=0.1,
        subsample_plate=None,
        use_global_dais_params=False,
    ):
        # init_loc_fn is only used to inspect the model.
        super().__init__(model, prefix=prefix, init_loc_fn=init_to_uniform)
        if K < 1:
            raise ValueError("K must satisfy K >= 1 (got K = {})".format(K))
        if eta_init <= 0.0 or eta_init >= eta_max:
            raise ValueError(
                "eta_init must be positive and satisfy eta_init < eta_max."
            )
        if eta_max <= 0.0:
            raise ValueError("eta_max must be positive.")
        if gamma_init <= 0.0 or gamma_init >= 1.0:
            raise ValueError("gamma_init must be in the open interval (0, 1).")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")

        self.local_model = local_model
        self.global_guide = global_guide
        self.local_guide = local_guide
        self.eta_init = eta_init
        self.eta_max = eta_max
        self.gamma_init = gamma_init
        self.K = K
        self.init_scale = init_scale
        self.subsample_plate = subsample_plate
        self.use_global_dais_params = use_global_dais_params

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # extract global/local/local_dim/plates
        assert self.prototype_trace is not None
        subsample_plates = {
            name: site
            for name, site in self.prototype_trace.items()
            if site["type"] == "plate"
            and isinstance(site["args"][1], int)
            and site["args"][0] > site["args"][1]
        }
        if self.subsample_plate is not None:
            subsample_plates[self.subsample_plate] = self.prototype_trace[
                self.subsample_plate
            ]
        elif not subsample_plates:
            # Consider all plates as subsample plates.
            subsample_plates = {
                name: site
                for name, site in self.prototype_trace.items()
                if site["type"] == "plate"
            }
        num_plates = len(subsample_plates)
        assert (
            num_plates == 1
        ), f"AutoSemiDAIS assumes that the model contains exactly 1 plate with data subsampling but got {num_plates}."
        plate_name = list(subsample_plates.keys())[0]
        local_vars = []
        subsample_axes = {}
        plate_dim = None
        for name, site in self.prototype_trace.items():
            if site["type"] == "sample" and not site["is_observed"]:
                for frame in site["cond_indep_stack"]:
                    if frame.name == plate_name:
                        if plate_dim is None:
                            plate_dim = frame.dim
                        local_vars.append(name)
                        subsample_axes[name] = plate_dim - site["fn"].event_dim
                        break
        if len(local_vars) == 0:
            raise RuntimeError(
                "There are no local variables in the `{plate_name}` plate."
                " AutoSemiDAIS is appropriate for models with local variables."
            )

        local_init_locs = {
            name: value for name, value in self._init_locs.items() if name in local_vars
        }

        one_sample = {
            k: jnp.take(v, 0, axis=subsample_axes[k])
            for k, v in local_init_locs.items()
        }
        _, shape_dict = _ravel_dict(one_sample)
        self._pack_local_latent = jax.vmap(
            lambda x: _ravel_dict(x)[0], in_axes=(subsample_axes,)
        )
        local_init_latent = self._pack_local_latent(local_init_locs)
        unpack_latent = partial(_unravel_dict, shape_dict=shape_dict)
        # this is to match the behavior of Pyro, where we can apply
        # unpack_latent for a batch of samples
        self._unpack_local_latent = jax.vmap(
            UnpackTransform(unpack_latent), out_axes=subsample_axes
        )
        plate_full_size, plate_subsample_size = subsample_plates[plate_name]["args"]
        if plate_subsample_size is None:
            plate_subsample_size = plate_full_size
        self._local_latent_dim = jnp.size(local_init_latent) // plate_subsample_size
        self._local_plate = (plate_name, plate_full_size, plate_subsample_size)

        if self.global_guide is not None:
            with handlers.block(), handlers.seed(rng_seed=0):
                local_args = (self.global_guide.model(*args, **kwargs),)
                local_kwargs = {}
        else:
            local_args = args
            local_kwargs = kwargs.copy()

        if self.local_guide is not None:
            with handlers.block(), handlers.trace() as tr, handlers.seed(rng_seed=0):
                self.local_guide(*local_args, **local_kwargs)
            self.prototype_local_guide_trace = tr

        with handlers.block(), handlers.trace() as tr, handlers.seed(rng_seed=0):
            self.local_model(*local_args, **local_kwargs)
        self.prototype_local_model_trace = tr

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        global_latents, local_latent_flat = self._sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = global_latents.copy()
        _, N, subsample_size = self._local_plate

        for name, unconstrained_value in self._unpack_local_latent(
            local_latent_flat
        ).items():
            site = self.prototype_trace[name]
            with helpful_support_errors(site):
                transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            event_ndim = site["fn"].event_dim
            if numpyro.get_mask() is False:
                log_density = 0.0
            else:
                log_density = -transform.log_abs_det_jacobian(
                    unconstrained_value, value
                )
                log_density = (N / subsample_size) * sum_rightmost(
                    log_density, jnp.ndim(log_density) - jnp.ndim(value) + event_ndim
                )
            delta_dist = dist.Delta(
                value, log_density=log_density, event_dim=event_ndim
            )
            result[name] = numpyro.sample(name, delta_dist)

        return result

    def _get_posterior(self):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        kwargs.pop("sample_shape", ())

        if self.global_guide is not None:
            global_latents = self.global_guide(*args, **kwargs)
            rng_key = numpyro.prng_key()
            with handlers.block(), handlers.seed(rng_seed=rng_key), handlers.substitute(
                data=global_latents
            ):
                global_outputs = self.global_guide.model(*args, **kwargs)
            local_args = (global_outputs,)
            local_kwargs = {}
        else:
            global_latents = {}
            local_args = args
            local_kwargs = kwargs.copy()

        local_guide_params = {}
        if self.local_guide is not None:
            for name, site in self.prototype_local_guide_trace.items():
                if site["type"] == "param":
                    local_guide_params[name] = numpyro.param(
                        name, site["value"], **site["kwargs"]
                    )

        local_model_params = {}
        for name, site in self.prototype_local_model_trace.items():
            if site["type"] == "param":
                local_model_params[name] = numpyro.param(
                    name, site["value"], **site["kwargs"]
                )

        def make_local_log_density(*local_args, **local_kwargs):
            def fn(x):
                x_unpack = self._unpack_local_latent(x)
                with numpyro.handlers.block():
                    return -potential_energy(
                        partial(_subsample_model, self.local_model),
                        local_args,
                        local_kwargs,
                        {**x_unpack, **local_model_params},
                    )

            return fn

        plate_name, N, subsample_size = self._local_plate
        D, K = self._local_latent_dim, self.K

        with numpyro.plate(plate_name, N, subsample_size=subsample_size) as idx:
            if self.use_global_dais_params:
                eta0 = numpyro.param(
                    "{}_eta0".format(self.prefix),
                    self.eta_init,
                    constraint=constraints.interval(0, self.eta_max),
                )
                eta0 = jnp.broadcast_to(eta0, idx.shape)
                eta_coeff = numpyro.param(
                    "{}_eta_coeff".format(self.prefix),
                    0.0,
                )
                eta_coeff = jnp.broadcast_to(eta_coeff, idx.shape)
                gamma = numpyro.param(
                    "{}_gamma".format(self.prefix),
                    0.9,
                    constraint=constraints.interval(0, 1),
                )
                gamma = jnp.broadcast_to(gamma, idx.shape)
                betas = numpyro.param(
                    "{}_beta_increments".format(self.prefix),
                    jnp.ones(K),
                    constraint=constraints.positive,
                )
                betas = jnp.broadcast_to(betas, idx.shape + (K,))
                mass_matrix = numpyro.param(
                    "{}_mass_matrix".format(self.prefix),
                    jnp.ones(D),
                    constraint=constraints.positive,
                )
                mass_matrix = jnp.broadcast_to(mass_matrix, idx.shape + (D,))
            else:
                eta0 = numpyro.param(
                    "{}_eta0".format(self.prefix),
                    jnp.ones(N) * self.eta_init,
                    constraint=constraints.interval(0, self.eta_max),
                    event_dim=0,
                )
                eta_coeff = numpyro.param(
                    "{}_eta_coeff".format(self.prefix), jnp.zeros(N), event_dim=0
                )
                gamma = numpyro.param(
                    "{}_gamma".format(self.prefix),
                    jnp.ones(N) * 0.9,
                    constraint=constraints.interval(0, 1),
                    event_dim=0,
                )
                betas = numpyro.param(
                    "{}_beta_increments".format(self.prefix),
                    jnp.ones((N, K)),
                    constraint=constraints.positive,
                    event_dim=1,
                )
                mass_matrix = numpyro.param(
                    "{}_mass_matrix".format(self.prefix),
                    jnp.ones((N, D)),
                    constraint=constraints.positive,
                    event_dim=1,
                )

            betas = jnp.cumsum(betas, axis=-1)
            betas = betas / betas[..., -1:]

            inv_mass_matrix = 0.5 / mass_matrix
            assert inv_mass_matrix.shape == (subsample_size, D)

            local_kwargs["_subsample_idx"] = {plate_name: idx}
            if self.local_guide is not None:
                key = numpyro.prng_key()
                subsample_guide = partial(_subsample_model, self.local_guide)
                with handlers.block(), handlers.trace() as tr, handlers.seed(
                    rng_seed=key
                ), handlers.substitute(data=local_guide_params):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        subsample_guide(*local_args, **local_kwargs)
                    latent = {
                        name: biject_to(site["fn"].support).inv(site["value"])
                        for name, site in tr.items()
                        if site["type"] == "sample"
                        and not site.get("is_observed", False)
                    }
                    z_0 = self._pack_local_latent(latent)

                def base_z_dist_log_prob(z):
                    latent = self._unpack_local_latent(z)
                    assert isinstance(latent, dict)
                    with handlers.block():
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scale = N / subsample_size
                            return (
                                -potential_energy(
                                    subsample_guide,
                                    local_args,
                                    local_kwargs,
                                    {**local_guide_params, **latent},
                                )
                                / scale
                            )

                # The log_prob of z_0 will be broadcasted to `subsample_size` because this statement
                # is run under the subsample plate. Hence we divide the log_prob by `subsample_size`.
                numpyro.factor(
                    "{}_z_0_factor".format(self.prefix),
                    base_z_dist_log_prob(z_0) / subsample_size,
                )
            else:
                z_0_loc_init = jnp.zeros((N, D))
                z_0_loc = numpyro.param(
                    "{}_z_0_loc".format(self.prefix), z_0_loc_init, event_dim=1
                )
                z_0_scale_init = jnp.ones((N, D)) * self.init_scale
                z_0_scale = numpyro.param(
                    "{}_z_0_scale".format(self.prefix),
                    z_0_scale_init,
                    constraint=constraints.positive,
                    event_dim=1,
                )
                base_z_dist = dist.Normal(z_0_loc, z_0_scale).to_event(1)
                assert base_z_dist.shape() == (subsample_size, D)
                z_0 = numpyro.sample(
                    "{}_z_0".format(self.prefix),
                    base_z_dist,
                    infer={"is_auxiliary": True},
                )

                def base_z_dist_log_prob(x):
                    return base_z_dist.log_prob(x).sum()

            momentum_dist = dist.Normal(0, mass_matrix).to_event(1)
            eps = numpyro.sample(
                "{}_momentum".format(self.prefix),
                dist.Normal(0, mass_matrix[..., None])
                .expand([subsample_size, D, K])
                .to_event(2)
                .mask(False),
                infer={"is_auxiliary": True},
            )

            local_log_density = make_local_log_density(*local_args, **local_kwargs)

            def scan_body(carry, eps_beta):
                eps, beta = eps_beta
                eta = eta0 + eta_coeff * beta
                eta = jnp.clip(eta, 0.0, self.eta_max)
                assert eps.shape == (subsample_size, D)
                assert eta.shape == beta.shape == (subsample_size,)
                z_prev, v_prev, log_factor = carry
                z_half = z_prev + v_prev * eta[:, None] * inv_mass_matrix
                q_grad = (1.0 - beta[:, None]) * grad(base_z_dist_log_prob)(z_half)
                p_grad = (
                    beta[:, None]
                    * (subsample_size / N)
                    * grad(local_log_density)(z_half)
                )
                assert q_grad.shape == p_grad.shape == (subsample_size, D)
                v_hat = v_prev + eta[:, None] * (q_grad + p_grad)
                z = z_half + v_hat * eta[:, None] * inv_mass_matrix
                v = gamma[:, None] * v_hat + jnp.sqrt(1 - gamma[:, None] ** 2) * eps
                delta_ke = momentum_dist.log_prob(v_prev) - momentum_dist.log_prob(
                    v_hat
                )
                assert delta_ke.shape == (subsample_size,)
                log_factor = log_factor + delta_ke
                return (z, v, log_factor), None

            v_0 = eps[
                :, :, -1
            ]  # note the return value of scan doesn't depend on eps[:, :, -1]
            assert eps.shape == (subsample_size, D, K)
            assert betas.shape == (subsample_size, K)

            eps_T = jnp.moveaxis(eps, -1, 0)
            (z, _, log_factor), _ = jax.lax.scan(
                scan_body, (z_0, v_0, jnp.zeros(subsample_size)), (eps_T, betas.T)
            )
            assert log_factor.shape == (subsample_size,)

            numpyro.factor("{}_local_dais_factor".format(self.prefix), log_factor)
            return global_latents, z

    def sample_posterior(self, rng_key, params, *args, sample_shape=(), **kwargs):
        def _single_sample(_rng_key):
            global_latents, local_flat = handlers.substitute(
                handlers.seed(self._sample_latent, _rng_key), params
            )(*args, **kwargs)
            results = global_latents.copy()
            for name, unconstrained_value in self._unpack_local_latent(
                local_flat
            ).items():
                site = self.prototype_trace[name]
                transform = biject_to(site["fn"].support)
                value = transform(unconstrained_value)
                results[name] = value
            return results

        if sample_shape:
            rng_key = random.split(rng_key, int(np.prod(sample_shape)))
            samples = lax.map(_single_sample, rng_key)
            return jax.tree.map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                samples,
            )
        else:
            return _single_sample(rng_key)


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(model, ...)
        svi = SVI(model, guide, ...)
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _get_posterior(self):
        loc = numpyro.param("{}_loc".format(self.prefix), self._init_latent)
        scale = numpyro.param(
            "{}_scale".format(self.prefix),
            jnp.full(self.latent_dim, self._init_scale),
            constraint=self.scale_constraint,
        )
        return dist.Normal(loc, scale)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        loc = params["{}_loc".format(self.prefix)]
        scale = params["{}_scale".format(self.prefix)]
        return IndependentTransform(AffineTransform(loc, scale), 1)

    def get_posterior(self, params):
        """
        Returns a diagonal Normal posterior distribution.
        """
        transform = self.get_transform(params).base_transform
        return dist.Normal(transform.loc, transform.scale)

    def median(self, params):
        loc = params["{}_loc".format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        quantiles = jnp.array(quantiles)[..., None]
        latent = self.get_posterior(params).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a MultivariateNormal
    distribution to construct a guide over the entire latent space.
    The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoMultivariateNormal(model, ...)
        svi = SVI(model, guide, ...)
    """

    scale_tril_constraint = constraints.scaled_unit_lower_cholesky

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _get_posterior(self):
        loc = numpyro.param("{}_loc".format(self.prefix), self._init_latent)
        scale_tril = numpyro.param(
            "{}_scale_tril".format(self.prefix),
            jnp.identity(self.latent_dim) * self._init_scale,
            constraint=self.scale_tril_constraint,
        )
        return dist.MultivariateNormal(loc, scale_tril=scale_tril)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        loc = params["{}_loc".format(self.prefix)]
        scale_tril = params["{}_scale_tril".format(self.prefix)]
        return LowerCholeskyAffine(loc, scale_tril)

    def get_posterior(self, params):
        """
        Returns a multivariate Normal posterior distribution.
        """
        transform = self.get_transform(params)
        return dist.MultivariateNormal(transform.loc, scale_tril=transform.scale_tril)

    def median(self, params):
        loc = params["{}_loc".format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        transform = self.get_transform(params)
        quantiles = jnp.array(quantiles)[..., None]
        latent = dist.Normal(
            transform.loc, jnp.linalg.norm(transform.scale_tril, axis=-1)
        ).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoBatchedMixin:
    """
    Mixin to infer the batch and event shapes of batched auto guides.
    """

    # Available from AutoContinuous.
    latent_dim: int

    def __init__(self, *args, **kwargs):
        self._batch_shape = None
        self._event_shape = None
        # Pop the number of batch dimensions and pass the rest to the other constructor.
        self.batch_ndim = kwargs.pop("batch_ndim")
        super().__init__(*args, **kwargs)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        # Extract the batch shape.
        batch_shape = None
        for site in self.prototype_trace.values():
            if site["type"] == "sample" and not site["is_observed"]:
                shape = site["value"].shape
                if site["value"].ndim < self.batch_ndim + site["fn"].event_dim:
                    raise ValueError(
                        f"Expected {self.batch_ndim} batch dimensions, but site "
                        f"`{site['name']}` only has shape {shape}."
                    )
                shape = shape[: self.batch_ndim]
                if batch_shape is None:
                    batch_shape = shape
                elif shape != batch_shape:
                    raise ValueError("Encountered inconsistent batch shapes.")
        self._batch_shape = batch_shape

        # Save the event shape of the non-batched part. This will always be a vector.
        batch_size = math.prod(self._batch_shape)
        if self.latent_dim % batch_size:
            raise RuntimeError(
                f"Incompatible batch shape {batch_shape} (size {batch_size}) and "
                f"latent dims {self.latent_dim}."
            )
        self._event_shape = (self.latent_dim // batch_size,)

    def _get_batched_posterior(self):
        raise NotImplementedError

    def _get_posterior(self):
        return dist.TransformedDistribution(
            self._get_batched_posterior(),
            self._get_reshape_transform(),
        )

    def _get_reshape_transform(self) -> ReshapeTransform:
        return ReshapeTransform(
            (self.latent_dim,), self._batch_shape + self._event_shape
        )


class AutoBatchedMultivariateNormal(AutoBatchedMixin, AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a batched MultivariateNormal
    distribution to construct a guide over the entire latent space.
    The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoBatchedMultivariateNormal(model, batch_ndim=1, ...)
        svi = SVI(model, guide, ...)
    """

    scale_tril_constraint = constraints.scaled_unit_lower_cholesky

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
        batch_ndim=1,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(
            model,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            batch_ndim=batch_ndim,
        )

    def _get_batched_posterior(self):
        init_latent = self._init_latent.reshape(self._batch_shape + self._event_shape)
        loc = numpyro.param("{}_loc".format(self.prefix), init_latent)
        init_scale = (
            jnp.ones(self._batch_shape + (1, 1))
            * jnp.identity(init_latent.shape[-1])
            * self._init_scale
        )
        scale_tril = numpyro.param(
            "{}_scale_tril".format(self.prefix),
            init_scale,
            constraint=self.scale_tril_constraint,
        )
        return dist.MultivariateNormal(loc, scale_tril=scale_tril)

    def median(self, params):
        loc = self._get_reshape_transform()(params["{}_loc".format(self.prefix)])
        return self._unpack_and_constrain(loc, params)


class AutoLowRankMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a LowRankMultivariateNormal
    distribution to construct a guide over the entire latent space.
    The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoLowRankMultivariateNormal(model, rank=2, ...)
        svi = SVI(model, guide, ...)
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
        rank=None,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.rank = rank
        super(AutoLowRankMultivariateNormal, self).__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn
        )

    def _get_posterior(self, *args, **kwargs):
        rank = int(round(self.latent_dim**0.5)) if self.rank is None else self.rank
        loc = numpyro.param("{}_loc".format(self.prefix), self._init_latent)
        cov_factor = numpyro.param(
            "{}_cov_factor".format(self.prefix), jnp.zeros((self.latent_dim, rank))
        )
        scale = numpyro.param(
            "{}_scale".format(self.prefix),
            jnp.full(self.latent_dim, self._init_scale),
            constraint=self.scale_constraint,
        )
        cov_diag = scale * scale
        cov_factor = cov_factor * scale[..., None]
        return dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        posterior = self.get_posterior(params)
        return LowerCholeskyAffine(posterior.loc, posterior.scale_tril)

    def get_posterior(self, params):
        """
        Returns a lowrank multivariate Normal posterior distribution.
        """
        loc = params["{}_loc".format(self.prefix)]
        cov_factor = params["{}_cov_factor".format(self.prefix)]
        scale = params["{}_scale".format(self.prefix)]
        cov_diag = scale * scale
        cov_factor = cov_factor * scale[..., None]
        return dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def median(self, params):
        loc = params["{}_loc".format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        loc = params[f"{self.prefix}_loc"]
        cov_factor = params[f"{self.prefix}_cov_factor"]
        scale = params[f"{self.prefix}_scale"]
        scale = scale * jnp.sqrt(jnp.square(cov_factor).sum(-1) + 1)
        quantiles = jnp.array(quantiles)[..., None]
        latent = dist.Normal(loc, scale).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoBatchedLowRankMultivariateNormal(AutoBatchedMixin, AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a batched
    AutoLowRankMultivariateNormal distribution to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoBatchedLowRankMultivariateNormal(model, batch_ndim=1, ...)
        svi = SVI(model, guide, ...)
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=0.1,
        rank=None,
        batch_ndim=1,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.rank = rank
        super().__init__(
            model,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            batch_ndim=batch_ndim,
        )

    def _get_batched_posterior(self):
        rank = (
            int(round(self._event_shape[0] ** 0.5)) if self.rank is None else self.rank
        )
        init_latent = self._init_latent.reshape(self._batch_shape + self._event_shape)
        loc = numpyro.param("{}_loc".format(self.prefix), init_latent)
        cov_factor = numpyro.param(
            "{}_cov_factor".format(self.prefix),
            jnp.zeros(self._batch_shape + self._event_shape + (rank,)),
        )
        scale = numpyro.param(
            "{}_scale".format(self.prefix),
            jnp.full(self._batch_shape + self._event_shape, self._init_scale),
            constraint=self.scale_constraint,
        )
        cov_diag = scale * scale
        cov_factor = cov_factor * scale[..., None]
        return dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def median(self, params):
        loc = self._get_reshape_transform()(params["{}_loc".format(self.prefix)])
        return self._unpack_and_constrain(loc, params)


class AutoLaplaceApproximation(AutoContinuous):
    r"""
    Laplace approximation (quadratic approximation) approximates the posterior
    :math:`\log p(z | x)` by a multivariate normal distribution in the
    unconstrained space. Under the hood, it uses Delta distributions to
    construct a MAP (i.e. point estimate) guide over the entire (unconstrained) latent
    space. Its covariance is given by the inverse of the hessian of :math:`-\log p(x, z)`
    at the MAP point of `z`.

    Usage::

        guide = AutoLaplaceApproximation(model, ...)
        svi = SVI(model, guide, ...)

    :param callable hessian_fn: EXPERIMENTAL a function that takes a function `f`
        and a vector `x`and returns the hessian of `f` at `x`. By default, we use
        ``lambda f, x: jax.hessian(f)(x)``. Other alternatives can be
        ``lambda f, x: jax.jacobian(jax.jacobian(f))(x)`` or
        ``lambda f, x: jax.hessian(f)(x) + 1e-3 * jnp.eye(x.shape[0])``. The later
        example is helpful when the hessian of `f` at `x` is not positive definite.
        Note that the output hessian is the precision matrix of the laplace
        approximation.
    """

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        create_plates=None,
        hessian_fn=None,
    ):
        super().__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates
        )
        self._hessian_fn = (
            hessian_fn if hessian_fn is not None else (lambda f, x: hessian(f)(x))
        )

    def _setup_prototype(self, *args, **kwargs):
        super(AutoLaplaceApproximation, self)._setup_prototype(*args, **kwargs)

        def loss_fn(params):
            # we are doing maximum likelihood, so only require `num_particles=1` and an arbitrary rng_key.
            return Trace_ELBO().loss(
                random.PRNGKey(0), params, self.model, self, *args, **kwargs
            )

        self._loss_fn = loss_fn

    def _get_posterior(self, *args, **kwargs):
        # sample from Delta guide
        loc = numpyro.param("{}_loc".format(self.prefix), self._init_latent)
        return dist.Delta(loc, event_dim=1)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        def loss_fn(z):
            params1 = params.copy()
            params1["{}_loc".format(self.prefix)] = z
            return self._loss_fn(params1)

        loc = params["{}_loc".format(self.prefix)]
        precision = self._hessian_fn(loss_fn, loc)
        scale_tril = cholesky_of_inverse(precision)
        if not_jax_tracer(scale_tril):
            if np.any(np.isnan(scale_tril)):
                warnings.warn(
                    "Hessian of log posterior at the MAP point is singular. Posterior"
                    " samples from AutoLaplaceApproxmiation will be constant (equal to"
                    " the MAP point). Please consider using an AutoNormal guide.",
                    stacklevel=find_stack_level(),
                )
        scale_tril = jnp.where(jnp.isnan(scale_tril), 0.0, scale_tril)
        return LowerCholeskyAffine(loc, scale_tril)

    def get_posterior(self, params):
        """
        Returns a multivariate Normal posterior distribution.
        """
        transform = self.get_transform(params)
        return dist.MultivariateNormal(transform.loc, scale_tril=transform.scale_tril)

    def sample_posterior(self, rng_key, params, *, sample_shape=()):
        latent_sample = self.get_posterior(params).sample(rng_key, sample_shape)
        return self._unpack_and_constrain(latent_sample, params)

    def median(self, params):
        loc = params["{}_loc".format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        transform = self.get_transform(params)
        quantiles = jnp.array(quantiles)[..., None]
        latent = dist.Normal(
            transform.loc, jnp.linalg.norm(transform.scale_tril, axis=-1)
        ).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoIAFNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a
    :class:`~numpyro.distributions.flows.InverseAutoregressiveTransform`
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoIAFNormal(model, hidden_dims=[20], skip_connections=True, ...)
        svi = SVI(model, guide, ...)

    :param callable model: a generative model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param list hidden_dims: the dimensionality of the hidden units per layer.
        Defaults to ``[latent_dim, latent_dim]``.
    :param bool skip_connections: whether to add skip connections from the input to the
        output of each flow. Defaults to False.
    :param callable nonlinearity: the nonlinearity to use in the feedforward network.
        Defaults to :func:`jax.example_libraries.stax.Elu`.
    """

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        num_flows=3,
        hidden_dims=None,
        skip_connections=False,
        nonlinearity=stax.Elu,
    ):
        self.num_flows = num_flows
        # 2-layer, stax.Elu, skip_connections=False by default following the experiments in
        # IAF paper (https://arxiv.org/abs/1606.04934)
        # and Neutra paper (https://arxiv.org/abs/1903.03704)
        self._hidden_dims = hidden_dims
        self._skip_connections = skip_connections
        self._nonlinearity = nonlinearity
        super(AutoIAFNormal, self).__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn
        )

    def _get_posterior(self):
        if self.latent_dim == 1:
            raise ValueError(
                "latent dim = 1. Consider using AutoDiagonalNormal instead"
            )
        hidden_dims = (
            [self.latent_dim, self.latent_dim]
            if self._hidden_dims is None
            else self._hidden_dims
        )
        flows = []
        for i in range(self.num_flows):
            if i > 0:
                flows.append(PermuteTransform(jnp.arange(self.latent_dim)[::-1]))
            arn = AutoregressiveNN(
                self.latent_dim,
                hidden_dims,
                permutation=jnp.arange(self.latent_dim),
                skip_connections=self._skip_connections,
                nonlinearity=self._nonlinearity,
            )
            arnn = numpyro.module(
                "{}_arn__{}".format(self.prefix, i), arn, (self.latent_dim,)
            )
            flows.append(InverseAutoregressiveTransform(arnn))
        return dist.TransformedDistribution(self.get_base_dist(), flows)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)


class AutoBNAFNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a
    :class:`~numpyro.distributions.flows.BlockNeuralAutoregressiveTransform`
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoBNAFNormal(model, num_flows=1, hidden_factors=[50, 50], ...)
        svi = SVI(model, guide, ...)

    **References**

    1. *Block Neural Autoregressive Flow*,
       Nicola De Cao, Ivan Titov, Wilker Aziz

    :param callable model: a generative model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 1.
    :param list hidden_factors: Hidden layer i has ``hidden_factors[i]`` hidden units per
        input dimension. This corresponds to both :math:`a` and :math:`b` in reference [1].
        The elements of hidden_factors must be integers.
    """

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        num_flows=1,
        hidden_factors=[8, 8],
    ):
        self.num_flows = num_flows
        self._hidden_factors = hidden_factors
        super(AutoBNAFNormal, self).__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn
        )

    def _get_posterior(self):
        if self.latent_dim == 1:
            raise ValueError(
                "latent dim = 1. Consider using AutoDiagonalNormal instead"
            )
        flows = []
        for i in range(self.num_flows):
            if i > 0:
                flows.append(PermuteTransform(jnp.arange(self.latent_dim)[::-1]))
            residual = "gated" if i < (self.num_flows - 1) else None
            arn = BlockNeuralAutoregressiveNN(
                self.latent_dim, self._hidden_factors, residual
            )
            arnn = numpyro.module(
                "{}_arn__{}".format(self.prefix, i), arn, (self.latent_dim,)
            )
            flows.append(BlockNeuralAutoregressiveTransform(arnn))
        return dist.TransformedDistribution(self.get_base_dist(), flows)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)
