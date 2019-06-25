# Adapted from pyro.contrib.autoguide

import scipy.stats as osp

from jax.flatten_util import ravel_pytree
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import biject_to
from numpyro.distributions.util import sum_rightmost
from numpyro.handlers import sample, substitute, trace, param


class AutoGuide(object):
    """
    Base class for automatic guides.

    Derived classes must implement the :meth:`__call__` method.

    :param callable model: a pyro model
    :param str prefix: a prefix that will be prefixed to all param internal sites
    """

    def __init__(self, model, get_params_fn, prefix="auto"):
        self.model = model
        self.get_params = get_params_fn
        self.prefix = prefix
        self.prototype_trace = None

    def __call__(self, *args, **kwargs):
        """
        A guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        raise NotImplementedError

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        raise NotImplementedError

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        self.prototype_trace = trace(self.model).get_trace(*args, **kwargs)

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        raise NotImplementedError


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].

    Each derived class implements its own :meth:`get_posterior` method.

    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.

    :param callable model: a Pyro model

    Reference:

    [1] `Automatic Differentiation Variational Inference`,
        Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M.
        Blei

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """
    def __init__(self, model, get_params_fn, prefix="auto", init_loc_fn=init_to_median):
        model = substitute(model, substitute_fn=init_loc_fn)
        super(AutoContinuous, self).__init__(model, get_params_fn, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        self._unconstrained_values = {}
        self._inv_transforms = {}
        for name, site in self.prototype_trace.items():
            if site['type'] == 'sample':
                # Collect the shapes of unconstrained values.
                # These may differ from the shapes of constrained values.
                transform = biject_to(site['fn'].support)
                unconstrained_val = biject_to(site['fn'].support).inv(site['value'])
                self._unconstrained_values[name] = unconstrained_val

        self.latent_size = sum(np.size(x) for x in self._unconstrained_values.values())
        if self.latent_size == 0:
            raise RuntimeError('{} found no latent variables; Use an empty guide instead'.format(type(self).__name__))
        _, self.unravel_fn = ravel_pytree(self._unconstrained_values)

    def _init_loc(self):
        """
        Creates an initial latent vector using a per-site init function.
        """
        return ravel_pytree(self._unconstrained_values)[0]

    def get_posterior(self, *args, **kwargs):
        """
        Returns the posterior distribution.
        """
        raise NotImplementedError

    def sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        pos_dist = self.get_posterior(*args, **kwargs)
        return sample("_{}_latent".format(self.prefix), pos_dist)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self.sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = {}

        for site, unconstrained_value in self.unravel_fn(latent):
            name = site["name"]
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
            log_density = sum_rightmost(log_density, np.ndim(log_density) - np.ndim(value) +
                                        len(site['fn'].event_shape))
            delta_dist = dist.Delta(value, log_density=log_density, event_ndim=len(site['fn'].event_shape))
            result[name] = sample(name, delta_dist)

        return result

    def _loc_scale(self, *args, **kwargs):
        """
        :returns: a tuple ``(loc, scale)`` used by :meth:`median` and
            :meth:`quantiles`
        """
        raise NotImplementedError

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = self._loc_scale(*args, **kwargs)
        return {site["name"]: biject_to(site["fn"].support)(unconstrained_value)
                for site, unconstrained_value in self.unravel_fn(loc)}

    def quantiles(self, opt_state, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        loc, scale = self._loc_scale(opt_state, *args, **kwargs)
        latents = osp.norm(loc, scale).ppf(quantiles)
        result = {}
        for latent in latents:
            for site, unconstrained_value in self.unravel_fn(latent):
                result.setdefault(site["name"], []).append(biject_to(site["fn"].support)(unconstrained_value))
        return result


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(model, get_params, ..)
        params = guide.get_optimizable_params(*args, **kwargs)
        svi_init, svi_update, _ = svi(model, guide, ...)
        opt_state, constrain_fn = svi_init(rng, model_args, guide_args, params)
    """
    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution.
        """
        loc = param('{}_loc'.format(self.prefix), self._init_loc)
        scale = param('{}_scale'.format(self.prefix), np.ones(self.latent_size),
                      constraint=constraints.positive)
        # TODO: Support for `.to_event()` to treat the batch dim as event dim.
        return dist.Normal(loc, scale)

    def _loc_scale(self, opt_state, *args, **kwargs):
        params = self.get_params(opt_state)
        loc = params["{}_loc".format(self.prefix)]
        scale = params["{}_scale".format(self.prefix)]
        return loc, scale
