# Adapted from pyro.contrib.autoguide

import numpy as onp
import scipy.stats as osp
from jax import vmap

from jax.flatten_util import ravel_pytree
import jax.numpy as np

import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import biject_to
from numpyro.distributions.util import sum_rightmost
from numpyro.handlers import block, param, sample, seed, substitute, trace

__all__ = [
    'AutoContinuous',
    'AutoGuide',
    'AutoDiagonalNormal',
    'init_to_feasible',
    'init_to_median',
]


def init_to_feasible(site):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    if site['is_observed']:
        return None
    value = sample('_init', site['fn'])
    t = biject_to(site['fn'].support)
    return t(np.zeros(np.shape(t.inv(value))))


def init_to_median(site, num_samples=15):
    """
    Initialize to the prior median; fallback to a feasible point if median is
    undefined.
    """
    if site['is_observed']:
        return None
    try:
        # Try to compute empirical median.
        samples = sample('_init', site['fn'], sample_shape=(num_samples,))
        value = onp.median(samples, axis=0)
        if np.isnan(value):
            raise ValueError
        return value
    except ValueError:
        # Fall back to feasible point.
        return init_to_feasible(site)


class AutoGuide(object):
    """
    Base class for automatic guides.

    Derived classes must implement the :meth:`__call__` method.

    :param callable model: a pyro model
    :param str prefix: a prefix that will be prefixed to all param internal sites
    """

    def __init__(self, model, get_params_fn, prefix='auto'):
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

    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        """
        Generate samples from the approximate posterior over the latent
        sites in the model.

        :param jax.random.PRNGKey rng: PRNG seed.
        :param opt_state: Current state of the optmizer.
        :param sample_shape: (keyword argument) shape of samples to be drawn.
        :return: batch of samples from the approximate posterior.
        """
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        """
        Samples an encoded latent given the same ``*args, **kwargs`` as the
        base ``model``.
        """
        raise NotImplementedError

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        self.prototype_trace = block(trace(self.model).get_trace)(*args, **kwargs)


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
    def __init__(self, rng, model, get_params_fn, prefix="auto", init_loc_fn=init_to_median):
        model = substitute(model, substitute_fn=block(seed(init_loc_fn, rng)))
        super(AutoContinuous, self).__init__(model, get_params_fn, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        self._unconstrained_values = {}
        self._inv_transforms = {}
        for name, site in self.prototype_trace.items():
            if site['type'] == 'sample' and not site['is_observed']:
                # Collect the shapes of unconstrained values.
                # These may differ from the shapes of constrained values.
                transform = biject_to(site['fn'].support)
                unconstrained_val = transform.inv(site['value'])
                self._inv_transforms[name] = transform
                self._unconstrained_values[name] = unconstrained_val

        self.latent_size = sum(np.size(x) for x in self._unconstrained_values.values())
        if self.latent_size == 0:
            raise RuntimeError('{} found no latent variables; Use an empty guide instead'.format(type(self).__name__))
        _, self.unravel_fn = ravel_pytree(self._unconstrained_values)

    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # run model to inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        latent = self._sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = {}

        for name, unconstrained_value in self.unravel_fn(latent).items():
            transform = self._inv_transforms[name]
            site = self.prototype_trace[name]
            value = transform(unconstrained_value)
            log_density = - transform.log_abs_det_jacobian(unconstrained_value, value)
            log_density = sum_rightmost(log_density, np.ndim(log_density) - np.ndim(value) +
                                        len(site['fn'].event_shape))
            delta_dist = dist.Delta(value, log_density=log_density, event_ndim=len(site['fn'].event_shape))
            result[name] = sample(name, delta_dist)

        return result


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(rng, model, get_params, ..)
        svi_init, svi_update, _ = svi(model, guide, ...)
    """
    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        loc, scale = self._loc_scale(opt_state)
        num_samples = int(np.prod(sample_shape))
        latent_sample = dist.Normal(loc, scale).sample(rng, sample_shape)
        if not sample_shape:
            unpacked_samples = self.unravel_fn(latent_sample)
        else:
            latent_sample = np.reshape(latent_sample, (num_samples,) +
                                       np.shape(latent_sample)[len(sample_shape):])
            unpacked_samples = vmap(self.unravel_fn)(latent_sample)
            unpacked_samples = {name: np.reshape(val, sample_shape + np.shape(val)[1:])
                                for name, val in unpacked_samples.items()}
        return {name: self._inv_transforms[name](unconstrained_value)
                for name, unconstrained_value in unpacked_samples.items()}

    def _sample_latent(self, *args, **kwargs):
        init_loc = ravel_pytree(self._unconstrained_values)[0]
        loc = param('{}_loc'.format(self.prefix), init_loc)
        scale = param('{}_scale'.format(self.prefix), np.ones(self.latent_size),
                      constraint=constraints.positive)
        # TODO: Support for `.to_event()` to treat the batch dim as event dim.
        return sample("_{}_latent".format(self.prefix), dist.Normal(loc, scale))

    def _loc_scale(self, opt_state):
        params = self.get_params(opt_state)
        loc = params['{}_loc'.format(self.prefix)]
        scale = biject_to(constraints.positive)(params['{}_scale'.format(self.prefix)])
        return loc, scale

    def median(self, opt_state):
        """
        Returns the posterior median value of each latent variable.

        :param opt_state: Current state of the optimizer.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = self._loc_scale(opt_state)
        return {name: self._inv_transforms[name](unconstrained_value)
                for name, unconstrained_value in self.unravel_fn(loc).items()}

    def quantiles(self, opt_state, quantiles):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles(opt_state, [0.05, 0.5, 0.95]))

        :param opt_state: Current state of the optimizer.
        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        loc, scale = self._loc_scale(opt_state)
        result = {}
        for q in quantiles:
            latent = osp.norm(loc, scale).ppf(q)
            for name, unconstrained_value in self.unravel_fn(latent).items():
                transform = self._inv_transforms[name]
                result.setdefault(name, []).append(transform(unconstrained_value))
        return result
