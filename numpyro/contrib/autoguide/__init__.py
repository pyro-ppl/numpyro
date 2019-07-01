# Adapted from pyro.contrib.autoguide
from abc import ABC, abstractmethod

import numpy as onp
import scipy.stats as osp

from jax import random, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.tree_util import tree_map

import numpyro.distributions as dist
from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
from numpyro.distributions import constraints
from numpyro.distributions.constraints import biject_to, PermuteTransform
from numpyro.distributions.util import sum_rightmost
from numpyro.handlers import block, param, sample, seed, substitute, trace
from numpyro.infer_util import transform_fn

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


class AutoGuide(ABC):
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

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        A guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        """
        Generate samples from the approximate posterior over the latent
        sites in the model.

        :param jax.random.PRNGKey rng: PRNG seed.
        :param opt_state: Current state of the optimizer.
        :param sample_shape: (keyword argument) shape of samples to be drawn.
        :return: batch of samples from the approximate posterior.
        """
        raise NotImplementedError

    @abstractmethod
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
        # Wrap model in a `substitute` handler to initialize from `init_loc_fn`.
        # Use `block` to not record sample primitives in `init_loc_fn`.
        model = substitute(model, substitute_fn=block(seed(init_loc_fn, rng)))
        super(AutoContinuous, self).__init__(model, get_params_fn, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        self._inv_transforms = {}
        unconstrained_sites = {}
        for name, site in self.prototype_trace.items():
            if site['type'] == 'sample' and not site['is_observed']:
                # Collect the shapes of unconstrained values.
                # These may differ from the shapes of constrained values.
                transform = biject_to(site['fn'].support)
                unconstrained_val = transform.inv(site['value'])
                self._inv_transforms[name] = transform
                unconstrained_sites[name] = unconstrained_val

        latent_size = sum(np.size(x) for x in unconstrained_sites.values())
        if latent_size == 0:
            raise RuntimeError('{} found no latent variables; Use an empty guide instead'.format(type(self).__name__))
        self._init_latent, self._unravel_fn = ravel_pytree(unconstrained_sites)

    def _unpack_latent(self, latent_sample):
        sample_shape = np.shape(latent_sample)[:-1]
        latent_sample = np.reshape(latent_sample, (-1, np.shape(latent_sample)[-1]))
        unpacked_samples = vmap(self._unravel_fn)(latent_sample)
        unpacked_samples = tree_map(lambda x: np.reshape(x, sample_shape + np.shape(x)[1:]),
                                    unpacked_samples)
        return transform_fn(self._inv_transforms, unpacked_samples)

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

        for name, unconstrained_value in self._unravel_fn(latent).items():
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

        guide = AutoDiagonalNormal(rng, model, get_params, ...)
        svi_init, svi_update, _ = svi(model, guide, ...)
    """
    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        loc, scale = self._loc_scale(opt_state)
        latent_sample = dist.Normal(loc, scale).sample(rng, sample_shape)
        return self._unpack_latent(latent_sample)

    def _sample_latent(self, *args, **kwargs):
        init_loc = self._init_latent
        loc = param('{}_loc'.format(self.prefix), init_loc)
        scale = param('{}_scale'.format(self.prefix), np.ones(np.size(init_loc)),
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
        return transform_fn(self._inv_transforms, self._unravel_fn(loc))

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
            for name, unconstrained_value in self._unravel_fn(latent).items():
                transform = self._inv_transforms[name]
                result.setdefault(name, []).append(transform(unconstrained_value))
        return result


def elu(x):
    # according to IAF paper, ELU works better than other nonlinearities
    return np.where(x > 0, x, np.exp(x) - 1)


# TODO: remove when to_event is supported
class _Normal(dist.Normal):
    # work as Normal but has event_dim=1
    def __init__(self, *args, **kwargs):
        super(_Normal, self).__init__(*args, **kwargs)
        self._event_shape = self._batch_shape[-1:]
        self._batch_shape = self._batch_shape[:-1]

    def log_prob(self, value):
        return super(_Normal, self).log_prob(value).sum(-1)


class AutoIAFNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a
    :class:`~numpyro.distributions.iaf.InverseAutoregressiveTransform`
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoIAFNormal(rng, model, get_params, hidden_dims=[10], ...)
        svi_init, svi_update, _ = svi(model, guide, ...)
    """
    def __init__(self, rng, model, get_params_fn, prefix="auto", init_loc_fn=init_to_median,
                 num_flows=3, **arn_kwargs):
        self.arn_kwargs = arn_kwargs
        self.arns = []
        self.num_flows = num_flows
        rng, *self.arn_rngs = random.split(rng, num_flows + 1)
        super(AutoIAFNormal, self).__init__(rng, model, get_params_fn, prefix=prefix,
                                            init_loc_fn=init_loc_fn)

    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        params = self.get_params(opt_state)
        latent_size = np.size(self._init_latent)
        flows = []
        for i in range(self.num_flows):
            arn_params = params['{}_arn__{}'.format(self.prefix, i)]
            flows.append(dist.InverseAutoregressiveTransform(self.arns[i], arn_params))
        iaf_dist = dist.TransformedDistribution(_Normal(np.zeros(latent_size), 1.), flows)
        latent_sample = iaf_dist.sample(rng, sample_shape)
        return self._unpack_latent(latent_sample)

    def _sample_latent(self, *args, **kwargs):
        latent_size = np.size(self._init_latent)
        if latent_size == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        flows = []
        if not self.arns:
            # 2-layer by default following the experiments in IAF paper
            # (https://arxiv.org/abs/1606.04934) and Neutra paper (https://arxiv.org/abs/1903.03704)
            hidden_dims = self.arn_kwargs.get('hidden_dims', [latent_size, latent_size])
            nonlinearity = self.arn_kwargs.get('nonlinearity', elu)
            permutation = self.arn_kwargs.get('permutation', onp.arange(latent_size))
            for i in range(self.num_flows):
                arn = AutoregressiveNN(latent_size, hidden_dims, nonlinearity=nonlinearity)
                _, init_params = arn.init_fun(self.arn_rngs[i], (latent_size,), permutation)
                permutation = onp.arange(latent_size)[::-1]
                arn_params = param('{}_arn__{}'.format(self.prefix, i), init_params)
                self.arns.append(arn)
                flows.append(dist.InverseAutoregressiveTransform(arn, arn_params))
        else:
            for i in range(self.num_flows):
                arn_params = param('{}_arn__{}'.format(self.prefix, i), None)
                flows.append(dist.InverseAutoregressiveTransform(self.arns[i], arn_params))

        # TODO: support to_event for distributions
        iaf_dist = dist.TransformedDistribution(_Normal(np.zeros(latent_size), 1.), flows)
        return sample("_{}_latent".format(self.prefix), iaf_dist)
