# Adapted from pyro.contrib.autoguide
from abc import ABC, abstractmethod

import numpy as onp
import scipy.stats as osp

from jax import random, vmap
from jax.experimental import stax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.tree_util import tree_map

from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import AffineTransform, ComposeTransform, PermuteTransform, biject_to
from numpyro.distributions.flows import InverseAutoregressiveTransform
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

    @abstractmethod
    def _get_transform(self):
        raise NotImplementedError

    def _sample_latent(self, base_dist, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        transform = self._get_transform()
        posterior = dist.TransformedDistribution(base_dist, transform)
        return sample("_{}_latent".format(self.prefix), posterior, sample_shape=sample_shape)

    def setup(self, *args, **kwargs):
        self._setup_prototype(*args, **kwargs)
        return {}

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # run model to inspect the model structure
        params = {}
        if self.prototype_trace is None:
            params = self.setup(*args, **kwargs)

        base_dist = kwargs.pop('base_dist', None)
        latent_size = np.size(self._init_latent)
        if base_dist is None:
            base_dist = _Normal(np.zeros(latent_size), 1.)
        latent = substitute(self._sample_latent, params)(base_dist, *args, **kwargs)

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

    def unpack_latent(self, latent_sample, transform=None):
        sample_shape = np.shape(latent_sample)[:-1]
        latent_sample = np.reshape(latent_sample, (-1, np.shape(latent_sample)[-1]))
        unpacked_samples = vmap(self._unravel_fn)(latent_sample)
        unpacked_samples = tree_map(lambda x: np.reshape(x, sample_shape + np.shape(x)[1:]),
                                    unpacked_samples)

        transform = self._inv_transforms if transform is None else {}
        return transform_fn(transform, unpacked_samples)

    def get_transform(self, opt_state):
        return substitute(self._get_transform, self.get_params(opt_state))()

    def sample_posterior(self, rng, opt_state, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        base_dist = kwargs.pop('base_dist', None)
        latent_size = np.size(self._init_latent)
        if base_dist is None:
            base_dist = _Normal(np.zeros(latent_size), 1.)
        params = self.get_params(opt_state)
        latent_sample = substitute(seed(self._sample_latent, rng), params)(base_dist, sample_shape=sample_shape)
        return self.unpack_latent(latent_sample)


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(rng, model, get_params, ...)
        svi_init, svi_update, _ = svi(model, guide, ...)
    """
    def _get_transform(self):
        loc, scale = self._loc_scale()
        return AffineTransform(loc, scale, domain=constraints.real_vector)

    def _loc_scale(self):
        loc = param('{}_loc'.format(self.prefix), None)
        scale = biject_to(constraints.positive)(param('{}_scale'.format(self.prefix), None))
        return loc, scale

    def setup(self, *args, **kwargs):
        super(AutoDiagonalNormal, self).setup(*args, **kwargs)
        return {
            '{}_loc'.format(self.prefix): self._init_latent,
            '{}_scale'.format(self.prefix): np.ones(np.size(self._init_latent)),
        }

    def median(self, opt_state):
        """
        Returns the posterior median value of each latent variable.

        :param opt_state: Current state of the optimizer.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = substitute(self._loc_scale, self.get_params(opt_state))()
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
        loc, scale = substitute(self._loc_scale, self.get_params(opt_state))()
        result = {}
        for q in quantiles:
            latent = osp.norm(loc, scale).ppf(q)
            for name, unconstrained_value in self._unravel_fn(latent).items():
                transform = self._inv_transforms[name]
                result.setdefault(name, []).append(transform(unconstrained_value))
        return result


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

        guide = AutoIAFNormal(rng, model, get_params, hidden_dims=[20], skip_connections=True, ...)
        svi_init, svi_update, _ = svi(model, guide, ...)

    :param jax.random.PRNGKey rng: random key to be used as the source of randomness
        to initialize the guide.
    :param callable model: a generative model.
    :param callable get_params_fn: a function to get params from an optimization state.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_loc_fn: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param `**arn_kwargs`: keywords for constructing autoregressive neural networks.
    """
    def __init__(self, rng, model, get_params_fn, prefix="auto", init_loc_fn=init_to_median,
                 num_flows=3, **arn_kwargs):
        self.arn_kwargs = arn_kwargs
        self.arns = []
        self.num_flows = num_flows
        rng, *self.arn_rngs = random.split(rng, num_flows + 1)
        super(AutoIAFNormal, self).__init__(rng, model, get_params_fn, prefix=prefix,
                                            init_loc_fn=init_loc_fn)

    def setup(self, *args, **kwargs):
        super(AutoIAFNormal, self).setup(*args, **kwargs)
        latent_size = np.size(self._init_latent)
        if latent_size == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        params = {}
        if not self.arns:
            # 2-layer by default following the experiments in IAF paper
            # (https://arxiv.org/abs/1606.04934) and Neutra paper (https://arxiv.org/abs/1903.03704)
            hidden_dims = self.arn_kwargs.get('hidden_dims', [latent_size, latent_size])
            skip_connections = self.arn_kwargs.get('skip_connections', True)
            nonlinearity = self.arn_kwargs.get('nonlinearity', stax.Relu)

            for i in range(self.num_flows):
                arn_init, arn = AutoregressiveNN(latent_size, hidden_dims,
                                                 permutation=np.arange(latent_size),
                                                 skip_connections=skip_connections,
                                                 nonlinearity=nonlinearity)
                _, init_params = arn_init(self.arn_rngs[i], (latent_size,))
                params['{}_arn__{}'.format(self.prefix, i)] = init_params
                self.arns.append(arn)
        return params

    def _get_transform(self):
        latent_size = np.size(self._init_latent)
        flows = []
        for i in range(self.num_flows):
            arn_params = param('{}_arn__{}'.format(self.prefix, i), None)
            if i > 0:
                flows.append(PermuteTransform(np.arange(latent_size)[::-1]))
            flows.append(InverseAutoregressiveTransform(self.arns[i], arn_params))
        return ComposeTransform(flows)
