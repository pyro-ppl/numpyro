# Adapted from pyro.contrib.autoguide
from abc import ABC, abstractmethod

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
from numpyro.infer_util import constrain_fn, find_valid_initial_params, init_to_median, transform_fn

__all__ = [
    'AutoContinuous',
    'AutoGuide',
    'AutoDiagonalNormal',
]


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

    def setup(self, *args, **kwargs):
        """
        First call to set up any necessary state. Returns initial
        value of parameters in the guide.

        :param args: model args.
        :param kwargs: model kwargs.
        :return: dict of initial parameters.
        """
        self._setup_prototype(*args, **kwargs)
        return {}

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
        self._args = args
        self._kwargs = kwargs


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
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """
    def __init__(self, rng, model, get_params_fn, prefix="auto", init_strategy=init_to_median):
        self.rng = rng
        self.init_strategy = init_strategy
        model = seed(model, rng)
        super(AutoContinuous, self).__init__(model, get_params_fn, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        # FIXME: without block statement, get AssertionError: all sites must have unique names
        init_params, is_valid = block(find_valid_initial_params)(self.rng, self.model, *args,
                                                                 init_strategy=self.init_strategy,
                                                                 **kwargs)
        self._inv_transforms = {}
        self._has_transformed_dist = False
        unconstrained_sites = {}
        for name, site in self.prototype_trace.items():
            if site['type'] == 'sample' and not site['is_observed']:
                if site['intermediates']:
                    transform = biject_to(site['fn'].base_dist.support)
                    self._inv_transforms[name] = transform
                    unconstrained_sites[name] = transform.inv(site['intermediates'][0][0])
                    self._has_transformed_dist = True
                else:
                    transform = biject_to(site['fn'].support)
                    self._inv_transforms[name] = transform
                    unconstrained_sites[name] = transform.inv(site['value'])

        self._init_latent, self.unpack_latent = ravel_pytree(init_params)
        self.latent_size = np.size(self._init_latent)
        if self.latent_size == 0:
            raise RuntimeError('{} found no latent variables; Use an empty guide instead'
                               .format(type(self).__name__))

    @abstractmethod
    def _get_transform(self):
        raise NotImplementedError

    def _sample_latent(self, base_dist, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        transform = self._get_transform()
        posterior = dist.TransformedDistribution(base_dist, transform)
        return sample("_{}_latent".format(self.prefix), posterior, sample_shape=sample_shape)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        sample_latent_fn = self._sample_latent
        if self.prototype_trace is None:
            # run model to inspect the model structure
            params = self.setup(*args, **kwargs)
            sample_latent_fn = substitute(sample_latent_fn, params)

        base_dist = kwargs.pop('base_dist', None)
        if base_dist is None:
            base_dist = _Normal(np.zeros(self.latent_size), 1.)
        latent = sample_latent_fn(base_dist, *args, **kwargs)

        # unpack continuous latent samples
        result = {}

        for name, unconstrained_value in self.unpack_latent(latent).items():
            transform = self._inv_transforms[name]
            site = self.prototype_trace[name]
            value = transform(unconstrained_value)
            log_density = - transform.log_abs_det_jacobian(unconstrained_value, value)
            if site['intermediates']:
                event_ndim = len(site['fn'].base_dist.event_shape)
            else:
                event_ndim = len(site['fn'].event_shape)
            log_density = sum_rightmost(log_density,
                                        np.ndim(log_density) - np.ndim(value) + event_ndim)
            delta_dist = dist.Delta(value, log_density=log_density, event_ndim=event_ndim)
            result[name] = sample(name, delta_dist)

        return result

    def _unpack_and_transform(self, latent_sample, params, model_args=None, model_kwargs=None):
        sample_shape = np.shape(latent_sample)[:-1]
        latent_sample = np.reshape(latent_sample, (-1, np.shape(latent_sample)[-1]))
        model_args = self._args if model_args is None else model_args
        model_kwargs = self._kwargs if model_kwargs is None else model_kwargs

        def unpack_single_latent(latent):
            unpacked_samples = self.unpack_latent(latent)
            if self._has_transformed_dist:
                base_param_map = {**params, **unpacked_samples}
                return constrain_fn(self.model, model_args, model_kwargs,
                                    self._inv_transforms, base_param_map)
            else:
                return transform_fn(self._inv_transforms, unpacked_samples)

        unpacked_samples = vmap(unpack_single_latent)(latent_sample)
        unpacked_samples = tree_map(lambda x: np.reshape(x, sample_shape + np.shape(x)[1:]),
                                    unpacked_samples)
        return unpacked_samples

    def get_transform(self, opt_state):
        return substitute(self._get_transform, self.get_params(opt_state))()

    def sample_posterior(self, rng, opt_state, sample_shape=(), base_dist=None,
                         model_args=None, model_kwargs=None):
        if base_dist is None:
            base_dist = _Normal(np.zeros(self.latent_size), 1.)
        params = self.get_params(opt_state)
        latent_sample = substitute(seed(self._sample_latent, rng), params)(
            base_dist, sample_shape=sample_shape)
        return self._unpack_and_transform(latent_sample, params,
                                          model_args=model_args, model_kwargs=model_kwargs)


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
        scale = param('{}_scale'.format(self.prefix), None, constraint=constraints.positive)
        return loc, scale

    def setup(self, *args, **kwargs):
        super(AutoDiagonalNormal, self).setup(*args, **kwargs)
        return {
            '{}_loc'.format(self.prefix): self._init_latent,
            '{}_scale'.format(self.prefix): np.ones(self.latent_size),
        }

    def median(self, opt_state, model_args=None, model_kwargs=None):
        """
        Returns the posterior median value of each latent variable.

        :param opt_state: Current state of the optimizer.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        params = self.get_params(opt_state)
        loc, _ = substitute(self._loc_scale, params)()
        return self._unpack_and_transform(loc, params, model_args=model_args, model_kwargs=model_kwargs)

    def quantiles(self, opt_state, quantiles, model_args=None, model_kwargs=None):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles(opt_state, [0.05, 0.5, 0.95]))

        :param opt_state: Current state of the optimizer.
        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        params = self.get_params(opt_state)
        loc, scale = substitute(self._loc_scale, params)()
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(loc, scale).icdf(quantiles)
        return self._unpack_and_transform(latent, params, model_args=model_args, model_kwargs=model_kwargs)


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
    def __init__(self, rng, model, get_params_fn, prefix="auto", init_strategy=init_to_median,
                 num_flows=3, **arn_kwargs):
        self.arn_kwargs = arn_kwargs
        self.arns = []
        self.num_flows = num_flows
        rng, *self.arn_rngs = random.split(rng, num_flows + 1)
        super(AutoIAFNormal, self).__init__(rng, model, get_params_fn, prefix=prefix,
                                            init_strategy=init_strategy)

    def setup(self, *args, **kwargs):
        super(AutoIAFNormal, self).setup(*args, **kwargs)
        if self.latent_size == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        params = {}
        if not self.arns:
            # 2-layer by default following the experiments in IAF paper
            # (https://arxiv.org/abs/1606.04934) and Neutra paper (https://arxiv.org/abs/1903.03704)
            hidden_dims = self.arn_kwargs.get('hidden_dims', [self.latent_size, self.latent_size])
            skip_connections = self.arn_kwargs.get('skip_connections', True)
            nonlinearity = self.arn_kwargs.get('nonlinearity', stax.Relu)

            for i in range(self.num_flows):
                arn_init, arn = AutoregressiveNN(self.latent_size, hidden_dims,
                                                 permutation=np.arange(self.latent_size),
                                                 skip_connections=skip_connections,
                                                 nonlinearity=nonlinearity)
                _, init_params = arn_init(self.arn_rngs[i], (self.latent_size,))
                params['{}_arn__{}'.format(self.prefix, i)] = init_params
                self.arns.append(arn)
        return params

    def _get_transform(self):
        flows = []
        for i in range(self.num_flows):
            arn_params = param('{}_arn__{}'.format(self.prefix, i), None)
            if i > 0:
                flows.append(PermuteTransform(np.arange(self.latent_size)[::-1]))
            flows.append(InverseAutoregressiveTransform(self.arns[i], arn_params))
        return ComposeTransform(flows)
