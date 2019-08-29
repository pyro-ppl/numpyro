# Adapted from pyro.contrib.autoguide
from abc import ABC, abstractmethod

from jax import vmap
from jax.experimental import stax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.tree_util import tree_map

import numpyro
from numpyro import handlers
from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.constraints import AffineTransform, ComposeTransform, PermuteTransform, biject_to
from numpyro.distributions.flows import InverseAutoregressiveTransform
from numpyro.distributions.util import sum_rightmost
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

    def __init__(self, model, prefix='auto'):
        assert isinstance(prefix, str)
        self.model = model
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
    def sample_posterior(self, rng, params, *args, **kwargs):
        """
        Generate samples from the approximate posterior over the latent
        sites in the model.

        :param jax.random.PRNGKey rng: PRNG seed.
        :param params: Current parameters of model and autoguide.
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
        rng = numpyro.sample("_{}_rng_setup".format(self.prefix), dist.PRNGIdentity(), sample_shape=(2,))
        model = handlers.seed(self.model, rng)
        self.prototype_trace = handlers.block(handlers.trace(model).get_trace)(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].

    Each derived class implements its own :meth:`get_posterior` method.

    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.

    Reference:

    [1] `Automatic Differentiation Variational Inference`,
        Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M.
        Blei

    :param jax.random.PRNGKey rng: random key to be used as the source of randomness
        to initialize the guide.
    :param callable model: A NumPyro model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_median):
        self.init_strategy = init_strategy
        self._base_dist = None
        super(AutoContinuous, self).__init__(model, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        rng = numpyro.sample("_{}_rng_init".format(self.prefix), dist.PRNGIdentity())
        # FIXME: without block statement, get AssertionError: all sites must have unique names
        init_params, is_valid = handlers.block(find_valid_initial_params)(rng, self.model, *args,
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
        if self.base_dist is None:
            self.base_dist = _Normal(np.zeros(self.latent_size), 1.)
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
        return numpyro.sample("_{}_latent".format(self.prefix), posterior, sample_shape=sample_shape)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        latent = self._sample_latent(self.base_dist, *args, **kwargs)

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
            result[name] = numpyro.sample(name, delta_dist)

        return result

    def _unpack_and_constrain(self, latent_sample, params):
        sample_shape = np.shape(latent_sample)[:-1]
        latent_sample = np.reshape(latent_sample, (-1, np.shape(latent_sample)[-1]))
        # XXX: we do not support priors with supports depending on dynamic data
        # because it adds complexity to the interface.
        # Users can achieve that behaviour by changing the default `self._args`
        # but we will not recommend doing so.
        model_args = self._args
        model_kwargs = self._kwargs

        def unpack_single_latent(latent):
            unpacked_samples = self.unpack_latent(latent)
            if self._has_transformed_dist:
                # first, substitute to `param` statements in model
                model = handlers.substitute(self.model, params)
                return constrain_fn(model, model_args, model_kwargs,
                                    self._inv_transforms, unpacked_samples)
            else:
                return transform_fn(self._inv_transforms, unpacked_samples)

        unpacked_samples = vmap(unpack_single_latent)(latent_sample)
        unpacked_samples = tree_map(lambda x: np.reshape(x, sample_shape + np.shape(x)[1:]),
                                    unpacked_samples)
        return unpacked_samples

    @property
    def base_dist(self):
        """
        Base distribution of the posterior. By default, it is standard normal.
        """
        return self._base_dist

    @base_dist.setter
    def base_dist(self, base_dist):
        self._base_dist = base_dist

    def get_transform(self, params):
        """
        Returns the transformation learned by the guide to generate samples from the unconstrained
        (approximate) posterior.

        :param dict params: Current parameters of model and autoguide.
        :return: the transform of posterior distribution
        :rtype: :class:`~numpyro.distributions.constraints.Transform`
        """
        return handlers.substitute(self._get_transform, params)()

    def sample_posterior(self, rng, params, sample_shape=()):
        """
        Get samples from the learned posterior.

        :param jax.random.PRNGKey rng: random key to be used draw samples.
        :param dict params: Current parameters of model and autoguide.
        :param tuple sample_shape: batch shape of each latent sample, defaults to ().
        :return: a dict containing samples drawn the this guide.
        :rtype: dict
        """
        latent_sample = handlers.substitute(handlers.seed(self._sample_latent, rng), params)(
            self.base_dist, sample_shape=sample_shape)
        return self._unpack_and_constrain(latent_sample, params)


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(rng, model, ...)
        svi_init, svi_update, _ = svi(model, guide, ...)
    """
    def _get_transform(self):
        loc, scale = self._loc_scale()
        return AffineTransform(loc, scale, domain=constraints.real_vector)

    def _loc_scale(self):
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        scale = numpyro.param('{}_scale'.format(self.prefix), np.ones(self.latent_size),
                              constraint=constraints.positive)
        return loc, scale

    def median(self, params):
        """
        Returns the posterior median value of each latent variable.

        :param dict params: A dict containing parameter values.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        loc, _ = handlers.substitute(self._loc_scale, params)()
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles(opt_state, [0.05, 0.5, 0.95]))

        :param opt_state: Current state of the optimizer.
        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        loc, scale = handlers.substitute(self._loc_scale, params)()
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(loc, scale).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


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
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_strategy: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param `**arn_kwargs`: keywords for constructing autoregressive neural networks, which includes
        * **hidden_dims** (``list[int]``) - the dimensionality of the hidden units per layer.
            Defaults to ``[latent_size, latent_size]``.
        * **skip_connections** (``bool``) - whether to add skip connections from the input to the
            output of each flow. Defaults to False.
        * **nonlinearity** (``callable``) - the nonlinearity to use in the feedforward network.
            Defaults to :func:`jax.experimental.stax.Relu`.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_median,
                 num_flows=3, **arn_kwargs):
        self.num_flows = num_flows
        # 2-layer, stax.Elu, skip_connections=False by default following the experiments in
        # IAF paper (https://arxiv.org/abs/1606.04934)
        # and Neutra paper (https://arxiv.org/abs/1903.03704)
        self._hidden_dims = arn_kwargs.get('hidden_dims')
        self._skip_connections = arn_kwargs.get('skip_connections', False)
        # TODO: follow the recommendation of the above two papers, use stax.Elu by defaults
        # currently, using stax.Elu seems not stable
        self._nonlinearity = arn_kwargs.get('nonlinearity', stax.Relu)
        super(AutoIAFNormal, self).__init__(model, prefix=prefix, init_strategy=init_strategy)

    def _get_transform(self):
        if self.latent_size == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        hidden_dims = [self.latent_size, self.latent_size] if self._hidden_dims is None else self._hidden_dims
        flows = []
        for i in range(self.num_flows):
            if i > 0:
                flows.append(PermuteTransform(np.arange(self.latent_size)[::-1]))
            arn = AutoregressiveNN(self.latent_size, hidden_dims,
                                   permutation=np.arange(self.latent_size),
                                   skip_connections=self._skip_connections,
                                   nonlinearity=self._nonlinearity)
            arnn = numpyro.module('{}_arn__{}'.format(self.prefix, i), arn, (self.latent_size,))
            flows.append(InverseAutoregressiveTransform(arnn))
        return ComposeTransform(flows)
