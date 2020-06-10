# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Adapted from pyro.infer.autoguide
from abc import ABC, abstractmethod
import warnings

from jax import hessian, lax, random, tree_map
from jax.experimental import stax
from jax.flatten_util import ravel_pytree
import jax.numpy as np

import numpyro
from numpyro import handlers
from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
from numpyro.contrib.nn.block_neural_arn import BlockNeuralAutoregressiveNN
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.flows import BlockNeuralAutoregressiveTransform, InverseAutoregressiveTransform
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    MultivariateAffineTransform,
    PermuteTransform,
    UnpackTransform,
    biject_to
)
from numpyro.distributions.util import cholesky_of_inverse, sum_rightmost
from numpyro.infer.elbo import ELBO
from numpyro.infer.util import initialize_model, init_to_uniform
from numpyro.util import not_jax_tracer

__all__ = [
    'AutoContinuous',
    'AutoGuide',
    'AutoDiagonalNormal',
    'AutoLaplaceApproximation',
    'AutoLowRankMultivariateNormal',
    'AutoMultivariateNormal',
    'AutoBNAFNormal',
    'AutoIAFNormal',
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
    def sample_posterior(self, rng_key, params, *args, **kwargs):
        """
        Generate samples from the approximate posterior over the latent
        sites in the model.

        :param jax.random.PRNGKey rng_key: PRNG seed.
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
        rng_key = numpyro.sample("_{}_rng_key_setup".format(self.prefix), dist.PRNGIdentity())
        model = handlers.seed(self.model, rng_key)
        self.prototype_trace = handlers.block(handlers.trace(model).get_trace)(*args, **kwargs)


class AutoContinuous(AutoGuide):
    """
    Base class for implementations of continuous-valued Automatic
    Differentiation Variational Inference [1].

    Each derived class implements its own :meth:`_get_transform` method.

    Assumes model structure and latent dimension are fixed, and all latent
    variables are continuous.

    **Reference:**

    1. *Automatic Differentiation Variational Inference*,
       Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M.
       Blei

    :param callable model: A NumPyro model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform):
        self.init_strategy = init_strategy
        super(AutoContinuous, self).__init__(model, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        rng_key = numpyro.sample("_{}_rng_key_setup".format(self.prefix), dist.PRNGIdentity())
        with handlers.block():
            init_params, _, self._postprocess_fn, self.prototype_trace = initialize_model(
                rng_key, self.model,
                init_strategy=self.init_strategy,
                dynamic_args=False,
                model_args=args,
                model_kwargs=kwargs)

        self._init_latent, unpack_latent = ravel_pytree(init_params[0])
        # this is to match the behavior of Pyro, where we can apply
        # unpack_latent for a batch of samples
        self._unpack_latent = UnpackTransform(unpack_latent)
        self.latent_dim = np.size(self._init_latent)
        if self.latent_dim == 0:
            raise RuntimeError('{} found no latent variables; Use an empty guide instead'
                               .format(type(self).__name__))

    @abstractmethod
    def _get_posterior(self):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', ())
        posterior = self._get_posterior()
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

        latent = self._sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = {}

        for name, unconstrained_value in self._unpack_latent(latent).items():
            site = self.prototype_trace[name]
            transform = biject_to(site['fn'].support)
            value = transform(unconstrained_value)
            log_density = - transform.log_abs_det_jacobian(unconstrained_value, value)
            event_ndim = len(site['fn'].event_shape)
            log_density = sum_rightmost(log_density,
                                        np.ndim(log_density) - np.ndim(value) + event_ndim)
            delta_dist = dist.Delta(value, log_density=log_density, event_dim=event_ndim)
            result[name] = numpyro.sample(name, delta_dist)

        return result

    def _unpack_and_constrain(self, latent_sample, params):
        def unpack_single_latent(latent):
            unpacked_samples = self._unpack_latent(latent)
            # add param sites in model
            unpacked_samples.update({k: v for k, v in params.items() if k in self.prototype_trace
                                     and v['type'] == 'param'})
            return self._postprocess_fn(unpacked_samples)

        sample_shape = np.shape(latent_sample)[:-1]
        if sample_shape:
            latent_sample = np.reshape(latent_sample, (-1, np.shape(latent_sample)[-1]))
            unpacked_samples = lax.map(unpack_single_latent, latent_sample)
            return tree_map(lambda x: np.reshape(x, sample_shape + np.shape(x)[1:]),
                            unpacked_samples)
        else:
            return unpack_single_latent(latent_sample)

    @property
    def get_base_dist(self):
        """
        Returns the base distribution of the posterior when reparameterized
        as a :class:`~numpyro.distributions.TransformedDistribution`. This
        should not depend on the model's `*args, **kwargs`.
        """
        raise NotImplementedError

    def get_transform(self, params):
        """
        Returns the transformation learned by the guide to generate samples from the unconstrained
        (approximate) posterior.

        :param dict params: Current parameters of model and autoguide.
        :return: the transform of posterior distribution
        :rtype: :class:`~numpyro.distributions.transforms.Transform`
        """
        posterior = handlers.substitute(self._get_posterior, params)()
        assert isinstance(posterior, dist.TransformedDistribution), \
            "posterior is not a transformed distribution"
        if len(posterior.transforms) > 0:
            return ComposeTransform(posterior.transforms)
        else:
            return posterior.transforms[0]

    def get_posterior(self, params):
        """
        Returns the posterior distribution.

        :param dict params: Current parameters of model and autoguide.
        """
        base_dist = self.get_base_dist()
        transform = self.get_transform(params)
        return dist.TransformedDistribution(base_dist, transform)

    def sample_posterior(self, rng_key, params, sample_shape=()):
        """
        Get samples from the learned posterior.

        :param jax.random.PRNGKey rng_key: random key to be used draw samples.
        :param dict params: Current parameters of model and autoguide.
        :param tuple sample_shape: batch shape of each latent sample, defaults to ().
        :return: a dict containing samples drawn the this guide.
        :rtype: dict
        """
        latent_sample = handlers.substitute(
            handlers.seed(self._sample_latent, rng_key), params)(sample_shape=sample_shape)
        return self._unpack_and_constrain(latent_sample, params)

    def median(self, params):
        """
        Returns the posterior median value of each latent variable.

        :param dict params: A dict containing parameter values.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        raise NotImplementedError

    def quantiles(self, params, quantiles):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles(opt_state, [0.05, 0.5, 0.95]))

        :param dict params: A dict containing parameter values.
        :param list quantiles: A list of requested quantiles between 0 and 1.
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        raise NotImplementedError


class AutoDiagonalNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoDiagonalNormal(model, ...)
        svi = SVI(model, guide, ...)
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform, init_scale=0.1):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, prefix, init_strategy)

    def _get_posterior(self):
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        scale = numpyro.param('{}_scale'.format(self.prefix),
                              np.full(self.latent_dim, self._init_scale),
                              constraint=constraints.positive)
        return dist.Normal(loc, scale)

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        scale = params['{}_scale'.format(self.prefix)]
        return AffineTransform(loc, scale, domain=constraints.real_vector)

    def get_posterior(self, params):
        """
        Returns a diagonal Normal posterior distribution.
        """
        transform = self.get_transform(params)
        return dist.Normal(transform.loc, transform.scale)

    def median(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        quantiles = np.array(quantiles)[..., None]
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
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform, init_scale=0.1):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, prefix, init_strategy)

    def _get_posterior(self):
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        scale_tril = numpyro.param('{}_scale_tril'.format(self.prefix),
                                   np.identity(self.latent_dim) * self._init_scale,
                                   constraint=constraints.lower_cholesky)
        return dist.MultivariateNormal(loc, scale_tril=scale_tril)

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        scale_tril = params['{}_scale_tril'.format(self.prefix)]
        return MultivariateAffineTransform(loc, scale_tril)

    def get_posterior(self, params):
        """
        Returns a multivariate Normal posterior distribution.
        """
        transform = self.get_transform(params)
        return dist.MultivariateNormal(transform.loc, transform.scale_tril)

    def median(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        transform = self.get_transform(params)
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(transform.loc, np.diagonal(transform.scale_tril)).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoLowRankMultivariateNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a LowRankMultivariateNormal
    distribution to construct a guide over the entire latent space.
    The guide does not depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoLowRankMultivariateNormal(model, rank=2, ...)
        svi = SVI(model, guide, ...)
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform, init_scale=0.1, rank=None):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self.rank = rank
        super(AutoLowRankMultivariateNormal, self).__init__(
            model, prefix=prefix, init_strategy=init_strategy)

    def _get_posterior(self, *args, **kwargs):
        rank = int(round(self.latent_dim ** 0.5)) if self.rank is None else self.rank
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        cov_factor = numpyro.param('{}_cov_factor'.format(self.prefix), np.zeros((self.latent_dim, rank)))
        scale = numpyro.param('{}_scale'.format(self.prefix),
                              np.full(self.latent_dim, self._init_scale),
                              constraint=constraints.positive)
        cov_diag = scale * scale
        cov_factor = cov_factor * scale[..., None]
        return dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        posterior = self.get_posterior(params)
        return MultivariateAffineTransform(posterior.loc, posterior.scale_tril)

    def get_posterior(self, params):
        """
        Returns a lowrank multivariate Normal posterior distribution.
        """
        loc = params['{}_loc'.format(self.prefix)]
        cov_factor = params['{}_cov_factor'.format(self.prefix)]
        scale = params['{}_scale'.format(self.prefix)]
        cov_diag = scale * scale
        cov_factor = cov_factor * scale[..., None]
        return dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def median(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        transform = self.get_transform(params)
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(transform.loc, np.diagonal(transform.scale_tril)).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoLaplaceApproximation(AutoContinuous):
    r"""
    Laplace approximation (quadratic approximation) approximates the posterior
    :math:`\log p(z | x)` by a multivariate normal distribution in the
    unconstrained space. Under the hood, it uses Delta distributions to
    construct a MAP guide over the entire (unconstrained) latent space. Its
    covariance is given by the inverse of the hessian of :math:`-\log p(x, z)`
    at the MAP point of `z`.

    Usage::

        guide = AutoLaplaceApproximation(model, ...)
        svi = SVI(model, guide, ...)
    """
    def _setup_prototype(self, *args, **kwargs):
        super(AutoLaplaceApproximation, self)._setup_prototype(*args, **kwargs)

        def loss_fn(params):
            # we are doing maximum likelihood, so only require `num_particles=1` and an arbitrary rng_key.
            return ELBO().loss(random.PRNGKey(0), params, self.model, self, *args, **kwargs)

        self._loss_fn = loss_fn

    def _get_posterior(self, *args, **kwargs):
        # sample from Delta guide
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        return dist.Delta(loc, event_dim=1)

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        def loss_fn(z):
            params1 = params.copy()
            params1['{}_loc'.format(self.prefix)] = z
            return self._loss_fn(params1)

        loc = params['{}_loc'.format(self.prefix)]
        precision = hessian(loss_fn)(loc)
        scale_tril = cholesky_of_inverse(precision)
        if not_jax_tracer(scale_tril):
            if np.any(np.isnan(scale_tril)):
                warnings.warn("Hessian of log posterior at the MAP point is singular. Posterior"
                              " samples from AutoLaplaceApproxmiation will be constant (equal to"
                              " the MAP point).")
        scale_tril = np.where(np.isnan(scale_tril), 0., scale_tril)
        return MultivariateAffineTransform(loc, scale_tril)

    def get_posterior(self, params):
        """
        Returns a multivariate Normal posterior distribution.
        """
        transform = self.get_transform(params)
        return dist.MultivariateNormal(transform.loc, scale_tril=transform.scale_tril)

    def sample_posterior(self, rng_key, params, sample_shape=()):
        latent_sample = self.get_posterior(params).sample(rng_key, sample_shape)
        return self._unpack_and_constrain(latent_sample, params)

    def median(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        transform = self.get_transform(params)
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(transform.loc, np.diagonal(transform.scale_tril)).icdf(quantiles)
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
    :param callable init_strategy: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param list hidden_dims: the dimensionality of the hidden units per layer.
        Defaults to ``[latent_dim, latent_dim]``.
    :param bool skip_connections: whether to add skip connections from the input to the
        output of each flow. Defaults to False.
    :param callable nonlinearity: the nonlinearity to use in the feedforward network.
        Defaults to :func:`jax.experimental.stax.Elu`.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform,
                 num_flows=3, hidden_dims=None, skip_connections=False, nonlinearity=stax.Elu):
        self.num_flows = num_flows
        # 2-layer, stax.Elu, skip_connections=False by default following the experiments in
        # IAF paper (https://arxiv.org/abs/1606.04934)
        # and Neutra paper (https://arxiv.org/abs/1903.03704)
        self._hidden_dims = hidden_dims
        self._skip_connections = skip_connections
        self._nonlinearity = nonlinearity
        super(AutoIAFNormal, self).__init__(model, prefix=prefix, init_strategy=init_strategy)

    def _get_posterior(self):
        if self.latent_dim == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        hidden_dims = [self.latent_dim, self.latent_dim] if self._hidden_dims is None else self._hidden_dims
        flows = []
        for i in range(self.num_flows):
            if i > 0:
                flows.append(PermuteTransform(np.arange(self.latent_dim)[::-1]))
            arn = AutoregressiveNN(self.latent_dim, hidden_dims,
                                   permutation=np.arange(self.latent_dim),
                                   skip_connections=self._skip_connections,
                                   nonlinearity=self._nonlinearity)
            arnn = numpyro.module('{}_arn__{}'.format(self.prefix, i), arn, (self.latent_dim,))
            flows.append(InverseAutoregressiveTransform(arnn))
        return dist.TransformedDistribution(self.get_base_dist(), flows)

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)


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
    :param callable init_strategy: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param list hidden_factors: Hidden layer i has ``hidden_factors[i]`` hidden units per
        input dimension. This corresponds to both :math:`a` and :math:`b` in reference [1].
        The elements of hidden_factors must be integers.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform, num_flows=1,
                 hidden_factors=[8, 8]):
        self.num_flows = num_flows
        self._hidden_factors = hidden_factors
        super(AutoBNAFNormal, self).__init__(model, prefix=prefix, init_strategy=init_strategy)

    def _get_posterior(self):
        if self.latent_dim == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        flows = []
        for i in range(self.num_flows):
            if i > 0:
                flows.append(PermuteTransform(np.arange(self.latent_dim)[::-1]))
            residual = "gated" if i < (self.num_flows - 1) else None
            arn = BlockNeuralAutoregressiveNN(self.latent_dim, self._hidden_factors, residual)
            arnn = numpyro.module('{}_arn__{}'.format(self.prefix, i), arn, (self.latent_dim,))
            flows.append(BlockNeuralAutoregressiveTransform(arnn))
        return dist.TransformedDistribution(self.get_base_dist(), flows)

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)
