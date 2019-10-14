# Adapted from pyro.contrib.autoguide
from abc import ABC, abstractmethod

from jax import hessian, random, vmap
from jax.experimental import stax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.tree_util import tree_map

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
from numpyro.distributions.util import cholesky_inverse, sum_rightmost
from numpyro.handlers import seed, substitute
from numpyro.infer.elbo import ELBO
from numpyro.infer.util import constrain_fn, find_valid_initial_params, init_to_uniform, log_density, transform_fn

__all__ = [
    'AutoContinuous',
    'AutoContinuousELBO',
    'AutoGuide',
    'AutoDiagonalNormal',
    'AutoMultivariateNormal',
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
        rng = numpyro.sample("_{}_rng_setup".format(self.prefix), dist.PRNGIdentity())
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
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform()):
        self.init_strategy = init_strategy
        self._base_dist = None
        super(AutoContinuous, self).__init__(model, prefix=prefix)

    def _setup_prototype(self, *args, **kwargs):
        super(AutoContinuous, self)._setup_prototype(*args, **kwargs)
        rng = numpyro.sample("_{}_rng_init".format(self.prefix), dist.PRNGIdentity())
        init_params, _ = handlers.block(find_valid_initial_params)(rng, self.model, *args,
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

        self._init_latent, self._unpack_latent = ravel_pytree(init_params)
        self.latent_size = np.size(self._init_latent)
        if self.base_dist is None:
            self.base_dist = dist.Independent(dist.Normal(np.zeros(self.latent_size), 1.), 1)
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

        for name, unconstrained_value in self._unpack_latent(latent).items():
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
            unpacked_samples = self._unpack_latent(latent)
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
        return ComposeTransform([handlers.substitute(self._get_transform, params)(),
                                 UnpackTransform(self._unpack_latent)])

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

        :param opt_state: Current state of the optimizer.
        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
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
    def _get_transform(self):
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        scale = numpyro.param('{}_scale'.format(self.prefix), np.ones(self.latent_size),
                              constraint=constraints.positive)
        return AffineTransform(loc, scale, domain=constraints.real_vector)

    def median(self, params):
        transform = handlers.substitute(self._get_transform, params)()
        return self._unpack_and_constrain(transform.loc, params)

    def quantiles(self, params, quantiles):
        transform = handlers.substitute(self._get_transform, params)()
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(transform.loc, transform.scale).icdf(quantiles)
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
    def _get_transform(self):
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        scale_tril = numpyro.param('{}_scale_tril'.format(self.prefix), np.identity(self.latent_size),
                                   constraint=constraints.lower_cholesky)
        return MultivariateAffineTransform(loc, scale_tril)

    def median(self, params):
        transform = handlers.substitute(self._get_transform, params)()
        return self._unpack_and_constrain(transform.loc, params)

    def quantiles(self, params, quantiles):
        transform = handlers.substitute(self._get_transform, params)()
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
    def _sample_latent(self, base_dist, *args, **kwargs):
        # sample from Delta guide
        sample_shape = kwargs.pop('sample_shape', ())
        loc = numpyro.param('{}_loc'.format(self.prefix), self._init_latent)
        posterior = dist.Delta(loc, event_ndim=1)
        return numpyro.sample("_{}_latent".format(self.prefix), posterior, sample_shape=sample_shape)

    def _get_transform(self, params):
        def loss_fn(z):
            params1 = params.copy()
            params1['{}_loc'.format(self.prefix)] = z
            # we are doing maximum likelihood, so only require `num_particles=1` and an arbitrary rng.
            return AutoContinuousELBO().loss(random.PRNGKey(0), params1, self.model, self,
                                             *self._args, **self._kwargs)

        loc = params['{}_loc'.format(self.prefix)]
        precision = hessian(loss_fn)(loc)
        scale_tril = cholesky_inverse(precision)
        # fallback to AutoDelta if hessian is singular
        scale_tril = np.where(np.isnan(scale_tril), 0., scale_tril)
        return MultivariateAffineTransform(loc, scale_tril)

    def sample_posterior(self, rng, params, sample_shape=()):
        transform = self._get_transform(params)
        loc, scale_tril = transform.loc, transform.scale_tril
        latent_sample = dist.MultivariateNormal(loc, scale_tril=scale_tril).sample(rng, sample_shape)
        return self._unpack_and_constrain(latent_sample, params)

    def get_transform(self, params):
        return ComposeTransform([self._get_transform(params),
                                 UnpackTransform(self._unpack_latent)])

    def median(self, params):
        loc = params['{}_loc'.format(self.prefix)]
        return self._unpack_and_constrain(loc, params)

    def quantiles(self, params, quantiles):
        transform = self._get_transform(params)
        quantiles = np.array(quantiles)[..., None]
        latent = dist.Normal(transform.loc, np.diagonal(transform.scale_tril)).icdf(quantiles)
        return self._unpack_and_constrain(latent, params)


class AutoIAFNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a
    :class:`~numpyro.distributions.iaf.InverseAutoregressiveTransform`
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoIAFNormal(model, hidden_dims=[20], skip_connections=True, ...)
        svi = SVI(model, guide, ...)

    :param jax.random.PRNGKey rng: random key to be used as the source of randomness
        to initialize the guide.
    :param callable model: a generative model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_strategy: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param `**arn_kwargs`: keywords for constructing autoregressive neural networks, which includes:

        * **hidden_dims** (``list[int]``) - the dimensionality of the hidden units per layer.
          Defaults to ``[latent_size, latent_size]``.
        * **skip_connections** (``bool``) - whether to add skip connections from the input to the
          output of each flow. Defaults to False.
        * **nonlinearity** (``callable``) - the nonlinearity to use in the feedforward network.
          Defaults to :func:`jax.experimental.stax.Relu`.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform(),
                 num_flows=3, **arn_kwargs):
        self.num_flows = num_flows
        # 2-layer, stax.Elu, skip_connections=False by default following the experiments in
        # IAF paper (https://arxiv.org/abs/1606.04934)
        # and Neutra paper (https://arxiv.org/abs/1903.03704)
        self._hidden_dims = arn_kwargs.get('hidden_dims')
        self._skip_connections = arn_kwargs.get('skip_connections', False)
        self._nonlinearity = arn_kwargs.get('nonlinearity', stax.Elu)
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


class AutoBNAFNormal(AutoContinuous):
    """
    This implementation of :class:`AutoContinuous` uses a Diagonal Normal
    distribution transformed via a
    :class:`~numpyro.distributions.iaf.InverseAutoregressiveTransform`
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    Usage::

        guide = AutoBNAFNormal(rng, model, get_params, num_flows=1, hidden_factors=[50, 50])
        svi = SVI(model, guide, ...)

    **References**

    1. *Block Neural Autoregressive Flow*,
       Nicola De Cao, Ivan Titov, Wilker Aziz

    :param jax.random.PRNGKey rng: random key to be used as the source of randomness
        to initialize the guide.
    :param callable model: a generative model.
    :param str prefix: a prefix that will be prefixed to all param internal sites.
    :param callable init_strategy: A per-site initialization function.
    :param int num_flows: the number of flows to be used, defaults to 3.
    :param list hidden_factors: Hidden layer i has ``hidden_factors[i]`` hidden units per
        input dimension. This corresponds to both :math:`a` and :math:`b` in reference [1].
        The elements of hidden_factors must be integers.
    """
    def __init__(self, model, prefix="auto", init_strategy=init_to_uniform(), num_flows=1,
                 hidden_factors=[50, 50]):
        self.num_flows = num_flows
        self._hidden_factors = hidden_factors
        super(AutoBNAFNormal, self).__init__(model, prefix=prefix, init_strategy=init_strategy)

    def _get_transform(self):
        if self.latent_size == 1:
            raise ValueError('latent dim = 1. Consider using AutoDiagonalNormal instead')
        flows = []
        for i in range(self.num_flows):
            if i > 0:
                flows.append(PermuteTransform(np.arange(self.latent_size)[::-1]))
            residual = "gated" if i < (self.num_flows - 1) else None
            arn = BlockNeuralAutoregressiveNN(self.latent_size, self._hidden_factors, residual)
            arnn = numpyro.module('{}_arn__{}'.format(self.prefix, i), arn, (self.latent_size,))
            flows.append(BlockNeuralAutoregressiveTransform(arnn))
        return ComposeTransform(flows)


class AutoContinuousELBO(ELBO):
    """
    An ELBO implementation specific to :class:`AutoContinuous` guides. In those guide, the latent
    variables of the model are transformed to unconstrained domains. This class provides ELBO of
    the "transformed" model (i.e. the corresponding model with unconstrained variables) and the
    guide.
    """
    def loss(self, rng, param_map, model, guide, *args, **kwargs):
        assert isinstance(guide, AutoContinuous)

        def single_particle_elbo(rng):
            model_seed, guide_seed = random.split(rng)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(seeded_guide, args, kwargs, param_map)
            # first, we substitute `param_map` to `param` primitives of `model`
            seeded_model = substitute(seeded_model, param_map)
            # then creates a new `param_map` which holds base values of `sample` primitives
            base_param_map = {}
            # in autoguide, a site's value holds intermediate value
            for name, site in guide_trace.items():
                if site['type'] == 'sample':
                    base_param_map[name] = site['value']
            model_log_density, _ = log_density(seeded_model, args, kwargs, base_param_map,
                                               skip_dist_transforms=True)

            # log p(z) - log q(z)
            elbo = model_log_density - guide_log_density
            # Return (-elbo) since by convention we do gradient descent on a loss and
            # the ELBO is a lower bound that needs to be maximized.
            return -elbo

        if self.num_particles == 1:
            return single_particle_elbo(rng)
        else:
            rngs = random.split(rng, self.num_particles)
            return np.mean(vmap(single_particle_elbo)(rngs))
