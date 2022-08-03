# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import lax
import jax.numpy as jnp

from numpyro.distributions import Distribution, constraints
from numpyro.distributions.discrete import CategoricalLogits, CategoricalProbs
from numpyro.distributions.util import is_prng_key, validate_sample


def Mixture(mixing_distribution, component_distributions, *, validate_args=None):
    """
    Mixture distribution.

    :param numpyro.distribution.Distribution mixing_distribution:
        The mixing distribution to select the components. Must be a Categorical.
    :param distributions: Either a list of component distributions or a single vectorized distribution.
    :return: Either a MixtureGeneral or MixtureSameFamily distribution, depending on the type of
        `distributions`.
    """
    if isinstance(component_distributions, Distribution):
        return MixtureSameFamily(
            mixing_distribution, component_distributions, validate_args=validate_args
        )
    return MixtureGeneral(
        mixing_distribution, component_distributions, validate_args=validate_args
    )


class MixtureSameFamily(Distribution):
    """
    Marginalized Finite Mixture distribution of vectorized components.

    The components being a vectorized distribution implies that all components are from the same family,
    represented by a single Distribution object.

    :param numpyro.distribution.Distribution mixing_distribution:
        The mixing distribution to select the components. Needs to be a categorical.
    :param numpyro.distribution.Distribution component_distribution:
        Vectorized component distribution.

    As an example:

    .. doctest::

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import numpyro.distributions as dist
       >>> mixing_dist = dist.Categorical(probs=jnp.ones(3) / 3.)
       >>> component_dist = dist.Normal(loc=jnp.zeros(3), scale=jnp.ones(3))
       >>> mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
       >>> mixture.sample(jax.random.PRNGKey(42)).shape
       ()
    """

    def __init__(
        self, mixing_distribution, component_distribution, *, validate_args=None
    ):
        # Check arguments
        _check_mixing_distribution(mixing_distribution)

        mixture_size = mixing_distribution.probs.shape[-1]
        if not isinstance(component_distribution, Distribution):
            raise ValueError(
                "The component distribution need to be a numpyro.distributions.Distribution. "
                f"However, it is of type {type(component_distribution)}"
            )
        assert component_distribution.batch_shape[-1] == mixture_size, (
            "Component distribution batch shape last dimension "
            f"(size={component_distribution.batch_shape[-1]}) "
            "needs to correspond to the mixture_size={mixture_size}!"
        )
        # Assign checked arguments
        self._mixing_distribution = mixing_distribution
        self._component_distribution = component_distribution
        self._mixture_size = mixture_size
        batch_shape = lax.broadcast_shapes(
            mixing_distribution.batch_shape,
            component_distribution.batch_shape[:-1],  # Without probabilities
        )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=component_distribution.event_shape,
            validate_args=validate_args,
        )

    @property
    def mixture_size(self):
        """
        Returns the number of distributions in the mixture

        :return: number of mixtures.
        :rtype: int
        """
        return self._mixture_size

    @property
    def mixing_distribution(self):
        """
        Returns the mixing distribution

        :return: Categorical distribution
        :rtype: Categorical
        """
        return self._mixing_distribution

    @property
    def mixture_dim(self):
        return -self.event_dim - 1

    @property
    def component_distribution(self):
        """
        Return the vectorized distribution of components being mixed.

        :return: Component distribution
        :rtype: Distribution
        """
        return self._component_distribution

    @constraints.dependent_property
    def support(self):
        return self.component_distribution.support

    @property
    def is_discrete(self):
        return self.component_distribution.is_discrete

    def tree_flatten(self):
        mixing_flat, mixing_aux = self.mixing_distribution.tree_flatten()
        component_flat, component_aux = self.component_distribution.tree_flatten()
        params = (mixing_flat, component_flat)
        aux_data = (
            (type(self.mixing_distribution), type(self.component_distribution)),
            (mixing_aux, component_aux),
        )
        return params, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        mixing_params, component_params = params
        child_clss, child_aux = aux_data
        mixing_cls, component_cls = child_clss
        mixing_aux, component_aux = child_aux
        mixing_dist = mixing_cls.tree_unflatten(mixing_aux, mixing_params)
        component_dist = component_cls.tree_unflatten(component_aux, component_params)
        return cls(
            mixing_distribution=mixing_dist, component_distribution=component_dist
        )

    @property
    def mean(self):
        probs = self.mixing_distribution.probs
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        weighted_component_means = probs * self.component_distribution.mean
        return jnp.sum(weighted_component_means, axis=self.mixture_dim)

    @property
    def variance(self):
        probs = self.mixing_distribution.probs
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        # E[Var(Y|X)]
        mean_cond_var = jnp.sum(
            probs * self.component_distribution.variance, axis=self.mixture_dim
        )
        # Variance is the expectation of the squared deviation of a random variable from its mean
        sq_deviation = (
            self.component_distribution.mean
            - jnp.expand_dims(self.mean, axis=self.mixture_dim)
        ) ** 2
        # Var(E[Y|X])
        var_cond_mean = jnp.sum(probs * sq_deviation, axis=self.mixture_dim)
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        return mean_cond_var + var_cond_mean

    def cdf(self, samples):
        """
        The cumulative distribution function of this mixture distribution.

        :param value: samples from this distribution.
        :return: output of the cummulative distribution function evaluated at `value`.
        :raises: NotImplementedError if the component distribution does not implement the cdf method.
        """
        cdf_components = self.component_distribution.cdf(
            jnp.expand_dims(samples, axis=self.mixture_dim)
        )
        return jnp.sum(cdf_components * self.mixing_distribution.probs, axis=-1)

    def sample_with_intermediates(self, key, sample_shape=()):
        """
        Same as ``sample`` except that the sampled mixture components are also returned.

        :param jax.random.PRNGKey key: the rng_key key to be used for the distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: Tuple (samples, indices)
        :rtype: tuple
        """
        assert is_prng_key(key)
        key_comp, key_ind = jax.random.split(key)
        # Samples from component distribution will have shape:
        #  (*sample_shape, *batch_shape, mixture_size, *event_shape)
        samples = self.component_distribution.expand(
            sample_shape + self.batch_shape + (self.mixture_size,)
        ).sample(key_comp)
        # Sample selection indices from the categorical (shape will be sample_shape)
        indices = self.mixing_distribution.expand(
            sample_shape + self.batch_shape
        ).sample(key_ind)
        n_expand = self.event_dim + 1
        indices_expanded = indices.reshape(indices.shape + (1,) * n_expand)
        # Select samples according to indices samples from categorical
        samples_selected = jnp.take_along_axis(
            samples, indices=indices_expanded, axis=self.mixture_dim
        )
        # Final sample shape (*sample_shape, *batch_shape, *event_shape)
        return jnp.squeeze(samples_selected, axis=self.mixture_dim), indices

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]

    @validate_sample
    def log_prob(self, value, intermediates=None):
        del intermediates
        value = jnp.expand_dims(value, self.mixture_dim)
        component_log_probs = self.component_distribution.log_prob(value)
        sum_log_probs = self.mixing_distribution.logits + component_log_probs
        return jax.nn.logsumexp(sum_log_probs, axis=-1)


class MixtureGeneral(Distribution):
    def __init__(
        self, mixing_distribution, component_distributions, *, validate_args=None
    ):
        _check_mixing_distribution(mixing_distribution)

        self._mixture_size = jnp.shape(mixing_distribution.probs)[-1]
        try:
            component_distributions = list(component_distributions)
        except TypeError:
            raise ValueError(
                "The 'component_distributions' argument must be a list of Distribution objects"
            )
        for d in component_distributions:
            if not isinstance(d, Distribution):
                raise ValueError(
                    "All elements of 'component_distributions' must be instaces of "
                    "numpyro.distributions.Distribution subclasses"
                )
        if len(component_distributions) != self.mixture_size:
            raise ValueError(
                "The number of elements in 'component_distributions' must match the mixture size; "
                f"expected {self._mixture_size}, got {len(component_distributions)}"
            )

        # TODO: It would be good to check that the support of all the component
        # distributions match, but for now we just check the type, since __eq__
        # isn't consistently implemented for all support types.
        support_type = type(component_distributions[0].support)
        if any(type(d.support) != support_type for d in component_distributions[1:]):
            raise ValueError("All component distributions must have the same support.")

        self._mixing_distribution = mixing_distribution
        self._component_distributions = component_distributions

        batch_shape = lax.broadcast_shapes(
            mixing_distribution.batch_shape,
            *(d.batch_shape for d in component_distributions),
        )
        event_shape = component_distributions[0].event_shape
        for d in component_distributions[1:]:
            if d.event_shape != event_shape:
                raise ValueError(
                    "All component distributions must have the same event shape"
                )

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def mixture_size(self):
        """
        The number of distributions in the mixture

        :return: number of mixtures.
        :rtype: int
        """
        return self._mixture_size

    @property
    def mixing_distribution(self):
        """
        The mixing distribution

        :return: Categorical distribution
        :rtype: Categorical
        """
        return self._mixing_distribution

    @property
    def component_distributions(self):
        """
        The list of component distributions being mixed

        :return: The list of component distributions
        :rtype: List[Distribution]
        """
        return self._component_distributions

    @property
    def mixture_dim(self):
        return -self.event_dim - 1

    @constraints.dependent_property
    def support(self):
        return self.component_distributions[0].support

    @property
    def is_discrete(self):
        return self.component_distributions[0].is_discrete

    def tree_flatten(self):
        mixing_flat, mixing_aux = self.mixing_distribution.tree_flatten()
        dists_flat, dists_aux = zip(
            *(d.tree_flatten() for d in self.component_distributions)
        )
        params = (mixing_flat, dists_flat)
        aux_data = (
            (
                type(self.mixing_distribution),
                tuple(type(d) for d in self.component_distributions),
            ),
            (mixing_aux, dists_aux),
        )
        return params, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        params_mix, params_dists = params
        (cls_mix, cls_dists), (mixing_aux, dists_aux) = aux_data
        mixing_dist = cls_mix.tree_unflatten(mixing_aux, params_mix)
        distributions = [
            c.tree_unflatten(a, p)
            for c, a, p in zip(cls_dists, dists_aux, params_dists)
        ]
        return cls(
            mixing_distribution=mixing_dist, component_distributions=distributions
        )

    @property
    def mean(self):
        probs = self.mixing_distribution.probs
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        means = jnp.stack(
            [d.mean for d in self.component_distributions], axis=self.mixture_dim
        )
        return jnp.sum(probs * means, axis=self.mixture_dim)

    @property
    def variance(self):
        probs = self.mixing_distribution.probs
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)

        means = jnp.stack(
            [d.mean for d in self.component_distributions], axis=self.mixture_dim
        )
        variances = jnp.stack(
            [d.variance for d in self.component_distributions], axis=self.mixture_dim
        )

        # E[Var(Y|X)]
        mean_cond_var = jnp.sum(probs * variances, axis=self.mixture_dim)
        # Variance is the expectation of the squared deviation of a random variable from its mean
        sq_deviation = (means - jnp.expand_dims(self.mean, axis=self.mixture_dim)) ** 2
        # Var(E[Y|X])
        var_cond_mean = jnp.sum(probs * sq_deviation, axis=self.mixture_dim)
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        return mean_cond_var + var_cond_mean

    def cdf(self, samples):
        """
        The cumulative distribution function of this mixture distribution.

        :param value: samples from this distribution.
        :return: output of the cummulative distribution function evaluated at `value`.
        :raises: NotImplementedError if the component distribution does not implement the cdf method.
        """
        cdfs = jnp.stack(
            [d.cdf(samples) for d in self.component_distributions],
            axis=self.mixture_dim,
        )
        return jnp.sum(cdfs * self.mixing_distribution.probs, axis=-1)

    def sample_with_intermediates(self, key, sample_shape=()):
        assert is_prng_key(key)

        key_ind, *key_comp = jax.random.split(
            key, 1 + len(self.component_distributions)
        )

        samples = []
        for k, d in zip(key_comp, self.component_distributions):
            samples.append(d.expand(sample_shape + self.batch_shape).sample(k))
        samples = jnp.stack(samples, axis=self.mixture_dim)

        indices = self.mixing_distribution.expand(
            sample_shape + self.batch_shape
        ).sample(key_ind)
        n_expand = self.event_dim + 1
        indices_expanded = indices.reshape(indices.shape + (1,) * n_expand)

        samples_selected = jnp.take_along_axis(
            samples, indices=indices_expanded, axis=self.mixture_dim
        )

        return jnp.squeeze(samples_selected, axis=self.mixture_dim), indices

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]

    @validate_sample
    def component_log_probs(self, value):
        component_log_probs = jnp.stack(
            [d.log_prob(value) for d in self.component_distributions],
            axis=self.mixture_dim,
        )
        return self.mixing_distribution.logits + component_log_probs

    @validate_sample
    def log_prob(self, value, intermediates=None):
        del intermediates
        log_probs = self.component_log_probs(value)
        return jax.nn.logsumexp(log_probs, axis=self.mixture_dim)


def _check_mixing_distribution(mixing_distribution):
    if not isinstance(mixing_distribution, (CategoricalLogits, CategoricalProbs)):
        raise ValueError(
            "The mixing distribution must be a numpyro.distributions.Categorical. "
            f"However, it is of type {type(mixing_distribution)}"
        )
