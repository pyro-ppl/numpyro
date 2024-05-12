# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import lax
import jax.numpy as jnp

from numpyro.distributions import Distribution, constraints
from numpyro.distributions.discrete import CategoricalLogits, CategoricalProbs
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


def Mixture(mixing_distribution, component_distributions, *, validate_args=None):
    """
    A marginalized finite mixture of component distributions

    The returned distribution will be either a:

    1. :class:`~numpyro.distributions.MixtureGeneral`, when
       ``component_distributions`` is a list, or
    2. :class:`~numpyro.distributions.MixtureSameFamily`, when
       ``component_distributions`` is a single distribution.

    and more details can be found in the documentation for each of these
    classes.

    :param mixing_distribution: A :class:`~numpyro.distributions.Categorical`
        specifying the weights for each mixture components. The size of this
        distribution specifies the number of components in the mixture,
        ``mixture_size``.
    :param component_distributions: Either a list of component distributions or
        a single vectorized distribution. When a list is provided, the number of
        elements must equal ``mixture_size``. Otherwise, the last batch
        dimension of the distribution must equal ``mixture_size``.
    :return: The mixture distribution.
    """
    if isinstance(component_distributions, Distribution):
        return MixtureSameFamily(
            mixing_distribution, component_distributions, validate_args=validate_args
        )
    return MixtureGeneral(
        mixing_distribution, component_distributions, validate_args=validate_args
    )


class _MixtureBase(Distribution):
    """An abstract base class for mixture distributions

    This consolidates all the shared logic for the mixture distributions, and
    subclasses should implement the ``component_*`` methods to specialize.
    """

    @property
    def component_mean(self):
        raise NotImplementedError

    @property
    def component_variance(self):
        raise NotImplementedError

    def component_log_probs(self, value):
        raise NotImplementedError

    def component_sample(self, key, sample_shape=()):
        raise NotImplementedError

    def component_cdf(self, samples):
        raise NotImplementedError

    @property
    def mixture_size(self):
        """The number of components in the mixture"""
        return self._mixture_size

    @property
    def mixing_distribution(self):
        """The ``Categorical`` distribution over components"""
        return self._mixing_distribution

    @property
    def mixture_dim(self):
        return -self.event_dim - 1

    @property
    def mean(self):
        probs = self.mixing_distribution.probs
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        weighted_component_means = probs * self.component_mean
        return jnp.sum(weighted_component_means, axis=self.mixture_dim)

    @property
    def variance(self):
        probs = self.mixing_distribution.probs
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        mean_cond_var = jnp.sum(probs * self.component_variance, axis=self.mixture_dim)
        sq_deviation = (
            self.component_mean - jnp.expand_dims(self.mean, axis=self.mixture_dim)
        ) ** 2
        var_cond_mean = jnp.sum(probs * sq_deviation, axis=self.mixture_dim)
        return mean_cond_var + var_cond_mean

    def cdf(self, samples):
        """The cumulative distribution function

        :param value: samples from this distribution.
        :return: output of the cumulative distribution function evaluated at
            `value`.
        :raises: NotImplementedError if the component distribution does not
            implement the cdf method.
        """
        cdf_components = self.component_cdf(samples)
        return jnp.sum(cdf_components * self.mixing_distribution.probs, axis=-1)

    def sample_with_intermediates(self, key, sample_shape=()):
        """
        A version of ``sample`` that also returns the sampled component indices

        :param jax.random.PRNGKey key: the rng_key key to be used for the
            distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: A 2-element tuple with the samples from the distribution, and
            the indices of the sampled components.
        :rtype: tuple
        """
        assert is_prng_key(key)
        key_comp, key_ind = jax.random.split(key)
        samples = self.component_sample(key_comp, sample_shape=sample_shape)

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
        return jnp.squeeze(samples_selected, axis=self.mixture_dim), [indices]

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]

    @validate_sample
    def log_prob(self, value, intermediates=None):
        del intermediates
        sum_log_probs = self.component_log_probs(value)
        return jax.nn.logsumexp(sum_log_probs, axis=-1)


class MixtureSameFamily(_MixtureBase):
    """
    A finite mixture of component distributions from the same family

    This mixture only supports a mixture of component distributions that are all
    of the same family. The different components are specified along the last
    batch dimension of the input ``component_distribution``. If you need a
    mixture of distributions from different families, use the more general
    implementation in :class:`~numpyro.distributions.MixtureGeneral`.

    :param mixing_distribution: A :class:`~numpyro.distributions.Categorical`
        specifying the weights for each mixture components. The size of this
        distribution specifies the number of components in the mixture,
        ``mixture_size``.
    :param component_distribution: A single vectorized
        :class:`~numpyro.distributions.Distribution`, whose last batch dimension
        equals ``mixture_size`` as specified by ``mixing_distribution``.

    **Example**

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

    pytree_data_fields = ("_mixing_distribution", "_component_distribution")
    pytree_aux_fields = ("_mixture_size",)

    def __init__(
        self, mixing_distribution, component_distribution, *, validate_args=None
    ):
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
            f"needs to correspond to the mixture_size={mixture_size}!"
        )
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

    @property
    def component_mean(self):
        return self.component_distribution.mean

    @property
    def component_variance(self):
        return self.component_distribution.variance

    def component_cdf(self, samples):
        return self.component_distribution.cdf(
            jnp.expand_dims(samples, axis=self.mixture_dim)
        )

    def component_sample(self, key, sample_shape=()):
        return self.component_distribution.expand(
            sample_shape + self.batch_shape + (self.mixture_size,)
        ).sample(key)

    def component_log_probs(self, value):
        value = jnp.expand_dims(value, self.mixture_dim)
        component_log_probs = self.component_distribution.log_prob(value)
        return jax.nn.log_softmax(self.mixing_distribution.logits) + component_log_probs


class MixtureGeneral(_MixtureBase):
    """
    A finite mixture of component distributions from different families

    If all of the component distributions are from the same family, the more
    specific implementation in :class:`~numpyro.distributions.MixtureSameFamily`
    will be somewhat more efficient.

    :param mixing_distribution: A :class:`~numpyro.distributions.Categorical`
        specifying the weights for each mixture components. The size of this
        distribution specifies the number of components in the mixture,
        ``mixture_size``.
    :param component_distributions: A list of ``mixture_size``
        :class:`~numpyro.distributions.Distribution` objects.
    :param support: A :class:`~numpyro.distributions.constraints.Constraint`
        object specifying the support of the mixture distribution. If not
        provided, the support will be inferred from the component distributions.

    **Example**

    .. doctest::

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import numpyro.distributions as dist
       >>> mixing_dist = dist.Categorical(probs=jnp.ones(3) / 3.)
       >>> component_dists = [
       ...     dist.Normal(loc=0.0, scale=1.0),
       ...     dist.Normal(loc=-0.5, scale=0.3),
       ...     dist.Normal(loc=0.6, scale=1.2),
       ... ]
       >>> mixture = dist.MixtureGeneral(mixing_dist, component_dists)
       >>> mixture.sample(jax.random.PRNGKey(42)).shape
       ()

    .. doctest::

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import numpyro.distributions as dist
       >>> mixing_dist = dist.Categorical(probs=jnp.ones(2) / 2.)
       >>> component_dists = [
       ...     dist.Normal(loc=0.0, scale=1.0),
       ...     dist.HalfNormal(scale=0.3),
       ... ]
       >>> mixture = dist.MixtureGeneral(mixing_dist, component_dists, support=dist.constraints.real)
       >>> mixture.sample(jax.random.PRNGKey(42)).shape
       ()
    """

    pytree_data_fields = (
        "_mixing_distribution",
        "_component_distributions",
        "_support",
    )
    pytree_aux_fields = ("_mixture_size",)

    def __init__(
        self,
        mixing_distribution,
        component_distributions,
        *,
        support=None,
        validate_args=None,
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
                    "All elements of 'component_distributions' must be instances of "
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
        self._support = support
        if support is None:
            support_type = type(component_distributions[0].support)
            if any(
                type(d.support) is not support_type for d in component_distributions[1:]
            ):
                raise ValueError(
                    "All component distributions must have the same support."
                )
        else:
            assert isinstance(
                support, constraints.Constraint
            ), "support must be a Constraint object"

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
    def component_distributions(self):
        """The list of component distributions in the mixture

        :return: The list of component distributions
        :rtype: list[Distribution]
        """
        return self._component_distributions

    @constraints.dependent_property
    def support(self):
        if self._support is not None:
            return self._support
        return self.component_distributions[0].support

    @property
    def is_discrete(self):
        return self.component_distributions[0].is_discrete

    @property
    def component_mean(self):
        return jnp.stack(
            [d.mean for d in self.component_distributions], axis=self.mixture_dim
        )

    @property
    def component_variance(self):
        return jnp.stack(
            [d.variance for d in self.component_distributions], axis=self.mixture_dim
        )

    def component_cdf(self, samples):
        return jnp.stack(
            [d.cdf(samples) for d in self.component_distributions],
            axis=self.mixture_dim,
        )

    def component_sample(self, key, sample_shape=()):
        keys = jax.random.split(key, self.mixture_size)
        samples = []
        for k, d in zip(keys, self.component_distributions):
            samples.append(d.expand(sample_shape + self.batch_shape).sample(k))
        return jnp.stack(samples, axis=self.mixture_dim)

    def component_log_probs(self, value):
        component_log_probs = []
        for d in self.component_distributions:
            log_prob = d.log_prob(value)
            if (self._support is not None) and (not d._validate_args):
                mask = d.support(value)
                log_prob = jnp.where(mask, log_prob, -jnp.inf)
            component_log_probs.append(log_prob)
        component_log_probs = jnp.stack(component_log_probs, axis=-1)
        return jax.nn.log_softmax(self.mixing_distribution.logits) + component_log_probs


def _check_mixing_distribution(mixing_distribution):
    if not isinstance(mixing_distribution, (CategoricalLogits, CategoricalProbs)):
        raise ValueError(
            "The mixing distribution must be a numpyro.distributions.Categorical. "
            f"However, it is of type {type(mixing_distribution)}"
        )
