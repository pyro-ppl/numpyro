# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike

from numpyro._typing import ConstraintT, DistributionT
import numpyro.distributions as dist


# Test class that properly implements DistributionLike
class ValidDistributionLike:
    arg_constraints: dict[str, ConstraintT] = {"", dist.constraints.real}
    support: ConstraintT = dist.constraints.real
    has_enumerate_support: bool = False
    reparametrized_params: list[str] = [""]
    _validate_args: bool = True
    pytree_data_fields: tuple = tuple()
    pytree_aux_fields: tuple = tuple()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return ()

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def event_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def sample(self, key, sample_shape=()):
        return jnp.array(0.0)

    def log_prob(self, value):
        return jnp.array(0.0)

    @property
    def mean(self):
        return jnp.array(0.0)

    @property
    def variance(self):
        return jnp.array(1.0)

    def cdf(self, value):
        return jnp.array(0.5)

    def icdf(self, q):
        return jnp.array(0.0)

    def rsample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.sample(key, sample_shape)

    def entropy(self) -> ArrayLike:
        return jnp.array(0.0)

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        return jnp.array([])

    def shape(self, sample_shape: tuple[int, ...] = ()) -> tuple[int, ...]:
        return sample_shape + self.batch_shape + self.event_shape

    @property
    def has_rsample(self) -> bool:
        return True

    @property
    def is_discrete(self) -> bool:
        return False


# Test class missing required methods
class InvalidDistributionLike:
    @property
    def batch_shape(self):
        return ()


def test_valid_distribution_implementations():
    """Test that valid implementations are recognized as DistributionLike"""
    # Test standard NumPyro distribution
    assert isinstance(dist.Normal(0, 1), DistributionT)

    # Test custom implementation
    assert isinstance(ValidDistributionLike(), DistributionT)


def test_invalid_distribution_implementations():
    """Test that invalid implementations are not recognized as DistributionLike"""
    assert not isinstance(InvalidDistributionLike(), DistributionT)
    assert not isinstance(object(), DistributionT)


def test_distribution_like_interface():
    """Test that we can use a custom DistributionLike implementation where a Distribution is expected"""
    my_dist = ValidDistributionLike()

    # Test basic properties
    assert my_dist() == ()
    assert my_dist.batch_shape == ()
    assert my_dist.event_shape == ()
    assert my_dist.event_dim == 0

    # Test methods
    key = PRNGKey(0)
    sample = my_dist.sample(key)
    assert isinstance(sample, jnp.ndarray)

    log_prob = my_dist.log_prob(0.0)
    assert isinstance(log_prob, jnp.ndarray)

    mean = my_dist.mean
    assert isinstance(mean, jnp.ndarray)

    var = my_dist.variance
    assert isinstance(var, jnp.ndarray)

    cdf = my_dist.cdf(0.0)
    assert isinstance(cdf, jnp.ndarray)

    icdf = my_dist.icdf(0.5)
    assert isinstance(icdf, jnp.ndarray)


def test_distribution_like_with_shapes():
    """Test a DistributionLike implementation with non-trivial shapes"""

    class ShapedDistributionLike:
        @property
        def batch_shape(self):
            return (2, 3)

        @property
        def event_shape(self):
            return (4,)

        @property
        def event_dim(self):
            return 1

        def sample(self, key, sample_shape=()):
            shape = sample_shape + self.batch_shape + self.event_shape
            return jnp.zeros(shape)

        def log_prob(self, value):
            return jnp.zeros(self.batch_shape)

        @property
        def mean(self):
            return jnp.zeros(self.batch_shape + self.event_shape)

        @property
        def variance(self):
            return jnp.ones(self.batch_shape + self.event_shape)

        def cdf(self, value):
            return jnp.full(self.batch_shape, 0.5)

        def icdf(self, q):
            return jnp.zeros(self.batch_shape + self.event_shape)

    my_dist = ShapedDistributionLike()

    assert my_dist.batch_shape == (2, 3)
    assert my_dist.event_shape == (4,)
    assert my_dist.event_dim == 1

    key = PRNGKey(0)
    sample = my_dist.sample(key, sample_shape=(5,))
    assert sample.shape == (5, 2, 3, 4)

    log_prob = my_dist.log_prob(jnp.zeros((2, 3, 4)))
    assert log_prob.shape == (2, 3)

    assert my_dist.mean.shape == (2, 3, 4)
    assert my_dist.variance.shape == (2, 3, 4)
