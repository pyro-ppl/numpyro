# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# The implementation follows the design in PyTorch: torch.distributions.distribution.py
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from collections import OrderedDict
from contextlib import contextmanager
import functools
import warnings

import numpy as np

from jax import lax, tree_util
import jax.numpy as jnp

from numpyro.distributions.constraints import is_dependent, real
from numpyro.distributions.transforms import ComposeTransform, Transform
from numpyro.distributions.util import lazy_property, promote_shapes, sum_rightmost, validate_sample
from numpyro.util import not_jax_tracer

_VALIDATION_ENABLED = False


def enable_validation(is_validate=True):
    """
    Enable or disable validation checks in NumPyro. Validation checks provide useful warnings and
    errors, e.g. NaN checks, validating distribution arguments and support values, etc. which is
    useful for debugging.

    .. note:: This utility does not take effect under JAX's JIT compilation or vectorized
        transformation :func:`jax.vmap`.

    :param bool is_validate: whether to enable validation checks.
    """
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = is_validate
    Distribution.set_default_validate_args(is_validate)


@contextmanager
def validation_enabled(is_validate=True):
    """
    Context manager that is useful when temporarily enabling/disabling validation checks.

    :param bool is_validate: whether to enable validation checks.
    """
    distribution_validation_status = _VALIDATION_ENABLED
    try:
        enable_validation(is_validate)
        yield
    finally:
        enable_validation(distribution_validation_status)


COERCIONS = []


class DistributionMeta(type):
    def __call__(cls, *args, **kwargs):
        for coerce_ in COERCIONS:
            result = coerce_(cls, args, kwargs)
            if result is not None:
                return result
        return super().__call__(*args, **kwargs)

    @property
    def __wrapped__(cls):
        return functools.partial(cls.__init__, None)


class Distribution(metaclass=DistributionMeta):
    """
    Base class for probability distributions in NumPyro. The design largely
    follows from :mod:`torch.distributions`.

    :param batch_shape: The batch shape for the distribution. This designates
        independent (possibly non-identical) dimensions of a sample from the
        distribution. This is fixed for a distribution instance and is inferred
        from the shape of the distribution parameters.
    :param event_shape: The event shape for the distribution. This designates
        the dependent dimensions of a sample from the distribution. These are
        collapsed when we evaluate the log probability density of a batch of
        samples using `.log_prob`.
    :param validate_args: Whether to enable validation of distribution
        parameters and arguments to `.log_prob` method.

    As an example:

    .. doctest::

       >>> import jax.numpy as jnp
       >>> import numpyro.distributions as dist
       >>> d = dist.Dirichlet(jnp.ones((2, 3, 4)))
       >>> d.batch_shape
       (2, 3)
       >>> d.event_shape
       (4,)
    """
    arg_constraints = {}
    support = None
    has_enumerate_support = False
    is_discrete = False
    reparametrized_params = []
    _validate_args = False

    # register Distribution as a pytree
    # ref: https://github.com/google/jax/issues/2916
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tree_util.register_pytree_node(cls,
                                       cls.tree_flatten,
                                       cls.tree_unflatten)

    def tree_flatten(self):
        return tuple(getattr(self, param) for param in sorted(self.arg_constraints.keys())), None

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(**dict(zip(sorted(cls.arg_constraints.keys()), params)))

    @staticmethod
    def set_default_validate_args(value):
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            for param, constraint in self.arg_constraints.items():
                if param not in self.__dict__ and isinstance(getattr(type(self), param), lazy_property):
                    continue
                if is_dependent(constraint):
                    continue  # skip constraints that cannot be checked
                is_valid = constraint(getattr(self, param))
                if not_jax_tracer(is_valid):
                    if not np.all(is_valid):
                        raise ValueError("{} distribution got invalid {} parameter.".format(
                            self.__class__.__name__, param))
        super(Distribution, self).__init__()

    @property
    def batch_shape(self):
        """
        Returns the shape over which the distribution parameters are batched.

        :return: batch shape of the distribution.
        :rtype: tuple
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """
        Returns the shape of a single sample from the distribution without
        batching.

        :return: event shape of the distribution.
        :rtype: tuple
        """
        return self._event_shape

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)

    def shape(self, sample_shape=()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape::

            d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

        :param tuple sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :return: shape of samples.
        :rtype: tuple
        """
        return sample_shape + self.batch_shape + self.event_shape

    def sample(self, key, sample_shape=()):
        """
        Returns a sample from the distribution having shape given by
        `sample_shape + batch_shape + event_shape`. Note that when `sample_shape` is non-empty,
        leading dimensions (of size `sample_shape`) of the returned sample will
        be filled with iid draws from the distribution instance.

        :param jax.random.PRNGKey key: the rng_key key to be used for the distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: an array of shape `sample_shape + batch_shape + event_shape`
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    def sample_with_intermediates(self, key, sample_shape=()):
        """
        Same as ``sample`` except that any intermediate computations are
        returned (useful for `TransformedDistribution`).

        :param jax.random.PRNGKey key: the rng_key key to be used for the distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: an array of shape `sample_shape + batch_shape + event_shape`
        :rtype: numpy.ndarray
        """
        return self.sample(key, sample_shape=sample_shape), []

    def log_prob(self, value):
        """
        Evaluates the log probability density for a batch of samples given by
        `value`.

        :param value: A batch of samples from the distribution.
        :return: an array with shape `value.shape[:-self.event_shape]`
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    @property
    def mean(self):
        """
        Mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        """
        Variance of the distribution.
        """
        raise NotImplementedError

    def _validate_sample(self, value):
        mask = self.support(value)
        if not_jax_tracer(mask):
            if not np.all(mask):
                warnings.warn('Out-of-support values provided to log prob method. '
                              'The value argument should be within the support.')
        return mask

    def __call__(self, *args, **kwargs):
        key = kwargs.pop('rng_key')
        sample_intermediates = kwargs.pop('sample_intermediates', False)
        if sample_intermediates:
            return self.sample_with_intermediates(key, *args, **kwargs)
        return self.sample(key, *args, **kwargs)

    def to_event(self, reinterpreted_batch_ndims=None):
        """
        Interpret the rightmost `reinterpreted_batch_ndims` batch dimensions as
        dependent event dimensions.

        :param reinterpreted_batch_ndims: Number of rightmost batch dims to
            interpret as event dims.
        :return: An instance of `Independent` distribution.
        :rtype: numpyro.distributions.distribution.Independent
        """
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(self.batch_shape)
        elif reinterpreted_batch_ndims == 0:
            return self
        return Independent(self, reinterpreted_batch_ndims)

    def enumerate_support(self, expand=True):
        """
        Returns an array with shape `len(support) x batch_shape`
        containing all values in the support.
        """
        raise NotImplementedError

    def expand(self, batch_shape):
        """
        Returns a new :class:`ExpandedDistribution` instance with batch
        dimensions expanded to `batch_shape`.

        :param tuple batch_shape: batch shape to expand to.
        :return: an instance of `ExpandedDistribution`.
        :rtype: :class:`ExpandedDistribution`
        """
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        return ExpandedDistribution(self, batch_shape)

    def expand_by(self, sample_shape):
        """
        Expands a distribution by adding ``sample_shape`` to the left side of
        its :attr:`~numpyro.distributions.distribution.Distribution.batch_shape`.
        To expand internal dims of ``self.batch_shape`` from 1 to something
        larger, use :meth:`expand` instead.

        :param tuple sample_shape: The size of the iid batch to be drawn
            from the distribution.
        :return: An expanded version of this distribution.
        :rtype: :class:`ExpandedDistribution`
        """
        return self.expand(tuple(sample_shape) + self.batch_shape)

    def mask(self, mask):
        """
        Masks a distribution by a boolean or boolean-valued array that is
        broadcastable to the distributions
        :attr:`Distribution.batch_shape` .

        :param mask: A boolean or boolean valued array (`True` includes
            a site, `False` excludes a site).
        :type mask: bool or jnp.ndarray
        :return: A masked copy of this distribution.
        :rtype: :class:`MaskedDistribution`
        """
        if mask is True:
            return self
        return MaskedDistribution(self, mask)


class ExpandedDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, base_dist, batch_shape=()):
        if isinstance(base_dist, ExpandedDistribution):
            batch_shape, _, _ = self._broadcast_shape(base_dist.batch_shape, batch_shape)
            base_dist = base_dist.base_dist
        self.base_dist = base_dist

        # adjust batch shape
        # Do basic validation. e.g. we should not "unexpand" distributions even if that is possible.
        new_shape, _, _ = self._broadcast_shape(base_dist.batch_shape, batch_shape)
        # Record interstitial and expanded dims/sizes w.r.t. the base distribution
        new_shape, expanded_sizes, interstitial_sizes = self._broadcast_shape(base_dist.batch_shape,
                                                                              new_shape)
        self._expanded_sizes = expanded_sizes
        self._interstitial_sizes = interstitial_sizes
        super().__init__(new_shape, base_dist.event_shape)

    @staticmethod
    def _broadcast_shape(existing_shape, new_shape):
        if len(new_shape) < len(existing_shape):
            raise ValueError("Cannot broadcast distribution of shape {} to shape {}"
                             .format(existing_shape, new_shape))
        reversed_shape = list(reversed(existing_shape))
        expanded_sizes, interstitial_sizes = [], []
        for i, size in enumerate(reversed(new_shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
                expanded_sizes.append((-i - 1, size))
            elif reversed_shape[i] == 1:
                if size != 1:
                    reversed_shape[i] = size
                    interstitial_sizes.append((-i - 1, size))
            elif reversed_shape[i] != size:
                raise ValueError("Cannot broadcast distribution of shape {} to shape {}"
                                 .format(existing_shape, new_shape))
        return tuple(reversed(reversed_shape)), OrderedDict(expanded_sizes), OrderedDict(interstitial_sizes)

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @property
    def is_discrete(self):
        return self.base_dist.is_discrete

    @property
    def support(self):
        return self.base_dist.support

    def sample(self, key, sample_shape=()):
        interstitial_dims = tuple(self._interstitial_sizes.keys())
        event_dim = len(self.event_shape)
        interstitial_dims = tuple(i - event_dim for i in interstitial_dims)
        interstitial_sizes = tuple(self._interstitial_sizes.values())
        expanded_sizes = tuple(self._expanded_sizes.values())
        batch_shape = expanded_sizes + interstitial_sizes
        samples = self.base_dist(rng_key=key, sample_shape=sample_shape + batch_shape)
        interstitial_idx = len(sample_shape) + len(expanded_sizes)
        interstitial_sample_dims = tuple(range(interstitial_idx, interstitial_idx + len(interstitial_sizes)))
        for dim1, dim2 in zip(interstitial_dims, interstitial_sample_dims):
            samples = jnp.swapaxes(samples, dim1, dim2)
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        shape = lax.broadcast_shapes(self.batch_shape,
                                     jnp.shape(value)[:max(jnp.ndim(value) - self.event_dim, 0)])
        log_prob = self.base_dist.log_prob(value)
        return jnp.broadcast_to(log_prob, shape)

    def enumerate_support(self, expand=True):
        samples = self.base_dist.enumerate_support(expand=False)
        enum_shape = samples.shape[:1]
        samples = samples.reshape(enum_shape + (1,) * len(self.batch_shape))
        if expand:
            samples = samples.expand(enum_shape + self.batch_shape)
        return samples

    @property
    def mean(self):
        return jnp.broadcast_to(self.base_dist.mean, self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.base_dist.variance, self.batch_shape + self.event_shape)

    def tree_flatten(self):
        prepend_ndim = len(self.batch_shape) - len(self.base_dist.batch_shape)
        base_dist = tree_util.tree_map(
            lambda x: promote_shapes(x, shape=(1,) * prepend_ndim + jnp.shape(x))[0],
            self.base_dist)
        base_flatten, base_aux = base_dist.tree_flatten()
        return base_flatten, (type(self.base_dist), base_aux, self.batch_shape)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        base_cls, base_aux, batch_shape = aux_data
        base_dist = base_cls.tree_unflatten(base_aux, params)
        prepend_shape = base_dist.batch_shape[:len(base_dist.batch_shape) - len(batch_shape)]
        return cls(base_dist, batch_shape=prepend_shape + batch_shape)


class ImproperUniform(Distribution):
    """
    A helper distribution with zero :meth:`log_prob` over the `support` domain.

    .. note:: `sample` method is not implemented for this distribution. In autoguide and mcmc,
        initial parameters for improper sites are derived from `init_to_uniform` or `init_to_value`
        strategies.

    **Usage:**

    .. doctest::

       >>> from numpyro import sample
       >>> from numpyro.distributions import ImproperUniform, Normal, constraints
       >>>
       >>> def model():
       ...     # ordered vector with length 10
       ...     x = sample('x', ImproperUniform(constraints.ordered_vector, (), event_shape=(10,)))
       ...
       ...     # real matrix with shape (3, 4)
       ...     y = sample('y', ImproperUniform(constraints.real, (), event_shape=(3, 4)))
       ...
       ...     # a shape-(6, 8) batch of length-5 vectors greater than 3
       ...     z = sample('z', ImproperUniform(constraints.greater_than(3), (6, 8), event_shape=(5,)))

    If you want to set improper prior over all values greater than `a`, where `a` is
    another random variable, you might use

       >>> def model():
       ...     a = sample('a', Normal(0, 1))
       ...     x = sample('x', ImproperUniform(constraints.greater_than(a), (), event_shape=()))

    or if you want to reparameterize it

       >>> from numpyro.distributions import TransformedDistribution, transforms
       >>> from numpyro.handlers import reparam
       >>> from numpyro.infer.reparam import TransformReparam
       >>>
       >>> def model():
       ...     a = sample('a', Normal(0, 1))
       ...     with reparam(config={'x': TransformReparam()}):
       ...         x = sample('x',
       ...                    TransformedDistribution(ImproperUniform(constraints.positive, (), ()),
       ...                                            transforms.AffineTransform(a, 1)))

    :param ~numpyro.distributions.constraints.Constraint support: the support of this distribution.
    :param tuple batch_shape: batch shape of this distribution. It is usually safe to
        set `batch_shape=()`.
    :param tuple event_shape: event shape of this distribution.
    """
    arg_constraints = {}

    def __init__(self, support, batch_shape, event_shape, validate_args=None):
        self.support = support
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        batch_shape = jnp.shape(value)[:jnp.ndim(value) - len(self.event_shape)]
        batch_shape = lax.broadcast_shapes(batch_shape, self.batch_shape)
        return jnp.zeros(batch_shape)

    def _validate_sample(self, value):
        mask = super(ImproperUniform, self)._validate_sample(value)
        batch_dim = jnp.ndim(value) - len(self.event_shape)
        if batch_dim < jnp.ndim(mask):
            mask = jnp.all(jnp.reshape(mask, jnp.shape(mask)[:batch_dim] + (-1,)), -1)
        return mask

    def tree_flatten(self):
        raise NotImplementedError(
            "Cannot flattening ImproperPrior distribution for general supports. "
            "Please raising a feature request for your specific `support`. "
            "Alternatively, you can use '.mask(False)' pattern. "
            "For example, to define an improper prior over positive domain, "
            "we can use the distribution `dist.LogNormal(0, 1).mask(False)`.")


class Independent(Distribution):
    """
    Reinterprets batch dimensions of a distribution as event dims by shifting
    the batch-event dim boundary further to the left.

    From a practical standpoint, this is useful when changing the result of
    :meth:`log_prob`. For example, a univariate Normal distribution can be
    interpreted as a multivariate Normal with diagonal covariance:

    .. doctest::

        >>> import numpyro.distributions as dist
        >>> normal = dist.Normal(jnp.zeros(3), jnp.ones(3))
        >>> [normal.batch_shape, normal.event_shape]
        [(3,), ()]
        >>> diag_normal = dist.Independent(normal, 1)
        >>> [diag_normal.batch_shape, diag_normal.event_shape]
        [(), (3,)]

    :param numpyro.distribution.Distribution base_distribution: a distribution instance.
    :param int reinterpreted_batch_ndims: the number of batch dims to reinterpret as event dims.
    """
    arg_constraints = {}

    def __init__(self, base_dist, reinterpreted_batch_ndims, validate_args=None):
        if reinterpreted_batch_ndims > len(base_dist.batch_shape):
            raise ValueError("Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                             "actual {} vs {}".format(reinterpreted_batch_ndims,
                                                      len(base_dist.batch_shape)))
        shape = base_dist.batch_shape + base_dist.event_shape
        event_dim = reinterpreted_batch_ndims + len(base_dist.event_shape)
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        self.base_dist = base_dist
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super(Independent, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def support(self):
        return self.base_dist.support

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @property
    def is_discrete(self):
        return self.base_dist.is_discrete

    @property
    def reparameterized_params(self):
        return self.base_dist.reparameterized_params

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, key, sample_shape=()):
        return self.base_dist(rng_key=key, sample_shape=sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def expand(self, batch_shape):
        base_batch_shape = batch_shape + self.event_shape[:self.reinterpreted_batch_ndims]
        return self.base_dist.expand(base_batch_shape).to_event(self.reinterpreted_batch_ndims)

    def tree_flatten(self):
        base_flatten, base_aux = self.base_dist.tree_flatten()
        return base_flatten, (type(self.base_dist), base_aux, self.reinterpreted_batch_ndims)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        base_cls, base_aux, reinterpreted_batch_ndims = aux_data
        base_dist = base_cls.tree_unflatten(base_aux, params)
        return cls(base_dist, reinterpreted_batch_ndims)


class MaskedDistribution(Distribution):
    """
    Masks a distribution by a boolean array that is broadcastable to the
    distribution's :attr:`Distribution.batch_shape`.
    In the special case ``mask is False``, computation of :meth:`log_prob` , is skipped,
    and constant zero values are returned instead.

    :param mask: A boolean or boolean-valued array.
    :type mask: jnp.ndarray or bool
    """
    arg_constraints = {}

    def __init__(self, base_dist, mask):
        if isinstance(mask, bool):
            self._mask = mask
        else:
            batch_shape = lax.broadcast_shapes(jnp.shape(mask), tuple(base_dist.batch_shape))
            if mask.shape != batch_shape:
                mask = jnp.broadcast_to(mask, batch_shape)
            if base_dist.batch_shape != batch_shape:
                base_dist = base_dist.expand(batch_shape)
            self._mask = mask.astype('bool')
        self.base_dist = base_dist
        super().__init__(base_dist.batch_shape, base_dist.event_shape)

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @property
    def is_discrete(self):
        return self.base_dist.is_discrete

    @property
    def support(self):
        return self.base_dist.support

    def sample(self, key, sample_shape=()):
        return self.base_dist(rng_key=key, sample_shape=sample_shape)

    def log_prob(self, value):
        if self._mask is False:
            shape = lax.broadcast_shapes(tuple(self.base_dist.batch_shape),
                                         jnp.shape(value)[:max(jnp.ndim(value) - len(self.event_shape), 0)])
            return jnp.zeros(shape)
        if self._mask is True:
            return self.base_dist.log_prob(value)
        return jnp.where(self._mask, self.base_dist.log_prob(value), 0.)

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def tree_flatten(self):
        base_flatten, base_aux = self.base_dist.tree_flatten()
        if isinstance(self._mask, bool):
            return base_flatten, (type(self.base_dist), base_aux, self._mask)
        else:
            return (base_flatten, self._mask), (type(self.base_dist), base_aux)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        if len(aux_data) == 2:
            base_flatten, mask = params
            base_cls, base_aux = aux_data
        else:
            base_flatten = params
            base_cls, base_aux, mask = aux_data
        base_dist = base_cls.tree_unflatten(base_aux, base_flatten)
        return cls(base_dist, mask)


class TransformedDistribution(Distribution):
    """
    Returns a distribution instance obtained as a result of applying
    a sequence of transforms to a base distribution. For an example,
    see :class:`~numpyro.distributions.LogNormal` and
    :class:`~numpyro.distributions.HalfNormal`.

    :param base_distribution: the base distribution over which to apply transforms.
    :param transforms: a single transform or a list of transforms.
    :param validate_args: Whether to enable validation of distribution
        parameters and arguments to `.log_prob` method.
    """
    arg_constraints = {}

    def __init__(self, base_distribution, transforms, validate_args=None):
        if isinstance(transforms, Transform):
            transforms = [transforms, ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
        else:
            raise ValueError("transforms must be a Transform or list, but was {}".format(transforms))
        # XXX: this logic will not be valid when IndependentDistribution is support;
        # in that case, it is more involved to support Transform(Indep(Transform));
        # however, we might not need to support such kind of distribution
        # and should raise an error if base_distribution is an Indep one
        if isinstance(base_distribution, TransformedDistribution):
            self.base_dist = base_distribution.base_dist
            self.transforms = base_distribution.transforms + transforms
        else:
            self.base_dist = base_distribution
            self.transforms = transforms
        # NB: here we assume that base_dist.shape == transformed_dist.shape
        # but that might not be True for some transforms such as StickBreakingTransform
        # because the event dimension is transformed from (n - 1,) to (n,).
        # Currently, we have no mechanism to fix this issue. Given that
        # this is just an edge case, we might skip this issue but need
        # to pay attention to any inference function that inspects
        # transformed distribution's shape.
        shape = base_distribution.batch_shape + base_distribution.event_shape
        base_ndim = len(shape)
        transform = ComposeTransform(self.transforms)
        transform_input_event_dim = transform.input_event_dim
        if base_ndim < transform_input_event_dim:
            raise ValueError("Base distribution needs to have shape with size at least {}, but got {}."
                             .format(transform_input_event_dim, base_ndim))
        event_dim = transform.output_event_dim + max(self.base_dist.event_dim - transform_input_event_dim, 0)
        # See the above note. Currently, there is no way to interpret the shape of output after
        # transforming. To solve this issue, we need something like Bijector.forward_event_shape
        # as in TFP. For now, we will prepend singleton dimensions to compromise, so that
        # event_dim, len(batch_shape) are still correct.
        if event_dim <= base_ndim:
            batch_shape = shape[:base_ndim - event_dim]
            event_shape = shape[base_ndim - event_dim:]
        else:
            event_shape = (-1,) * event_dim
            batch_shape = ()
        super(TransformedDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def support(self):
        domain = self.base_dist.support
        for t in self.transforms:
            t.domain = domain
            domain = t.codomain
        return domain

    def sample(self, key, sample_shape=()):
        x = self.base_dist(rng_key=key, sample_shape=sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def sample_with_intermediates(self, key, sample_shape=()):
        x = self.base_dist(rng_key=key, sample_shape=sample_shape)
        intermediates = []
        for transform in self.transforms:
            x_tmp = x
            x, t_inter = transform.call_with_intermediates(x)
            intermediates.append([x_tmp, t_inter])
        return x, intermediates

    @validate_sample
    def log_prob(self, value, intermediates=None):
        if intermediates is not None:
            if len(intermediates) != len(self.transforms):
                raise ValueError('Intermediates array has length = {}. Expected = {}.'
                                 .format(len(intermediates), len(self.transforms)))
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for i, transform in enumerate(reversed(self.transforms)):
            x = transform.inv(y) if intermediates is None else intermediates[-i - 1][0]
            t_inter = None if intermediates is None else intermediates[-i - 1][1]
            t_log_det = transform.log_abs_det_jacobian(x, y, t_inter)
            batch_ndim = event_dim - transform.output_event_dim
            log_prob = log_prob - sum_rightmost(t_log_det, batch_ndim)
            event_dim = transform.input_event_dim + batch_ndim
            y = x

        log_prob = log_prob + sum_rightmost(self.base_dist.log_prob(y),
                                            event_dim - len(self.base_dist.event_shape))
        return log_prob

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def tree_flatten(self):
        raise NotImplementedError(
            "Flatenning TransformedDistribution is only supported for some specific cases."
            " Consider using `TransformReparam` to convert this distribution to the base_dist,"
            " which is supported in most situtations. In addition, please reach out to us with"
            " your usage cases.")


class Unit(Distribution):
    """
    Trivial nonnormalized distribution representing the unit type.

    The unit type has a single value with no data, i.e. ``value.size == 0``.

    This is used for :func:`numpyro.factor` statements.
    """
    arg_constraints = {'log_factor': real}
    support = real

    def __init__(self, log_factor, validate_args=None):
        batch_shape = jnp.shape(log_factor)
        event_shape = (0,)  # This satisfies .size == 0.
        self.log_factor = log_factor
        super(Unit, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return jnp.empty(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        shape = lax.broadcast_shapes(self.batch_shape, jnp.shape(value)[:-1])
        return jnp.broadcast_to(self.log_factor, shape)
