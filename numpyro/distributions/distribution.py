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

from contextlib import contextmanager
import warnings

import jax.numpy as np
from jax import lax

from numpyro.distributions.constraints import is_dependent, real
from numpyro.distributions.transforms import Transform
from numpyro.distributions.util import lazy_property, sum_rightmost, validate_sample
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


class Distribution(object):
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

       >>> import jax.numpy as np
       >>> import numpyro.distributions as dist
       >>> d = dist.Dirichlet(np.ones((2, 3, 4)))
       >>> d.batch_shape
       (2, 3)
       >>> d.event_shape
       (4,)
    """
    arg_constraints = {}
    support = None
    reparametrized_params = []
    _validate_args = False

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
                is_valid = np.all(constraint(getattr(self, param)))
                if not_jax_tracer(is_valid):
                    if not is_valid:
                        raise ValueError("The parameter {} has invalid values".format(param))
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

    def transform_with_intermediates(self, base_value):
        return base_value, []

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
        :rtype: Independent
        """
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(self.batch_shape)
        return Independent(self, reinterpreted_batch_ndims)


class Independent(Distribution):
    """
    Reinterprets batch dimensions of a distribution as event dims by shifting
    the batch-event dim boundary further to the left.

    From a practical standpoint, this is useful when changing the result of
    :meth:`log_prob`. For example, a univariate Normal distribution can be
    interpreted as a multivariate Normal with diagonal covariance:

    .. doctest::

        >>> import numpyro.distributions as dist
        >>> normal = dist.Normal(np.zeros(3), np.ones(3))
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
    def reparameterized_params(self):
        return self.base_dist.reparameterized_params

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, key, sample_shape=()):
        return self.base_dist.sample(key, sample_shape=sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return sum_rightmost(log_prob, self.reinterpreted_batch_ndims)


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
        shape = base_distribution.batch_shape + base_distribution.event_shape
        event_dim = max([len(base_distribution.event_shape)] + [t.event_dim for t in transforms])
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        super(TransformedDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def support(self):
        domain = self.base_dist.support
        for t in self.transforms:
            t.domain = domain
            domain = t.codomain
        return domain

    def sample(self, key, sample_shape=()):
        x = self.base_dist.sample(key, sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def sample_with_intermediates(self, key, sample_shape=()):
        base_value = self.base_dist.sample(key, sample_shape)
        return self.transform_with_intermediates(base_value)

    def transform_with_intermediates(self, base_value):
        x = base_value
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
            log_prob = log_prob - sum_rightmost(t_log_det, event_dim - transform.event_dim)
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


class Unit(Distribution):
    """
    Trivial nonnormalized distribution representing the unit type.

    The unit type has a single value with no data, i.e. ``value.size == 0``.

    This is used for :func:`numpyro.factor` statements.
    """
    arg_constraints = {'log_factor': real}
    support = real

    def __init__(self, log_factor, validate_args=None):
        batch_shape = np.shape(log_factor)
        event_shape = (0,)  # This satisfies .size == 0.
        self.log_factor = log_factor
        super(Unit, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return np.empty(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        shape = lax.broadcast_shapes(self.batch_shape, np.shape(value)[:-1])
        return np.broadcast_to(self.log_factor, shape)
