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

import jax.numpy as np

from numpyro.distributions.constraints import Transform, is_dependent
from numpyro.distributions.util import sum_rightmost


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

    .. testsetup::

       import jax.numpy as np
       import numpyro.distributions as dist

    .. doctest::

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

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            for param, constraint in self.arg_constraints.items():
                if is_dependent(constraint):
                    continue  # skip constraints that cannot be checked
                if not np.all(constraint(getattr(self, param))):
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

        :param jax.random.PRNGKey key: the rng key to be used for the distribution.
        :param size: the sample shape for the distribution.
        :return: a `numpy.ndarray` of shape `sample_shape + batch_shape + event_shape`
        """
        raise NotImplementedError

    def log_prob(self, value):
        """
        Evaluates the log probability density for a batch of samples given by
        `value`.

        :param value: A batch of samples from the distribution.
        :return: a `numpy.ndarray` with shape `value.shape[:-self.event_shape]`
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
        if not np.all(self.support(value)):
            raise ValueError('Invalid values provided to log prob method. '
                             'The value argument must be within the support.')

    def __call__(self, *args, **kwargs):
        key = kwargs.pop('random_state')
        return self.sample(key, *args, **kwargs)


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
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms, ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError("transforms must be a Transform or list, but was {}".format(transforms))
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max([len(self.base_dist.event_shape)] + [t.event_dim for t in self.transforms])
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

    @property
    def is_reparametrized(self):
        return self.base_dist.reparametrized

    def sample(self, key, sample_shape=()):
        x = self.base_dist.sample(key, sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_prob = log_prob - sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                                event_dim - transform.event_dim)
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
