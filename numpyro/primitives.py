# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from contextlib import ExitStack, contextmanager
import functools

from jax import lax, random
import jax.numpy as jnp

import numpyro
from numpyro.distributions.discrete import PRNGIdentity
from numpyro.util import identity

_PYRO_STACK = []


CondIndepStackFrame = namedtuple('CondIndepStackFrame', ['name', 'dim', 'size'])


def apply_stack(msg):
    pointer = 0
    for pointer, handler in enumerate(reversed(_PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg['value'] is None:
        if msg['type'] == 'sample':
            msg['value'], msg['intermediates'] = msg['fn'](*msg['args'],
                                                           sample_intermediates=True,
                                                           **msg['kwargs'])
        else:
            msg['value'] = msg['fn'](*msg['args'], **msg['kwargs'])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in _PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


class Messenger(object):
    def __init__(self, fn=None):
        if fn is not None and not callable(fn):
            raise ValueError("Expected `fn` to be a Python callable object; "
                             "instead found type(fn) = {}.".format(type(fn)))
        self.fn = fn
        functools.update_wrapper(self, fn, updated=[])

    def __enter__(self):
        _PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert _PYRO_STACK[-1] is self
        _PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


def sample(name, fn, obs=None, rng_key=None, sample_shape=(), infer=None):
    """
    Returns a random sample from the stochastic function `fn`. This can have
    additional side effects when wrapped inside effect handlers like
    :class:`~numpyro.handlers.substitute`.

    .. note::
        By design, `sample` primitive is meant to be used inside a NumPyro model.
        Then :class:`~numpyro.handlers.seed` handler is used to inject a random
        state to `fn`. In those situations, `rng_key` keyword will take no
        effect.

    :param str name: name of the sample site.
    :param fn: a stochastic function that returns a sample.
    :param numpy.ndarray obs: observed value
    :param jax.random.PRNGKey rng_key: an optional random key for `fn`.
    :param sample_shape: Shape of samples to be drawn.
    :param dict infer: an optional dictionary containing additional information
        for inference algorithms. For example, if `fn` is a discrete distribution,
        setting `infer={'enumerate': 'parallel'}` to tell MCMC marginalize
        this discrete latent site.
    :return: sample from the stochastic `fn`.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn(rng_key=rng_key, sample_shape=sample_shape)

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'sample',
        'name': name,
        'fn': fn,
        'args': (),
        'kwargs': {'rng_key': rng_key, 'sample_shape': sample_shape},
        'value': obs,
        'scale': None,
        'is_observed': obs is not None,
        'intermediates': [],
        'cond_indep_stack': [],
        'infer': {} if infer is None else infer,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']


def param(name, init_value=None, **kwargs):
    """
    Annotate the given site as an optimizable parameter for use with
    :mod:`jax.experimental.optimizers`. For an example of how `param` statements
    can be used in inference algorithms, refer to :func:`~numpyro.svi.svi`.

    :param str name: name of site.
    :param numpy.ndarray init_value: initial value specified by the user. Note that
        the onus of using this to initialize the optimizer is on the user /
        inference algorithm, since there is no global parameter store in
        NumPyro.
    :param constraint: NumPyro constraint, defaults to ``constraints.real``.
    :type constraint: numpyro.distributions.constraints.Constraint
    :param int event_dim: (optional) number of rightmost dimensions unrelated
        to batching. Dimension to the left of this will be considered batch
        dimensions; if the param statement is inside a subsampled plate, then
        corresponding batch dimensions of the parameter will be correspondingly
        subsampled. If unspecified, all dimensions will be considered event
        dims and no subsampling will be performed.
    :return: value for the parameter. Unless wrapped inside a
        handler like :class:`~numpyro.handlers.substitute`, this will simply
        return the initial value.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return init_value

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'param',
        'name': name,
        'fn': identity,
        'args': (init_value,),
        'kwargs': kwargs,
        'value': None,
        'scale': None,
        'cond_indep_stack': [],
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']


def deterministic(name, value):
    """
    Used to designate deterministic sites in the model. Note that most effect
    handlers will not operate on deterministic sites (except
    :func:`~numpyro.handlers.trace`), so deterministic sites should be
    side-effect free. The use case for deterministic nodes is to record any
    values in the model execution trace.

    :param str name: name of the deterministic site.
    :param numpy.ndarray value: deterministic value to record in the trace.
    """
    if not _PYRO_STACK:
        return value

    initial_msg = {
        'type': 'deterministic',
        'name': name,
        'value': value,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']


def module(name, nn, input_shape=None):
    """
    Declare a :mod:`~jax.experimental.stax` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    :param str name: name of the module to be registered.
    :param tuple nn: a tuple of `(init_fn, apply_fn)` obtained by a :mod:`~jax.experimental.stax`
        constructor function.
    :param tuple input_shape: shape of the input taken by the
        neural network.
    :return: a `apply_fn` with bound parameters that takes an array
        as an input and returns the neural network transformed output
        array.
    """
    module_key = name + '$params'
    nn_init, nn_apply = nn
    nn_params = param(module_key)
    if nn_params is None:
        if input_shape is None:
            raise ValueError('Valid value for `input_shape` needed to initialize.')
        rng_key = numpyro.sample(name + '$rng_key', PRNGIdentity())
        _, nn_params = nn_init(rng_key, input_shape)
        param(module_key, nn_params)
    return functools.partial(nn_apply, nn_params)


def _subsample_fn(size, subsample_size, rng_key=None):
    assert rng_key is not None, "Missing random key to generate subsample indices."
    return random.permutation(rng_key, size)[:subsample_size]


class plate(Messenger):
    """
    Construct for annotating conditionally independent variables. Within a
    `plate` context manager, `sample` sites will be automatically broadcasted to
    the size of the plate. Additionally, a scale factor might be applied by
    certain inference algorithms if `subsample_size` is specified.

    .. note:: This can be used to subsample minibatches of data:

        .. code-block:: python

            with plate("data", len(data), subsample_size=100) as ind:
                batch = data[ind]
                assert len(batch) == 100

    :param str name: Name of the plate.
    :param int size: Size of the plate.
    :param int subsample_size: Optional argument denoting the size of the mini-batch.
        This can be used to apply a scaling factor by inference algorithms. e.g.
        when computing ELBO using a mini-batch.
    :param int dim: Optional argument to specify which dimension in the tensor
        is used as the plate dim. If `None` (default), the leftmost available dim
        is allocated.
    """
    def __init__(self, name, size, subsample_size=None, dim=None):
        self.name = name
        self.size = size
        if dim is not None and dim >= 0:
            raise ValueError('dim arg must be negative.')
        self.dim, self._indices = self._subsample(
            self.name, self.size, subsample_size, dim)
        self.subsample_size = self._indices.shape[0]
        super(plate, self).__init__()

    # XXX: different from Pyro, this method returns dim and indices
    @staticmethod
    def _subsample(name, size, subsample_size, dim):
        msg = {
            'type': 'plate',
            'fn': _subsample_fn,
            'name': name,
            'args': (size, subsample_size),
            'kwargs': {'rng_key': None},
            'value': (None
                      if (subsample_size is not None and size != subsample_size)
                      else jnp.arange(size)),
            'scale': 1.0,
            'cond_indep_stack': [],
        }
        apply_stack(msg)
        subsample = msg['value']
        if subsample_size is not None and subsample_size != subsample.shape[0]:
            raise ValueError("subsample_size does not match len(subsample), {} vs {}.".format(
                subsample_size, len(subsample)) +
                " Did you accidentally use different subsample_size in the model and guide?")
        cond_indep_stack = msg['cond_indep_stack']
        occupied_dims = {f.dim for f in cond_indep_stack}
        if dim is None:
            new_dim = -1
            while new_dim in occupied_dims:
                new_dim -= 1
            dim = new_dim
        else:
            assert dim not in occupied_dims
        return dim, subsample

    def __enter__(self):
        super().__enter__()
        return self._indices

    @staticmethod
    def _get_batch_shape(cond_indep_stack):
        n_dims = max(-f.dim for f in cond_indep_stack)
        batch_shape = [1] * n_dims
        for f in cond_indep_stack:
            batch_shape[f.dim] = f.size
        return tuple(batch_shape)

    def process_message(self, msg):
        if msg['type'] not in ('param', 'sample', 'plate'):
            if msg['type'] == 'control_flow':
                raise NotImplementedError('Cannot use control flow primitive under a `plate` primitive.'
                                          ' Please move those `plate` statements into the control flow'
                                          ' body function. See `scan` documentation for more information.')
            return

        cond_indep_stack = msg['cond_indep_stack']
        frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
        cond_indep_stack.append(frame)
        if msg['type'] == 'sample':
            expected_shape = self._get_batch_shape(cond_indep_stack)
            dist_batch_shape = msg['fn'].batch_shape
            if 'sample_shape' in msg['kwargs']:
                dist_batch_shape = msg['kwargs']['sample_shape'] + dist_batch_shape
                msg['kwargs']['sample_shape'] = ()
            overlap_idx = max(len(expected_shape) - len(dist_batch_shape), 0)
            trailing_shape = expected_shape[overlap_idx:]
            broadcast_shape = lax.broadcast_shapes(trailing_shape, tuple(dist_batch_shape))
            batch_shape = expected_shape[:overlap_idx] + broadcast_shape
            msg['fn'] = msg['fn'].expand(batch_shape)
        if self.size != self.subsample_size:
            scale = 1. if msg['scale'] is None else msg['scale']
            msg['scale'] = scale * self.size / self.subsample_size

    def postprocess_message(self, msg):
        if msg["type"] in ("subsample", "param") and self.dim is not None:
            event_dim = msg["kwargs"].get("event_dim")
            if event_dim is not None:
                assert event_dim >= 0
                dim = self.dim - event_dim
                shape = jnp.shape(msg["value"])
                if len(shape) >= -dim and shape[dim] != 1:
                    if shape[dim] != self.size:
                        if msg["type"] == "param":
                            statement = "numpyro.param({}, ..., event_dim={})".format(msg["name"], event_dim)
                        else:
                            statement = "numpyro.subsample(..., event_dim={})".format(event_dim)
                        raise ValueError(
                            "Inside numpyro.plate({}, {}, dim={}) invalid shape of {}: {}"
                            .format(self.name, self.size, self.dim, statement, shape))
                    if self.subsample_size < self.size:
                        value = msg["value"]
                        new_value = jnp.take(value, self._indices, dim)
                        msg["value"] = new_value


@contextmanager
def plate_stack(prefix, sizes, rightmost_dim=-1):
    """
    Create a contiguous stack of :class:`plate` s with dimensions::

        rightmost_dim - len(sizes), ..., rightmost_dim

    :param str prefix: Name prefix for plates.
    :param iterable sizes: An iterable of plate sizes.
    :param int rightmost_dim: The rightmost dim, counting from the right.
    """
    assert rightmost_dim < 0
    with ExitStack() as stack:
        for i, size in enumerate(reversed(sizes)):
            plate_i = plate("{}_{}".format(prefix, i), size, dim=rightmost_dim - i)
            stack.enter_context(plate_i)
        yield


def factor(name, log_factor):
    """
    Factor statement to add arbitrary log probability factor to a
    probabilistic model.

    :param str name: Name of the trivial sample.
    :param numpy.ndarray log_factor: A possibly batched log probability factor.
    """
    unit_dist = numpyro.distributions.distribution.Unit(log_factor)
    unit_value = unit_dist.sample(None)
    sample(name, unit_dist, obs=unit_value)


def subsample(data, event_dim):
    """
    EXPERIMENTAL Subsampling statement to subsample data based on enclosing
    :class:`~numpyro.primitives.plate` s.

    This is typically called on arguments to ``model()`` when subsampling is
    performed automatically by :class:`~numpyro.primitives.plate` s by passing
    ``subsample_size`` kwarg. For example the following are equivalent::

        # Version 1. using indexing
        def model(data):
            with numpyro.plate("data", len(data), subsample_size=10, dim=-data.dim()) as ind:
                data = data[ind]
                # ...

        # Version 2. using numpyro.subsample()
        def model(data):
            with numpyro.plate("data", len(data), subsample_size=10, dim=-data.dim()):
                data = numpyro.subsample(data, event_dim=0)
                # ...

    :param numpy.ndarray data: A tensor of batched data.
    :param int event_dim: The event dimension of the data tensor. Dimensions to
        the left are considered batch dimensions.
    :returns: A subsampled version of ``data``
    :rtype: ~numpy.ndarray
    """
    if not _PYRO_STACK:
        return data

    assert isinstance(event_dim, int) and event_dim >= 0
    initial_msg = {
        'type': 'subsample',
        'value': data,
        'kwargs': {'event_dim': event_dim}
    }

    msg = apply_stack(initial_msg)
    return msg['value']
