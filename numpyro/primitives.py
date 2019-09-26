from collections import namedtuple
import functools
from operator import sub

import jax
from jax import lax

import numpyro
from numpyro.distributions.discrete import PRNGIdentity

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


def sample(name, fn, obs=None, random_state=None, sample_shape=()):
    """
    Returns a random sample from the stochastic function `fn`. This can have
    additional side effects when wrapped inside effect handlers like
    :class:`~numpyro.handlers.substitute`.

    .. note::
        By design, `sample` primitive is meant to be used inside a NumPyro model.
        Then :class:`~numpyro.handlers.seed` handler is used to inject a random
        state to `fn`. In those situations, `random_state` keyword will take no
        effect.

    :param str name: name of the sample site
    :param fn: Python callable
    :param numpy.ndarray obs: observed value
    :param jax.random.PRNGKey random_state: an optional random key for `fn`.
    :param sample_shape: Shape of samples to be drawn.
    :return: sample from the stochastic `fn`.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn(random_state=random_state, sample_shape=sample_shape)

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'sample',
        'name': name,
        'fn': fn,
        'args': (),
        'kwargs': {'random_state': random_state, 'sample_shape': sample_shape},
        'value': obs,
        'scale': 1.0,
        'is_observed': obs is not None,
        'intermediates': [],
        'cond_indep_stack': [],
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']


def identity(x, *args, **kwargs):
    return x


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
        'scale': 1.0,
        'cond_indep_stack': [],
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
            raise ValueError('Valid value for `input_size` needed to initialize.')
        rng = numpyro.sample(name + '$rng', PRNGIdentity())
        _, nn_params = nn_init(rng, input_shape)
        param(module_key, nn_params)
    return jax.partial(nn_apply, nn_params)


class plate(Messenger):
    """
    Construct for annotating conditionally independent variables. Within a
    `plate` context manager, `sample` sites will be automatically broadcasted to
    the size of the plate. Additionally, a scale factor might be applied by
    certain inference algorithms if `subsample_size` is specified.

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
        self.subsample_size = size if subsample_size is None else subsample_size
        if dim is not None and dim >= 0:
            raise ValueError('dim arg must be negative.')
        self.dim = dim
        self._validate_and_set_dim()
        super(plate, self).__init__()

    def _validate_and_set_dim(self):
        msg = {
            'type': 'plate',
            'is_observed': False,
            'fn': identity,
            'name': self.name,
            'args': (None,),
            'kwargs': {},
            'value': None,
            'scale': 1.0,
            'cond_indep_stack': [],
        }
        apply_stack(msg)
        cond_indep_stack = msg['cond_indep_stack']
        occupied_dims = {f.dim for f in cond_indep_stack}
        dim = -1
        while True:
            if dim not in occupied_dims:
                break
            dim -= 1
        if self.dim is None:
            self.dim = dim
        else:
            assert self.dim not in occupied_dims

    @staticmethod
    def _get_batch_shape(cond_indep_stack):
        n_dims = max(-f.dim for f in cond_indep_stack)
        batch_shape = [1] * n_dims
        for f in cond_indep_stack:
            batch_shape[f.dim] = f.size
        return tuple(batch_shape)

    def process_message(self, msg):
        cond_indep_stack = msg['cond_indep_stack']
        frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
        cond_indep_stack.append(frame)
        expected_shape = self._get_batch_shape(cond_indep_stack)
        dist_batch_shape = msg['fn'].batch_shape if msg['type'] == 'sample' else ()
        overlap_idx = len(expected_shape) - len(dist_batch_shape)
        if overlap_idx < 0:
            raise ValueError('Expected dimensions within plate = {}, which is less than the '
                             'distribution\'s batch shape = {}.'.format(len(expected_shape), len(dist_batch_shape)))
        trailing_shape = expected_shape[overlap_idx:]
        # e.g. distribution with batch shape (1, 5) cannot be broadcast to (5, 5)
        broadcast_shape = lax.broadcast_shapes(trailing_shape, dist_batch_shape)
        if sum(map(sub, broadcast_shape, dist_batch_shape)) > 0:
            raise ValueError('Distribution batch shape = {} cannot be broadcast up to {}. '
                             'Consider using unbatched distributions.'
                             .format(dist_batch_shape, broadcast_shape))
        batch_shape = expected_shape[:overlap_idx]
        if 'sample_shape' in msg['kwargs']:
            batch_shape = lax.broadcast_shapes(msg['kwargs']['sample_shape'], batch_shape)
        msg['kwargs']['sample_shape'] = batch_shape
        msg['scale'] = msg['scale'] * self.size / self.subsample_size
