import jax

import numpyro
from numpyro.distributions.discrete import PRNGIdentity

_PYRO_STACK = []


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


def sample(name, fn, obs=None, sample_shape=()):
    """
    Returns a random sample from the stochastic function `fn`. This can have
    additional side effects when wrapped inside effect handlers like
    :class:`~numpyro.handlers.substitute`.

    :param str name: name of the sample site
    :param fn: Python callable
    :param numpy.ndarray obs: observed value
    :param sample_shape: Shape of samples to be drawn.
    :return: sample from the stochastic `fn`.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn(sample_shape=sample_shape)

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'sample',
        'name': name,
        'fn': fn,
        'args': (),
        'kwargs': {'sample_shape': sample_shape},
        'value': obs,
        'is_observed': obs is not None,
        'intermediates': [],
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
