from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from jax import random

_PYRO_STACK = []


class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

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


class trace(Messenger):
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        assert msg['name'] not in self.trace, 'all sites must have unique names'
        self.trace[msg['name']] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg['name'] in self.guide_trace:
            msg['value'] = self.guide_trace[msg['name']]['value']


class block(Messenger):
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg['stop'] = True


class seed(Messenger):
    def __init__(self, fn, rng):
        self.rng = rng
        super(seed, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] == 'sample':
            msg['kwargs']['random_state'] = self.rng
            self.rng, = random.split(self.rng, 1)


class substitute(Messenger):
    def __init__(self, fn=None, param_map=None):
        self.param_map = param_map
        super(substitute, self).__init__(fn)

    def process_message(self, msg):
        if msg['name'] in self.param_map:
            msg['value'] = self.param_map[msg['name']]


def apply_stack(msg):
    pointer = 0
    for pointer, handler in enumerate(reversed(_PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg['value'] is None:
        msg['value'] = msg['fn'].rvs(*msg['args'], **msg['kwargs'])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in _PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


def sample(name, fn, obs=None):
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn.rvs()

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'sample',
        'name': name,
        'fn': fn,
        'args': (),
        'kwargs': {},
        'value': obs,
        'is_observed': obs is not None,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']


def identity(x):
    return x


def param(name, init_value):
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return init_value

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'param',
        'name': name,
        'fn': identity,
        'args': (init_value,),
        'kwargs': {},
        'value': None,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']
