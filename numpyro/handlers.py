from __future__ import absolute_import, division, print_function

from collections import OrderedDict

_PYRO_STACK = []
_PARAM_STORE = {}


class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        _PYRO_STACK.append(self)

    def __exit__(self):
        assert _PYRO_STACK[-1] is self
        _PYRO_STACK.pop()

    def process_message(self, msg):
        raise NotImplementedError

    def postprocess_message(self, msg):
        raise NotADirectoryError

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class trace(Messenger):
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        assert msg["name"] not in self.trace, "all sites must have unique names"
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


class block(Messenger):
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True


def apply_stack(msg):
    pointer = 0
    for pointer, handler in enumerate(reversed(_PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in _PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


def sample(fn, obs=None, name=None):

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn.sample('value')

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name if name is not None else "sample",
        "fn": fn,
        "args": (),
        "value": obs,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


def param(init_value=None, name=None):

    if init_value is None and name is None:
        raise ValueError("empty args to param")

    def fn(init_value):
        value = _PARAM_STORE.setdefault(name, init_value)
        value.requires_grad_()
        return value

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn(init_value)

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value,),
        "value": None,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]
