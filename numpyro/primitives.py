# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from contextlib import ExitStack, contextmanager
import functools
import warnings

import jax
from jax import lax, ops, random
import jax.numpy as jnp

import numpyro
from numpyro.util import identity

_PYRO_STACK = []

CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "dim", "size"])


def apply_stack(msg):
    pointer = 0
    for pointer, handler in enumerate(reversed(_PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg["value"] is None:
        if msg["type"] == "sample":
            msg["value"], msg["intermediates"] = msg["fn"](
                *msg["args"], sample_intermediates=True, **msg["kwargs"]
            )
        else:
            msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in _PYRO_STACK[-pointer - 1 :]:
        handler.postprocess_message(msg)
    return msg


class Messenger(object):
    def __init__(self, fn=None):
        if fn is not None and not callable(fn):
            raise ValueError(
                "Expected `fn` to be a Python callable object; "
                "instead found type(fn) = {}.".format(type(fn))
            )
        self.fn = fn
        functools.update_wrapper(self, fn, updated=[])

    def __enter__(self):
        _PYRO_STACK.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert _PYRO_STACK[-1] is self
            _PYRO_STACK.pop()
        else:
            # NB: this mimics Pyro exception handling
            # the wrapped function or block raised an exception
            # handler exception handling:
            # when the callee or enclosed block raises an exception,
            # find this handler's position in the stack,
            # then remove it and everything below it in the stack.
            if self in _PYRO_STACK:
                loc = _PYRO_STACK.index(self)
                for i in range(loc, len(_PYRO_STACK)):
                    _PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        if self.fn is None:
            # Assume self is being used as a decorator.
            assert len(args) == 1 and not kwargs
            self.fn = args[0]
            return self
        with self:
            return self.fn(*args, **kwargs)


def _masked_observe(name, fn, obs, obs_mask, **kwargs):
    # Split into two auxiliary sample sites.
    with numpyro.handlers.mask(mask=obs_mask):
        observed = sample(f"{name}_observed", fn, **kwargs, obs=obs)
    with numpyro.handlers.mask(mask=(obs_mask ^ True)):
        unobserved = sample(f"{name}_unobserved", fn, **kwargs)

    # Interleave observed and unobserved events.
    shape = jnp.shape(obs_mask) + (1,) * fn.event_dim
    batch_mask = jnp.reshape(obs_mask, shape)
    value = jnp.where(batch_mask, observed, unobserved)
    return deterministic(name, value)


def sample(
    name, fn, obs=None, rng_key=None, sample_shape=(), infer=None, obs_mask=None
):
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
    :param numpy.ndarray obs_mask: Optional boolean array mask of shape
        broadcastable with ``fn.batch_shape``. If provided, events with
        mask=True will be conditioned on ``obs`` and remaining events will be
        imputed by sampling. This introduces a latent sample site named ``name
        + "_unobserved"`` which should be used by guides.
    :return: sample from the stochastic `fn`.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn(rng_key=rng_key, sample_shape=sample_shape)

    if obs_mask is not None:
        return _masked_observe(
            name, fn, obs, obs_mask, rng_key=rng_key, sample_shape=(), infer=infer
        )

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": (),
        "kwargs": {"rng_key": rng_key, "sample_shape": sample_shape},
        "value": obs,
        "scale": None,
        "is_observed": obs is not None,
        "intermediates": [],
        "cond_indep_stack": [],
        "infer": {} if infer is None else infer,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


def param(name, init_value=None, **kwargs):
    """
    Annotate the given site as an optimizable parameter for use with
    :mod:`jax.experimental.optimizers`. For an example of how `param` statements
    can be used in inference algorithms, refer to :class:`~numpyro.infer.SVI`.

    :param str name: name of site.
    :param init_value: initial value specified by the user or a lazy callable
        that accepts a JAX random PRNGKey and returns an array.
        Note that the onus of using this to initialize the optimizer is
        on the user inference algorithm, since there is no global parameter
        store in NumPyro.
    :type init_value: numpy.ndarray or callable
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
        assert not callable(
            init_value
        ), "A callable init_value needs to be put inside a numpyro.handlers.seed handler."
        return init_value

    if callable(init_value):

        def fn(init_fn, *args, **kwargs):
            return init_fn(prng_key())

    else:
        fn = identity

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value,),
        "kwargs": kwargs,
        "value": None,
        "scale": None,
        "cond_indep_stack": [],
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


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

    initial_msg = {"type": "deterministic", "name": name, "value": value}

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


def mutable(name, init_value=None):
    """
    This primitive is used to store a mutable value that can be changed
    during model execution::

        a = numpyro.mutable("a", {"value": 1.})
        a["value"] = 2.
        assert numpyro.mutable("a")["value"] == 2.

    For example, this can be used to store and update information like
    running mean/variance in a neural network batch normalization layer.

    :param str name: name of the mutable site.
    :param init_value: mutable value to record in the trace.
    """
    if not _PYRO_STACK:
        return init_value

    initial_msg = {
        "type": "mutable",
        "name": name,
        "fn": identity,
        "args": (init_value,),
        "kwargs": {},
        "value": init_value,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


def _inspect():
    """
    EXPERIMENTAL Inspect the Pyro stack.

    .. warning:: The format of the returned message may change at any time and
        does not guarantee backwards compatibility.

    :returns: A message with mask effects applied.
    :rtype: dict
    """
    # NB: this is different from Pyro that in Pyro, all effects applied.
    # Here, we only apply mask effect handler.
    msg = {
        "type": "inspect",
        "fn": lambda: True,
        "args": (),
        "kwargs": {},
        "value": None,
        "mask": None,
    }
    apply_stack(msg)
    return msg


def get_mask():
    """
    Records the effects of enclosing ``handlers.mask`` handlers.
    This is useful for avoiding expensive ``numpyro.factor()`` computations during
    prediction, when the log density need not be computed, e.g.::

        def model():
            # ...
            if numpyro.get_mask() is not False:
                log_density = my_expensive_computation()
                numpyro.factor("foo", log_density)
            # ...

    :returns: The mask.
    :rtype: None, bool, or numpy.ndarray
    """
    return _inspect()["mask"]


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
    module_key = name + "$params"
    nn_init, nn_apply = nn
    nn_params = param(module_key)
    if nn_params is None:
        if input_shape is None:
            raise ValueError("Valid value for `input_shape` needed to initialize.")
        rng_key = prng_key()
        _, nn_params = nn_init(rng_key, input_shape)
        param(module_key, nn_params)
    return functools.partial(nn_apply, nn_params)


def _subsample_fn(size, subsample_size, rng_key=None):
    assert rng_key is not None, "Missing random key to generate subsample indices."
    if jax.default_backend() == "cpu":
        # ref: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        rng_keys = random.split(rng_key, subsample_size)

        def body_fn(val, idx):
            i_p1 = size - idx
            i = i_p1 - 1
            j = random.randint(rng_keys[idx], (), 0, i_p1)
            val = ops.index_update(
                val,
                ops.index[
                    [i, j],
                ],
                val[
                    ops.index[
                        [j, i],
                    ]
                ],
            )
            return val, None

        val, _ = lax.scan(body_fn, jnp.arange(size), jnp.arange(subsample_size))
        return val[-subsample_size:]
    else:
        return random.choice(rng_key, size, (subsample_size,), replace=False)


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
        assert size > 0, "size of plate should be positive"
        self.size = size
        if dim is not None and dim >= 0:
            raise ValueError("dim arg must be negative.")
        self.dim, self._indices = self._subsample(
            self.name, self.size, subsample_size, dim
        )
        self.subsample_size = self._indices.shape[0]
        super(plate, self).__init__()

    # XXX: different from Pyro, this method returns dim and indices
    @staticmethod
    def _subsample(name, size, subsample_size, dim):
        msg = {
            "type": "plate",
            "fn": _subsample_fn,
            "name": name,
            "args": (size, subsample_size),
            "kwargs": {"rng_key": None},
            "value": (
                None
                if (subsample_size is not None and size != subsample_size)
                else jnp.arange(size)
            ),
            "scale": 1.0,
            "cond_indep_stack": [],
        }
        apply_stack(msg)
        subsample = msg["value"]
        subsample_size = msg["args"][1]
        if subsample_size is not None and subsample_size != subsample.shape[0]:
            warnings.warn(
                "subsample_size does not match len(subsample), {} vs {}.".format(
                    subsample_size, len(subsample)
                )
                + " Did you accidentally use different subsample_size in the model and guide?"
            )
        cond_indep_stack = msg["cond_indep_stack"]
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
        if msg["type"] not in ("param", "sample", "plate"):
            if msg["type"] == "control_flow":
                raise NotImplementedError(
                    "Cannot use control flow primitive under a `plate` primitive."
                    " Please move those `plate` statements into the control flow"
                    " body function. See `scan` documentation for more information."
                )
            return

        cond_indep_stack = msg["cond_indep_stack"]
        frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size)
        cond_indep_stack.append(frame)
        if msg["type"] == "sample":
            expected_shape = self._get_batch_shape(cond_indep_stack)
            dist_batch_shape = msg["fn"].batch_shape
            if "sample_shape" in msg["kwargs"]:
                dist_batch_shape = msg["kwargs"]["sample_shape"] + dist_batch_shape
                msg["kwargs"]["sample_shape"] = ()
            overlap_idx = max(len(expected_shape) - len(dist_batch_shape), 0)
            trailing_shape = expected_shape[overlap_idx:]
            broadcast_shape = lax.broadcast_shapes(
                trailing_shape, tuple(dist_batch_shape)
            )
            batch_shape = expected_shape[:overlap_idx] + broadcast_shape
            msg["fn"] = msg["fn"].expand(batch_shape)
        if self.size != self.subsample_size:
            scale = 1.0 if msg["scale"] is None else msg["scale"]
            msg["scale"] = scale * (
                self.size / self.subsample_size if self.subsample_size else 1
            )

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
                            statement = "numpyro.param({}, ..., event_dim={})".format(
                                msg["name"], event_dim
                            )
                        else:
                            statement = "numpyro.subsample(..., event_dim={})".format(
                                event_dim
                            )
                        raise ValueError(
                            "Inside numpyro.plate({}, {}, dim={}) invalid shape of {}: {}".format(
                                self.name, self.size, self.dim, statement, shape
                            )
                        )
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


def prng_key():
    """
    A statement to draw a pseudo-random number generator key
    :func:`~jax.random.PRNGKey` under :class:`~numpyro.handlers.seed` handler.

    :return: a PRNG key of shape (2,) and dtype unit32.
    """
    if not _PYRO_STACK:
        return

    initial_msg = {
        "type": "prng_key",
        "fn": lambda rng_key: rng_key,
        "args": (),
        "kwargs": {"rng_key": None},
        "value": None,
    }

    msg = apply_stack(initial_msg)
    return msg["value"]


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
        "type": "subsample",
        "value": data,
        "kwargs": {"event_dim": event_dim},
    }

    msg = apply_stack(initial_msg)
    return msg["value"]
