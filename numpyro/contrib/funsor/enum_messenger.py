# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3
from enum import Enum

from jax import lax
import jax.numpy as jnp

import funsor
from numpyro.handlers import infer_config, trace as OrigTraceMessenger
from numpyro.primitives import Messenger, apply_stack, plate as OrigPlateMessenger

funsor.set_backend("jax")


__all__ = ["enum", "infer_config", "markov", "plate", "to_data", "to_funsor", "trace"]


##################################
# DimStack to store global state
##################################

# name_to_dim : dict, dim_to_name : dict, parents : tuple, iter_parents : tuple
class StackFrame(
    namedtuple(
        "StackFrame", ["name_to_dim", "dim_to_name", "parents", "iter_parents", "keep"]
    )
):
    def read(self, name, dim):
        found_name = self.dim_to_name.get(dim, name)
        found_dim = self.name_to_dim.get(name, dim)
        found = name in self.name_to_dim or dim in self.dim_to_name
        return found_name, found_dim, found

    def write(self, name, dim):
        assert name is not None and dim is not None
        self.dim_to_name[dim] = name
        self.name_to_dim[name] = dim

    def free(self, name, dim):
        self.dim_to_name.pop(dim, None)
        self.name_to_dim.pop(name, None)
        return name, dim


class DimType(Enum):
    """Enumerates the possible types of dimensions to allocate"""

    LOCAL = 0
    GLOBAL = 1
    VISIBLE = 2


DimRequest = namedtuple("DimRequest", ["dim", "dim_type"])
DimRequest.__new__.__defaults__ = (None, DimType.LOCAL)
NameRequest = namedtuple("NameRequest", ["name", "dim_type"])
NameRequest.__new__.__defaults__ = (None, DimType.LOCAL)


class DimStack:
    """
    Single piece of global state to keep track of the mapping between names and dimensions.

    Replaces the plate DimAllocator, the enum EnumAllocator, the stack in MarkovMessenger,
    _param_dims and _value_dims in EnumMessenger, and dim_to_symbol in msg['infer']
    """

    def __init__(self):
        self._stack = [
            StackFrame(
                name_to_dim=OrderedDict(),
                dim_to_name=OrderedDict(),
                parents=(),
                iter_parents=(),
                keep=False,
            )
        ]
        self._first_available_dim = -1
        self.outermost = None

    MAX_DIM = -25

    def set_first_available_dim(self, dim):
        assert dim is None or (self.MAX_DIM < dim < 0)
        old_dim, self._first_available_dim = self._first_available_dim, dim
        return old_dim

    def push(self, frame):
        self._stack.append(frame)

    def pop(self):
        assert len(self._stack) > 1, "cannot pop the global frame"
        return self._stack.pop()

    @property
    def current_frame(self):
        return self._stack[-1]

    @property
    def global_frame(self):
        return self._stack[0]

    def _gendim(self, name_request, dim_request):
        assert isinstance(name_request, NameRequest) and isinstance(
            dim_request, DimRequest
        )
        dim_type = dim_request.dim_type

        if name_request.name is None:
            fresh_name = f"_pyro_dim_{-dim_request.dim}"
        else:
            fresh_name = name_request.name

        conflict_frames = (
            (self.current_frame, self.global_frame)
            + self.current_frame.parents
            + self.current_frame.iter_parents
        )
        if dim_request.dim is None:
            fresh_dim = self._first_available_dim if dim_type != DimType.VISIBLE else -1
            fresh_dim = -1 if fresh_dim is None else fresh_dim
            while any(fresh_dim in p.dim_to_name for p in conflict_frames):
                fresh_dim -= 1
        else:
            fresh_dim = dim_request.dim

        if (
            fresh_dim < self.MAX_DIM
            or any(fresh_dim in p.dim_to_name for p in conflict_frames)
            or (dim_type == DimType.VISIBLE and fresh_dim <= self._first_available_dim)
        ):
            raise ValueError(f"Ran out of free dims during allocation for {fresh_name}")

        return fresh_name, fresh_dim

    def request(self, name, dim):
        assert isinstance(name, NameRequest) ^ isinstance(dim, DimRequest)
        if isinstance(dim, DimRequest):
            dim, dim_type = dim.dim, dim.dim_type
        elif isinstance(name, NameRequest):
            name, dim_type = name.name, name.dim_type

        read_frames = (
            (self.global_frame,)
            if dim_type != DimType.LOCAL
            else (self.current_frame,)
            + self.current_frame.parents
            + self.current_frame.iter_parents
            + (self.global_frame,)
        )

        # read dimension
        for frame in read_frames:
            name, dim, found = frame.read(name, dim)
            if found:
                break

        # generate fresh name or dimension
        if not found:
            name, dim = self._gendim(
                NameRequest(name, dim_type), DimRequest(dim, dim_type)
            )

            write_frames = (
                (self.global_frame,)
                if dim_type != DimType.LOCAL
                else (self.current_frame,)
                + (self.current_frame.parents if self.current_frame.keep else ())
            )

            # store the fresh dimension
            for frame in write_frames:
                frame.write(name, dim)

        return name, dim


_DIM_STACK = DimStack()  # only one global instance


#################################################
# Messengers that implement guts of enumeration
#################################################


class ReentrantMessenger(Messenger):
    def __init__(self, fn=None):
        self._ref_count = 0
        super().__init__(fn)

    # def __call__(self, fn):
    #     return functools.wraps(fn)(super().__call__(fn))

    def __enter__(self):
        self._ref_count += 1
        if self._ref_count == 1:
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ref_count -= 1
        if self._ref_count == 0:
            super().__exit__(exc_type, exc_value, traceback)


class DimStackCleanupMessenger(ReentrantMessenger):
    def __init__(self, fn=None):
        self._saved_dims = ()
        return super().__init__(fn)

    def __enter__(self):
        if self._ref_count == 0 and _DIM_STACK.outermost is None:
            _DIM_STACK.outermost = self
            for name, dim in self._saved_dims:
                _DIM_STACK.global_frame.write(name, dim)
            self._saved_dims = ()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1 and _DIM_STACK.outermost is self:
            _DIM_STACK.outermost = None
            for name, dim in reversed(
                tuple(_DIM_STACK.global_frame.name_to_dim.items())
            ):
                self._saved_dims += (_DIM_STACK.global_frame.free(name, dim),)
        return super().__exit__(*args, **kwargs)


class NamedMessenger(DimStackCleanupMessenger):
    def process_message(self, msg):
        if msg["type"] == "to_funsor":
            self._pyro_to_funsor(msg)
        elif msg["type"] == "to_data":
            self._pyro_to_data(msg)

    @staticmethod
    def _get_name_to_dim(batch_names, name_to_dim=None, dim_type=DimType.LOCAL):
        name_to_dim = OrderedDict() if name_to_dim is None else name_to_dim.copy()

        # interpret all names/dims as requests since we only run this function once
        for name in batch_names:
            dim = name_to_dim.get(name, None)
            name_to_dim[name] = (
                dim if isinstance(dim, DimRequest) else DimRequest(dim, dim_type)
            )

        # read dimensions and allocate fresh dimensions as necessary
        for name, dim_request in name_to_dim.items():
            name_to_dim[name] = _DIM_STACK.request(name, dim_request)[1]

        return name_to_dim

    @classmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_data(cls, msg):

        (funsor_value,) = msg["args"]
        name_to_dim = msg["kwargs"].setdefault("name_to_dim", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        batch_names = tuple(funsor_value.inputs.keys())

        name_to_dim.update(
            cls._get_name_to_dim(
                batch_names, name_to_dim=name_to_dim, dim_type=dim_type
            )
        )
        msg["stop"] = True  # only need to run this once per to_data call

    @staticmethod
    def _get_dim_to_name(batch_shape, dim_to_name=None, dim_type=DimType.LOCAL):
        dim_to_name = OrderedDict() if dim_to_name is None else dim_to_name.copy()
        batch_dim = len(batch_shape)

        # interpret all names/dims as requests since we only run this function once
        for dim in range(-batch_dim, 0):
            name = dim_to_name.get(dim, None)
            # the time dimension on the left sometimes necessitates empty dimensions appearing
            # before they have been assigned a name
            if batch_shape[dim] == 1 and name is None:
                continue
            dim_to_name[dim] = (
                name if isinstance(name, NameRequest) else NameRequest(name, dim_type)
            )

        for dim, name_request in dim_to_name.items():
            dim_to_name[dim] = _DIM_STACK.request(name_request, dim)[0]

        return dim_to_name

    @classmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_funsor(cls, msg):

        if len(msg["args"]) == 2:
            raw_value, output = msg["args"]
        else:
            raw_value = msg["args"][0]
            output = msg["kwargs"].setdefault("output", None)
        dim_to_name = msg["kwargs"].setdefault("dim_to_name", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        event_dim = len(output.shape) if output else 0
        try:
            batch_shape = raw_value.batch_shape  # TODO make make this more robust
        except AttributeError:
            batch_shape = raw_value.shape[: len(raw_value.shape) - event_dim]

        dim_to_name.update(
            cls._get_dim_to_name(
                batch_shape, dim_to_name=dim_to_name, dim_type=dim_type
            )
        )
        msg["stop"] = True  # only need to run this once per to_funsor call


class LocalNamedMessenger(NamedMessenger):
    """
    Handler for converting to/from funsors consistent with Pyro's positional batch dimensions.

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    """

    def __init__(self, fn=None, history=1, keep=False):
        self.history = history
        self.keep = keep
        self._iterable = None
        self._saved_frames = []
        self._iter_parents = ()
        super().__init__(fn)

    def generator(self, iterable):
        self._iterable = iterable
        return self

    def _get_iter_parents(self, frame):
        iter_parents = [frame]
        frontier = (frame,)
        while frontier:
            frontier = sum([p.iter_parents for p in frontier], ())
            iter_parents += frontier
        return tuple(iter_parents)

    def __iter__(self):
        assert self._iterable is not None
        self._iter_parents = self._get_iter_parents(_DIM_STACK.current_frame)
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        if self.keep and self._saved_frames:
            saved_frame = self._saved_frames.pop()
            name_to_dim, dim_to_name = saved_frame.name_to_dim, saved_frame.dim_to_name
        else:
            name_to_dim, dim_to_name = OrderedDict(), OrderedDict()

        frame = StackFrame(
            name_to_dim=name_to_dim,
            dim_to_name=dim_to_name,
            parents=tuple(
                reversed(_DIM_STACK._stack[len(_DIM_STACK._stack) - self.history :])
            ),
            iter_parents=tuple(self._iter_parents),
            keep=self.keep,
        )

        _DIM_STACK.push(frame)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self.keep:
            # don't keep around references to other frames
            old_frame = _DIM_STACK.pop()
            saved_frame = StackFrame(
                name_to_dim=old_frame.name_to_dim,
                dim_to_name=old_frame.dim_to_name,
                parents=(),
                iter_parents=(),
                keep=self.keep,
            )
            self._saved_frames.append(saved_frame)
        else:
            _DIM_STACK.pop()
        return super().__exit__(*args, **kwargs)


class GlobalNamedMessenger(NamedMessenger):
    def __init__(self, fn=None):
        self._saved_globals = ()
        super().__init__(fn)

    def __enter__(self):
        if self._ref_count == 0:
            for name, dim in self._saved_globals:
                _DIM_STACK.global_frame.write(name, dim)
            self._saved_globals = ()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1:
            for name, dim in self._saved_globals:
                _DIM_STACK.global_frame.free(name, dim)
        return super().__exit__(*args, **kwargs)

    def postprocess_message(self, msg):
        if msg["type"] == "to_funsor":
            self._pyro_post_to_funsor(msg)
        elif msg["type"] == "to_data":
            self._pyro_post_to_data(msg)

    def _pyro_post_to_funsor(self, msg):
        if msg["kwargs"]["dim_type"] in (DimType.GLOBAL, DimType.VISIBLE):
            for name in msg["value"].inputs:
                self._saved_globals += (
                    (name, _DIM_STACK.global_frame.name_to_dim[name]),
                )

    def _pyro_post_to_data(self, msg):
        if msg["kwargs"]["dim_type"] in (DimType.GLOBAL, DimType.VISIBLE):
            for name in msg["args"][0].inputs:
                self._saved_globals += (
                    (name, _DIM_STACK.global_frame.name_to_dim[name]),
                )


class BaseEnumMessenger(NamedMessenger):
    """
    Handles first_available_dim management, enum effects should inherit from this
    """

    def __init__(self, fn=None, first_available_dim=None):
        assert (
            first_available_dim is None or first_available_dim < 0
        ), first_available_dim
        self.first_available_dim = first_available_dim
        super().__init__(fn)

    def __enter__(self):
        if self._ref_count == 0 and self.first_available_dim is not None:
            self._prev_first_dim = _DIM_STACK.set_first_available_dim(
                self.first_available_dim
            )
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1 and self.first_available_dim is not None:
            _DIM_STACK.set_first_available_dim(self._prev_first_dim)
        return super().__exit__(*args, **kwargs)


##########################################
# User-facing handler implementations
##########################################


class plate(GlobalNamedMessenger):
    """
    An alternative implementation of :class:`numpyro.primitives.plate` primitive. Note
    that only this version is compatible with enumeration.

    There is also a context manager
    :func:`~numpyro.contrib.funsor.infer_util.plate_to_enum_plate`
    which converts `numpyro.plate` statements to this version.

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
            raise ValueError("dim arg must be negative.")
        self.dim, indices = OrigPlateMessenger._subsample(
            self.name, self.size, subsample_size, dim
        )
        self.subsample_size = indices.shape[0]
        self._indices = funsor.Tensor(
            indices,
            OrderedDict([(self.name, funsor.Bint[self.subsample_size])]),
            self.subsample_size,
        )
        super(plate, self).__init__(None)

    def __enter__(self):
        super().__enter__()  # do this first to take care of globals recycling
        name_to_dim = (
            OrderedDict([(self.name, self.dim)])
            if self.dim is not None
            else OrderedDict()
        )
        indices = to_data(
            self._indices, name_to_dim=name_to_dim, dim_type=DimType.VISIBLE
        )
        # extract the dimension allocated by to_data to match plate's current behavior
        self.dim, self.indices = -len(indices.shape), indices.squeeze()
        return self.indices

    @staticmethod
    def _get_batch_shape(cond_indep_stack):
        n_dims = max(-f.dim for f in cond_indep_stack)
        batch_shape = [1] * n_dims
        for f in cond_indep_stack:
            batch_shape[f.dim] = f.size
        return tuple(batch_shape)

    def process_message(self, msg):
        if msg["type"] in ["to_funsor", "to_data"]:
            return super().process_message(msg)
        return OrigPlateMessenger.process_message(self, msg)

    def postprocess_message(self, msg):
        if msg["type"] in ["to_funsor", "to_data"]:
            return super().postprocess_message(msg)
        # NB: copied literally from original plate messenger, with self._indices is replaced
        # by self.indices
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
                        new_value = jnp.take(value, self.indices, dim)
                        msg["value"] = new_value


class enum(BaseEnumMessenger):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param callable fn: Python callable with NumPyro primitives.
    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer or None.
    """

    def process_message(self, msg):
        if (
            msg["type"] != "sample"
            or msg.get("done", False)
            or msg["is_observed"]
            or msg["infer"].get("expand", False)
            or msg["infer"].get("enumerate") != "parallel"
            or (not msg["fn"].has_enumerate_support)
        ):
            if msg["type"] == "control_flow":
                msg["kwargs"]["enum"] = True
                msg["kwargs"]["first_available_dim"] = self.first_available_dim
            return super().process_message(msg)

        if msg["infer"].get("num_samples", None) is not None:
            raise NotImplementedError("TODO implement multiple sampling")

        if msg["infer"].get("expand", False):
            raise NotImplementedError("expand=True not implemented")

        size = msg["fn"].enumerate_support(expand=False).shape[0]
        raw_value = jnp.arange(0, size)
        funsor_value = funsor.Tensor(
            raw_value, OrderedDict([(msg["name"], funsor.Bint[size])]), size
        )

        msg["value"] = to_data(funsor_value)
        msg["done"] = True


class trace(OrigTraceMessenger):
    """
    This version of :class:`~numpyro.handlers.trace` handler records
    information necessary to do packing after execution.

    Each sample site is annotated with a "dim_to_name" dictionary,
    which can be passed directly to :func:`to_funsor`.
    """

    def postprocess_message(self, msg):
        if msg["type"] == "sample":
            total_batch_shape = lax.broadcast_shapes(
                tuple(msg["fn"].batch_shape),
                jnp.shape(msg["value"])[: jnp.ndim(msg["value"]) - msg["fn"].event_dim],
            )
            msg["infer"]["dim_to_name"] = NamedMessenger._get_dim_to_name(
                total_batch_shape
            )
            msg["infer"]["name_to_dim"] = {
                name: dim for dim, name in msg["infer"]["dim_to_name"].items()
            }
        if msg["type"] in ("sample", "param"):
            super().postprocess_message(msg)


def markov(fn=None, history=1, keep=False):
    """
    Markov dependency declaration.

    This is a statistical equivalent of a memory management arena.

    :param callable fn: Python callable with NumPyro primitives.
    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`numpyro.primitives.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    """
    if fn is not None and not callable(fn):  # Used as a generator
        return LocalNamedMessenger(fn=None, history=history, keep=keep).generator(
            iterable=fn
        )
    return LocalNamedMessenger(fn, history=history, keep=keep)


####################
# New primitives
####################


def to_funsor(x, output=None, dim_to_name=None, dim_type=DimType.LOCAL):
    """
    A primitive to convert a Python object to a :class:`~funsor.terms.Funsor`.

    :param x: An object.
    :param funsor.domains.Domain output: An optional output hint to uniquely
        convert a data to a Funsor (e.g. when `x` is a string).
    :param OrderedDict dim_to_name: An optional mapping from negative
        batch dimensions to name strings.
    :param int dim_type: Either 0, 1, or 2. This optional argument indicates
        a dimension should be treated as 'local', 'global', or 'visible',
        which can be used to interact with the global :class:`DimStack`.
    :return: A Funsor equivalent to `x`.
    :rtype: funsor.terms.Funsor
    """
    dim_to_name = OrderedDict() if dim_to_name is None else dim_to_name

    initial_msg = {
        "type": "to_funsor",
        "fn": lambda x, output, dim_to_name, dim_type: funsor.to_funsor(
            x, output=output, dim_to_name=dim_to_name
        ),
        "args": (x,),
        "kwargs": {"output": output, "dim_to_name": dim_to_name, "dim_type": dim_type},
        "value": None,
        "mask": None,
    }

    msg = apply_stack(initial_msg)
    return msg["value"]


def to_data(x, name_to_dim=None, dim_type=DimType.LOCAL):
    """
    A primitive to extract a python object from a :class:`~funsor.terms.Funsor`.

    :param ~funsor.terms.Funsor x: A funsor object
    :param OrderedDict name_to_dim: An optional inputs hint which maps
        dimension names from `x` to dimension positions of the returned value.
    :param int dim_type: Either 0, 1, or 2. This optional argument indicates
        a dimension should be treated as 'local', 'global', or 'visible',
        which can be used to interact with the global :class:`DimStack`.
    :return: A non-funsor equivalent to `x`.
    """
    name_to_dim = OrderedDict() if name_to_dim is None else name_to_dim

    initial_msg = {
        "type": "to_data",
        "fn": lambda x, name_to_dim, dim_type: funsor.to_data(
            x, name_to_dim=name_to_dim
        ),
        "args": (x,),
        "kwargs": {"name_to_dim": name_to_dim, "dim_type": dim_type},
        "value": None,
        "mask": None,
    }

    msg = apply_stack(initial_msg)
    return msg["value"]
