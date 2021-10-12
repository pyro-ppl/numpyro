# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from jax import device_put, lax

from numpyro import handlers
from numpyro.contrib.control_flow.util import PytreeTrace
from numpyro.primitives import _PYRO_STACK, apply_stack


def _subs_wrapper(subs_map, site):
    if isinstance(subs_map, dict) and site["name"] in subs_map:
        return subs_map[site["name"]]
    elif callable(subs_map):
        rng_key = site["kwargs"].get("rng_key")
        subs_map = (
            handlers.seed(subs_map, rng_seed=rng_key)
            if rng_key is not None
            else subs_map
        )
        return subs_map(site)
    return None


def wrap_fn(fn, substitute_stack):
    def wrapper(wrapped_operand):
        rng_key, operand = wrapped_operand

        with handlers.block():
            seeded_fn = handlers.seed(fn, rng_key) if rng_key is not None else fn
            for subs_type, subs_map in substitute_stack:
                subs_fn = partial(_subs_wrapper, subs_map)
                if subs_type == "condition":
                    seeded_fn = handlers.condition(seeded_fn, condition_fn=subs_fn)
                elif subs_type == "substitute":
                    seeded_fn = handlers.substitute(seeded_fn, substitute_fn=subs_fn)

            with handlers.trace() as trace:
                value = seeded_fn(operand)

        return value, PytreeTrace(trace)

    return wrapper


def cond_wrapper(
    pred,
    true_fun,
    false_fun,
    operand,
    rng_key=None,
    substitute_stack=None,
    enum=False,
    first_available_dim=None,
):
    if enum:
        # TODO: support enumeration. note that pred passed to lax.cond must be scalar
        # which means that even simple conditions like `x == 0` can get complicated if
        # x is an enumerated discrete random variable
        raise RuntimeError("The cond primitive does not currently support enumeration")

    if substitute_stack is None:
        substitute_stack = []

    wrapped_true_fun = wrap_fn(true_fun, substitute_stack)
    wrapped_false_fun = wrap_fn(false_fun, substitute_stack)
    wrapped_operand = device_put((rng_key, operand))
    return lax.cond(pred, wrapped_true_fun, wrapped_false_fun, wrapped_operand)


def cond(pred, true_fun, false_fun, operand):
    """
    This primitive conditionally applies ``true_fun`` or ``false_fun``. See
    :func:`jax.lax.cond` for more information.

    **Usage**:

    .. doctest::

       >>> import numpyro
       >>> import numpyro.distributions as dist
       >>> from jax import random
       >>> from numpyro.contrib.control_flow import cond
       >>> from numpyro.infer import SVI, Trace_ELBO
       >>>
       >>> def model():
       ...     def true_fun(_):
       ...         return numpyro.sample("x", dist.Normal(20.0))
       ...
       ...     def false_fun(_):
       ...         return numpyro.sample("x", dist.Normal(0.0))
       ...
       ...     cluster = numpyro.sample("cluster", dist.Normal())
       ...     return cond(cluster > 0, true_fun, false_fun, None)
       >>>
       >>> def guide():
       ...     m1 = numpyro.param("m1", 10.0)
       ...     s1 = numpyro.param("s1", 0.1, constraint=dist.constraints.positive)
       ...     m2 = numpyro.param("m2", 10.0)
       ...     s2 = numpyro.param("s2", 0.1, constraint=dist.constraints.positive)
       ...
       ...     def true_fun(_):
       ...         return numpyro.sample("x", dist.Normal(m1, s1))
       ...
       ...     def false_fun(_):
       ...         return numpyro.sample("x", dist.Normal(m2, s2))
       ...
       ...     cluster = numpyro.sample("cluster", dist.Normal())
       ...     return cond(cluster > 0, true_fun, false_fun, None)
       >>>
       >>> svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO(num_particles=100))
       >>> svi_result = svi.run(random.PRNGKey(0), num_steps=2500)

    .. warning:: This is an experimental utility function that allows users to use
        JAX control flow with NumPyro's effect handlers. Currently, `sample` and
        `deterministic` sites within `true_fun` and `false_fun` are supported. If you
        notice that any effect handlers or distributions are unsupported, please file
        an issue.

    .. warning:: The ``cond`` primitive does not currently support enumeration and can
        not be used inside a ``numpyro.plate`` context.

    .. note:: All ``sample`` sites must belong to the same distribution class. For
        example the following is not supported

        .. code-block:: python

            cond(
                True,
                lambda _: numpyro.sample("x", dist.Normal()),
                lambda _: numpyro.sample("x", dist.Laplace()),
                None,
            )

    :param bool pred: Boolean scalar type indicating which branch function to apply
    :param callable true_fun: A function to be applied if ``pred`` is true.
    :param callable false_fun: A function to be applied if ``pred`` is false.
    :param operand: Operand input to either branch depending on ``pred``. This can
        be any JAX PyTree (e.g. list / dict of arrays).
    :return: Output of the applied branch function.
    """
    if not _PYRO_STACK:
        value, _ = cond_wrapper(pred, true_fun, false_fun, operand)
        return value

    initial_msg = {
        "type": "control_flow",
        "fn": cond_wrapper,
        "args": (pred, true_fun, false_fun, operand),
        "kwargs": {"rng_key": None, "substitute_stack": []},
        "value": None,
    }

    msg = apply_stack(initial_msg)
    value, pytree_trace = msg["value"]

    for msg in pytree_trace.trace.values():
        apply_stack(msg)

    return value
