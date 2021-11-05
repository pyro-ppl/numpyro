# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp


def _is_batched(arg):
    return jnp.ndim(arg) > 0


def vindex(tensor, args):
    """
    Vectorized advanced indexing with broadcasting semantics.

    See also the convenience wrapper :class:`Vindex`.

    This is useful for writing indexing code that is compatible with batching
    and enumeration, especially for selecting mixture components with discrete
    random variables.

    For example suppose ``x`` is a parameter with ``len(x.shape) == 3`` and we wish
    to generalize the expression ``x[i, :, j]`` from integer ``i,j`` to tensors
    ``i,j`` with batch dims and enum dims (but no event dims). Then we can
    write the generalize version using :class:`Vindex` ::

        xij = Vindex(x)[i, :, j]

        batch_shape = broadcast_shape(i.shape, j.shape)
        event_shape = (x.size(1),)
        assert xij.shape == batch_shape + event_shape

    To handle the case when ``x`` may also contain batch dimensions (e.g. if
    ``x`` was sampled in a plated context as when using vectorized particles),
    :func:`vindex` uses the special convention that ``Ellipsis`` denotes batch
    dimensions (hence ``...`` can appear only on the left, never in the middle
    or in the right). Suppose ``x`` has event dim 3. Then we can write::

        old_batch_shape = x.shape[:-3]
        old_event_shape = x.shape[-3:]

        xij = Vindex(x)[..., i, :, j]   # The ... denotes unknown batch shape.

        new_batch_shape = broadcast_shape(old_batch_shape, i.shape, j.shape)
        new_event_shape = (x.size(1),)
        assert xij.shape = new_batch_shape + new_event_shape

    Note that this special handling of ``Ellipsis`` differs from the NEP [1].

    Formally, this function assumes:

    1.  Each arg is either ``Ellipsis``, ``slice(None)``, an integer, or a
        batched integer tensor (i.e. with empty event shape). This
        function does not support Nontrivial slices or boolean tensor
        masks. ``Ellipsis`` can only appear on the left as ``args[0]``.
    2.  If ``args[0] is not Ellipsis`` then ``tensor`` is not
        batched, and its event dim is equal to ``len(args)``.
    3.  If ``args[0] is Ellipsis`` then ``tensor`` is batched and
        its event dim is equal to ``len(args[1:])``. Dims of ``tensor``
        to the left of the event dims are considered batch dims and will be
        broadcasted with dims of tensor args.

    Note that if none of the args is a tensor with ``len(shape) > 0``, then this
    function behaves like standard indexing::

        if not any(isinstance(a, jnp.ndarray) and len(a.shape) > 0 for a in args):
            assert Vindex(x)[args] == x[args]

    **References**

    [1] https://www.numpy.org/neps/nep-0021-advanced-indexing.html
        introduces ``vindex`` as a helper for vectorized indexing.
        This implementation is similar to the proposed notation
        ``x.vindex[]`` except for slightly different handling of ``Ellipsis``.

    :param jnp.ndarray tensor: A tensor to be indexed.
    :param tuple args: An index, as args to ``__getitem__``.
    :returns: A nonstandard interpretation of ``tensor[args]``.
    :rtype: jnp.ndarray
    """
    if not isinstance(args, tuple):
        return tensor[args]
    if not args:
        return tensor

    assert jnp.ndim(tensor) > 0
    # Compute event dim before and after indexing.
    if args[0] is Ellipsis:
        args = args[1:]
        if not args:
            return tensor
        old_event_dim = len(args)
        args = (slice(None),) * (jnp.ndim(tensor) - len(args)) + args
    else:
        args = args + (slice(None),) * (jnp.ndim(tensor) - len(args))
        old_event_dim = len(args)
    assert len(args) == jnp.ndim(tensor)
    if any(a is Ellipsis for a in args):
        raise NotImplementedError("Non-leading Ellipsis is not supported")

    # In simple cases, standard advanced indexing broadcasts correctly.
    is_standard = True
    if jnp.ndim(tensor) > old_event_dim and _is_batched(args[0]):
        is_standard = False
    elif any(_is_batched(a) for a in args[1:]):
        is_standard = False
    if is_standard:
        return tensor[args]

    # Convert args to use broadcasting semantics.
    new_event_dim = sum(isinstance(a, slice) for a in args[-old_event_dim:])
    new_dim = 0
    args = list(args)
    for i, arg in reversed(list(enumerate(args))):
        if isinstance(arg, slice):
            # Convert slices to arange()s.
            if arg != slice(None):
                raise NotImplementedError("Nontrivial slices are not supported")
            arg = jnp.arange(tensor.shape[i], dtype=jnp.int32)
            arg = arg.reshape((-1,) + (1,) * new_dim)
            new_dim += 1
        elif _is_batched(arg):
            # Reshape nontrivial tensors.
            arg = arg.reshape(arg.shape + (1,) * new_event_dim)
        args[i] = arg
    args = tuple(args)

    return tensor[args]


class Vindex:
    """
    Convenience wrapper around :func:`vindex`.

    The following are equivalent::

        Vindex(x)[..., i, j, :]
        vindex(x, (Ellipsis, i, j, slice(None)))

    :param jnp.ndarray tensor: A tensor to be indexed.
    :return: An object with a special :meth:`__getitem__` method.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, args):
        return vindex(self._tensor, args)
