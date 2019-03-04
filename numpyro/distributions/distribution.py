# Source code modified from scipy.stats._distn_infrastructure.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

from collections import deque

import scipy.stats as sp
from jax import lax
from jax.numpy.lax_numpy import _promote_args
import jax.numpy as np
from numpy.random import mtrand


class jax_continuous(sp.rv_continuous):
    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert not isinstance(rng, mtrand.RandomState)
        args = deque(args)
        loc = kwargs.get('loc', args.popleft() if len(args) > 0 else 0)
        scale = kwargs.get('scale', args.popleft() if len(args) > 0 else 1)
        size = kwargs.get('size', args.popleft() if len(args) > 0 else None)
        # FIXME(fehiepsi): Using _promote_args_like requires calling `super(jax_continuous, self).rvs` but
        # it will call `self._rvs` (which is written using JAX and requires JAX random state).
        loc, scale, *args = _promote_args("rvs", loc, scale, *args)
        if not size:
            shapes = [np.shape(arg) for arg in args] + [np.shape(loc), np.shape(scale)]
            size = lax.broadcast_shapes(*shapes)
        else:
            args = [np.reshape(arg, size) for arg in args]
        self._random_state = rng
        self._size = size
        vals = self._rvs(*args)
        return vals * scale + loc

    def logpdf(self, x, *args, **kwargs):
        args = deque(args)
        loc = kwargs.get('loc', args.popleft() if len(args) > 0 else 0)
        scale = kwargs.get('scale', args.popleft() if len(args) > 0 else 1)
        loc, scale, *args = _promote_args(self.logpdf, loc, scale, *args)
        x = (x - loc) / scale
        return self._logpdf(x) - np.log(scale)


class jax_discrete(sp.rv_discrete):
    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert not isinstance(rng, mtrand.RandomState)
        args = deque(args)
        loc = kwargs.get('loc', args.popleft() if len(args) > 0 else 0)
        size = kwargs.get('size', args.popleft() if len(args) > 0 else None)
        loc, *args = _promote_args("rvs", loc, *args)
        if not size:
            shapes = [np.shape(arg) for arg in args] + [np.shape(loc)]
            size = lax.broadcast_shapes(*shapes)
        else:
            args = [np.reshape(arg, size) for arg in args]
        self._random_state = rng
        self._size = size
        vals = self._rvs(*args)
        return vals + loc

    def logpmf(self, k, *args, **kwds):
        args = deque(args)
        loc = kwargs.get('loc', args.popleft() if len(args) > 0 else 0)
        k, loc = map(np.asarray, (k, loc))
        args = tuple(map(np.asarray, args))
        k = np.asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k <= self.b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = np.empty(np.shape(cond), 'd')
        output.fill(np.NINF)
        np.place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = np.argsreduce(cond, *((k,)+args))
            np.place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output
