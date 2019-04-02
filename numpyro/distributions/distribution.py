# Source code modified from scipy.stats._distn_infrastructure.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import numpy as onp
import scipy.stats as osp_stats

import jax.numpy as np
from jax import device_put, lax
from jax.numpy.lax_numpy import _promote_args
from jax.random import _is_prng_key
from jax.scipy import stats


class jax_continuous(osp_stats.rv_continuous):
    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert _is_prng_key(rng)
        args = list(args)
        # If 'size' is not in kwargs, then it is either the last element of args
        # or it will take default value (which is None).
        # Note: self.numargs is the number of shape parameters.
        size = kwargs.pop('size', args.pop() if len(args) > (self.numargs + 2) else None)
        # TODO: when args is not empty, parse_args requires either _pdf or _cdf method is implemented
        # to recognize valid arg signatures (e.g. `a` in `gamma` or `s` in lognormal)
        args, loc, scale = self._parse_args(*args, **kwargs)
        # FIXME(fehiepsi): Using _promote_args_like requires calling `super(jax_continuous, self).rvs` but
        # it will call `self._rvs` (which is written using JAX and requires JAX random state).
        loc, scale, *args = _promote_args("rvs", loc, scale, *args)
        if not size:
            shapes = [np.shape(arg) for arg in args] + [np.shape(loc), np.shape(scale)]
            size = lax.broadcast_shapes(*shapes)
        self._random_state = rng
        self._size = size
        vals = self._rvs(*args)
        return vals * scale + loc

    def pdf(self, x, *args, **kwargs):
        if hasattr(stats, self.name):
            return getattr(stats, self.name).pdf(x, *args, **kwargs)
        else:
            return super(jax_continuous, self).pdf(x, *args, **kwargs)

    def logpdf(self, x, *args, **kwargs):
        if hasattr(stats, self.name):
            return getattr(stats, self.name).logpdf(x, *args, **kwargs)
        else:
            return super(jax_continuous, self).logpdf(x, *args, **kwargs)


class jax_discrete(osp_stats.rv_discrete):
    args_check = True

    # Discrete distribution instances use scipy samplers directly
    # and put the samples on device later.
    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert _is_prng_key(rng)
        kwargs['random_state'] = onp.random.RandomState(rng)
        sample = super(osp_stats.rv_discrete, self).rvs(*args, **kwargs)
        return device_put(sample)

    def logpmf(self, k, *args, **kwds):
        args, loc, _ = self._parse_args(*args, **kwds)
        k = k - loc
        if self.args_check:
            cond0 = self._argcheck(*args)
            cond1 = (k >= self.a) & (k <= self.b) & (np.floor(k) == k)
            if not np.all(cond0):
                raise ValueError('Invalid distribution arguments provided to {}.logpmf'.format(self))
            if not np.all(cond1):
                raise ValueError('Invalid values provided to {}.logpmf'.format(self))
        return self._logpmf(k, *args)
