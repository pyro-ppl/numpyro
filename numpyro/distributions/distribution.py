# Source code modified from scipy.stats._distn_infrastructure.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.

import numpy as onp
import scipy.stats as osp_stats
from scipy._lib._util import getargspec_no_self
from scipy.stats._distn_infrastructure import instancemethod

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
        # TODO(fehiepsi): add test for int size
        elif isinstance(size, int):
            size = (size,)
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

    def _support_mask(self, k):
        return (k >= self.a) & (k <= self.b) & (np.floor(k) == k)

    # Discrete distribution instances use scipy samplers directly
    # and put the samples on device later.
    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert _is_prng_key(rng)
        kwargs['random_state'] = onp.random.RandomState(rng)
        sample = super(jax_discrete, self).rvs(*args, **kwargs)
        return device_put(sample)

    def logpmf(self, k, *args, **kwargs):
        args, loc, _ = self._parse_args(*args, **kwargs)
        k = k - loc
        if self.args_check:
            cond0 = self._argcheck(*args)
            cond1 = self._support_mask(k)
            if not np.all(cond0):
                raise ValueError('Invalid distribution arguments provided to {}.logpmf'.format(self))
            if not np.all(cond1):
                raise ValueError('Invalid values provided to {}.logpmf'.format(self))
        return self._logpmf(k, *args)


_parse_arg_template = """
def _parse_args(self, {shape_arg_str}):
    return ({shape_arg_str}), 0, 1
"""


class jax_mvcontinuous(osp_stats.rv_continuous):
    def _construct_argparser(self, *args, **kwargs):
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
        else:
            shapes = getargspec_no_self(self._rvs).args

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str)
        ns = {}
        exec(_parse_arg_template.format(**dct), ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args']:
            setattr(self, name, instancemethod(ns[name], self, self.__class__))

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            self.numargs = len(shapes)

    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert _is_prng_key(rng)

        args = list(args)
        size = kwargs.pop('size', args.pop() if len(args) > self.numargs else None)

        args, _, _ = self._parse_args(*args, **kwargs)
        # XXX we might not need to verify that args is empty for multivariate distributions
        args = _promote_args("rvs", *args)

        # TODO: make this code compatible to mvn distribution
        if not size:
            size = args[-1].shape[:-1]
        elif isinstance(size, int):
            size = (size,)

        self._random_state = rng
        self._size = size
        return self._rvs(*args)

    def logpdf(self, *args, **kwargs):
        # TODO: check args, check input
        return self._logpdf(*args, **kwargs)
