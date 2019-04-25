# Source code modified from scipy.stats._distn_infrastructure.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.
from contextlib import contextmanager

import scipy.stats as osp_stats
from scipy._lib._util import getargspec_no_self
from scipy.stats._distn_infrastructure import instancemethod, rv_frozen, rv_generic

import jax.numpy as np
from jax import lax
from jax.numpy.lax_numpy import _promote_args
from jax.random import _is_prng_key
from jax.scipy import stats

from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform


class jax_frozen(rv_frozen):
    _validate_args = False

    def __init__(self, dist, *args, **kwargs):
        self.args = args
        self.kwds = kwargs

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        shapes, _, scale = self.dist._parse_args(*args, **kwargs)
        if self._validate_args:
            # TODO: check more concretely for each parameter
            if not np.all(self.dist._argcheck(*shapes)):
                raise ValueError('Invalid parameters provided to the distribution.')
            if not np.all(scale > 0):
                raise ValueError('Invalid scale parameter provided to the distribution.')

        self.a, self.b = self.dist.a, self.dist.b

    @property
    def support(self):
        return self.dist._support(*self.args, **self.kwds)

    def __call__(self, size=None, random_state=None):
        return self.rvs(size, random_state)

    def logpdf(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def logpmf(self, k):
        if self._validate_args:
            self._validate_sample(k)
        return self.dist.logpmf(k, *self.args, **self.kwds)

    def _validate_sample(self, x):
        if not np.all(self.support(x)):
            raise ValueError('Invalid values provided to log prob method. '
                             'The value argument must be within the support.')


class jax_generic(rv_generic):
    arg_constraints = {}

    def freeze(self, *args, **kwargs):
        return jax_frozen(self, *args, **kwargs)

    def _argcheck(self, *args):
        cond = 1
        constraints = self.arg_constraints
        if args:
            for arg, arg_name in zip(args, self.shapes.split(', ')):
                if arg_name in constraints:
                    cond = np.logical_and(cond, constraints[arg_name](arg))
        return cond


class jax_continuous(jax_generic, osp_stats.rv_continuous):
    def _support(self, *args, **kwargs):
        # support of the transformed distribution
        _, loc, scale = self._parse_args(*args, **kwargs)
        return AffineTransform(loc, scale, domain=self._support_mask).codomain

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
        # XXX when args is not empty, parse_args requires either _pdf or _cdf method is implemented
        # to recognize valid arg signatures (e.g. `a` in `gamma` or `s` in lognormal)
        args, loc, scale = self._parse_args(*args, **kwargs)
        # XXX using _promote_args_like requires calling `super(jax_continuous, self).rvs` but
        # it will call `self._rvs` (which is written using JAX and requires JAX random state).
        loc, scale, *args = _promote_args("rvs", loc, scale, *args)
        if not size:
            shapes = [np.shape(arg) for arg in args] + [np.shape(loc), np.shape(scale)]
            size = lax.broadcast_shapes(*shapes)
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


class jax_discrete(jax_generic, osp_stats.rv_discrete):
    def __new__(cls, *args, **kwargs):
        return super(jax_discrete, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        self.is_logits = kwargs.pop("is_logits", False)
        super(jax_discrete, self).__init__(*args, **kwargs)

    def freeze(self, *args, **kwargs):
        self._ctor_param.update(is_logits=kwargs.pop("is_logits", False))
        return super(jax_discrete, self).freeze(*args, **kwargs)

    def _support(self, *args, **kwargs):
        args, loc, _ = self._parse_args(*args, **kwargs)
        support_mask = self._support_mask
        if isinstance(support_mask, constraints.integer_interval):
            return constraints.integer_interval(loc + support_mask.lower_bound,
                                                loc + support_mask.upper_bound)
        elif isinstance(support_mask, constraints.integer_greater_than):
            return constraints.integer_greater_than(loc + support_mask.lower_bound)
        else:
            raise NotImplementedError

    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert _is_prng_key(rng)

        args = list(args)
        size = kwargs.pop('size', args.pop() if len(args) > (self.numargs + 1) else None)
        args, loc, _ = self._parse_args(*args, **kwargs)
        loc, *args = _promote_args("rvs", loc, *args)
        if not size:
            shapes = [np.shape(arg) for arg in args] + [np.shape(loc)]
            size = lax.broadcast_shapes(*shapes)
        elif isinstance(size, int):
            size = (size,)

        self._random_state = rng
        self._size = size
        vals = self._rvs(*args)
        return vals + loc

    def logpmf(self, k, *args, **kwargs):
        args, loc, _ = self._parse_args(*args, **kwargs)
        k = k - loc
        return self._logpmf(k, *args)


_parse_arg_template = """
def _parse_args(self, {shape_arg_str}):
    return ({shape_arg_str}), 0, 1
"""


class jax_multivariate(jax_generic):
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

    def _support(self, *args, **kwargs):
        return self._support_mask

    def rvs(self, *args, **kwargs):
        rng = kwargs.pop('random_state')
        if rng is None:
            rng = self.random_state
        # assert that rng is PRNGKey and not mtrand.RandomState object from numpy.
        assert _is_prng_key(rng)

        args = list(args)
        size = kwargs.pop('size', args.pop() if len(args) > self.numargs else None)

        args, _, _ = self._parse_args(*args, **kwargs)

        if not size:
            size = self._batch_shape(*args)
        elif isinstance(size, int):
            size = (size,)

        self._random_state = rng
        self._size = size
        return self._rvs(*args)


@contextmanager
def validation_enabled():
    jax_frozen_flag = jax_frozen._validate_args
    try:
        jax_frozen._validate_args = True
        yield
    finally:
        jax_frozen._validate_args = jax_frozen_flag
