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
        size = kwargs.get('size', None)
        args = list(args)
        scale = kwargs.get('scale', args.pop() if len(args) > 0 else 1)
        loc = kwargs.get('loc', args.pop() if len(args) > 0 else 0)
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
        args = list(args)
        loc = kwargs.get('loc', args.pop() if len(args) > 0 else 0)
        scale = kwargs.get('scale', args.pop() if len(args) > 0 else 1)
        loc, scale, *args = _promote_args(self.logpdf, loc, scale, *args)
        x = (x - loc) / scale
        return self._logpdf(x) - np.log(scale)
