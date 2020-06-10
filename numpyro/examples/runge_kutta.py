import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
from jax import ops
from jax.experimental import loops

class DummyScope(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def range(self, *args):
        return range(*args)

@functools.partial(jax.jit, static_argnums=(0,1,2,3,4,5,6,7))
def _runge_kutta_4(f: Callable[[float, np.ndarray], np.ndarray], 
                   step_size,
                   num_steps,
                   dampening_rate,
                   lyapunov_scale,
                   clip,
                   unconstrain_fn,
                   constrain_fn,
                   rng_key,
                   y0: np.ndarray,
                   **kwargs):
    def step(t, y, **kwargs):
        k1 = clip(step_size * f(t, y, **kwargs))
        k2 = clip(step_size * f(t + step_size / 2, y + k1 / 2, **kwargs))
        k3 = clip(step_size * f(t + step_size / 2, y + k2 / 2, **kwargs))
        k4 = clip(step_size * f(t + step_size, y + k3, **kwargs))
        dy = clip((k1 + 2 * k2 + 2 * k3 + k4) / 6)
        return y + dy

    k1, rng_key = jax.random.split(rng_key)
    nkwargs = {}
    for kwa, kwv in kwargs.items():
        k1, rng_key = jax.random.split(rng_key)
        kwn = jax.random.normal(k1, np.shape(kwv)) * lyapunov_scale
        nkwargs[kwa] = constrain_fn(kwa, unconstrain_fn(kwa, kwv) + kwn)

    with loops.Scope() as s:
        s.res = np.empty((num_steps, *y0.shape))
        s.y = y0
        s.lyapunov_loss = np.array(0.)
        for i in s.range(num_steps):
            t = i * step_size
            k1, rng_key = jax.random.split(rng_key)
            noise = jax.random.normal(k1, np.shape(y0)) * lyapunov_scale
            ly_prev = constrain_fn('y', unconstrain_fn('y', s.y) + noise)
            ly_und = step(t, ly_prev, **nkwargs)
            ly = (1 - dampening_rate) * jax.lax.stop_gradient(ly_und) + dampening_rate * ly_und 
            y_und = step(t, s.y, **kwargs)
            s.y = (1 - dampening_rate) * jax.lax.stop_gradient(y_und) + dampening_rate * y_und
            s.res = ops.index_update(s.res, i, s.y)
            ll = np.sum(np.abs(s.y - ly)) / np.sum(np.abs(noise))
            s.lyapunov_loss = s.lyapunov_loss + np.maximum(0.0, np.log(ll))
        return s.res, s.lyapunov_loss

def runge_kutta_4(f: Callable[[float, np.ndarray], np.ndarray], step_size=0.1, num_steps=10, dampening_rate=0.9, lyapunov_scale=1e-3,
                  clip=lambda x: x, unconstrain_fn=lambda k, v: v, constrain_fn=lambda k, v: v):
    return jax.partial(_runge_kutta_4, f, step_size, num_steps, dampening_rate,
                       lyapunov_scale, clip, unconstrain_fn, constrain_fn)
