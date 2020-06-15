import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
from jax import ops
from jax.experimental import loops


def scan(f, s, as_):
    bs = []
    for a in as_:
        s, b = f(s, a)
        bs.append(b)
    return s, np.concatenate(bs)

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

    def body_fn(s, i):
        y, rng_key, lyapunov_loss = s
        t = i * step_size
        k1, rng_key = jax.random.split(rng_key)
        noise = jax.random.normal(k1, np.shape(y)) * lyapunov_scale
        ly_prev = constrain_fn('y', unconstrain_fn('y', y) + noise)
        ly = step(t, ly_prev, **nkwargs)
        y_und = step(t, y, **kwargs)
        y = (1 - dampening_rate) * jax.lax.stop_gradient(y_und) + dampening_rate * y_und
        ll = np.sum(np.abs(y - ly)) / np.sum(np.abs(noise))
        lyapunov_loss = lyapunov_loss + np.maximum(0.0, np.log(ll))
        return ((y, rng_key, lyapunov_loss), y)
    
    s = (y0, rng_key, np.array(0.))
    (_, _, lyapunov_loss), res = jax.lax.scan(body_fn, s, np.arange(num_steps))
    return res, lyapunov_loss

def runge_kutta_4(f: Callable[[float, np.ndarray], np.ndarray], step_size=0.1, num_steps=10, dampening_rate=0.9, lyapunov_scale=1e-3,
                  clip=lambda x: x, unconstrain_fn=lambda k, v: v, constrain_fn=lambda k, v: v):
    return jax.partial(_runge_kutta_4, f, step_size, num_steps, dampening_rate,
                       lyapunov_scale, clip, unconstrain_fn, constrain_fn)
