import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
from jax import ops
from jax.experimental import loops

@functools.partial(jax.jit, static_argnums=(0,1,2,3,4,5))
def runge_kutta_4(f: Callable[[float, np.ndarray], np.ndarray], 
                  step_size,
                  num_steps,
                  dampening_rate,
                  lyapunov_scale,
                  clip,
                  rng_key,
                  y0: np.ndarray,
                  **kwargs):
    def step(t, y):
        k1 = clip(step_size * f(t, y, **kwargs))
        k2 = clip(step_size * f(t + step_size / 2, y + k1 / 2, **kwargs))
        k3 = clip(step_size * f(t + step_size / 2, y + k2 / 2, **kwargs))
        k4 = clip(step_size * f(t + step_size, y + k3, **kwargs))
        dy = clip((k1 + 2 * k2 + 2 * k3 + k4) / 6)
        return y + dy

    with loops.Scope() as s:
        s.res = np.empty((num_steps, *y0.shape))
        s.y = y0
        s.lyapunov_loss = np.array(0.)
        for i in s.range(num_steps):
            t = i * step_size
            noise = jax.random.normal(rng_key, np.shape(y0)) * lyapunov_scale
            ly_und = step(t, s.y + noise)
            ly = (1 - dampening_rate) * jax.lax.stop_gradient(ly_und) + dampening_rate * ly_und 
            y_und = step(t, s.y)
            s.y = (1 - dampening_rate) * jax.lax.stop_gradient(y_und) + dampening_rate * y_und
            s.res = ops.index_update(s.res, i, s.y)
            ll = np.sum(np.abs(s.y - ly)) / np.sum(np.abs(noise))
            s.lyapunov_loss = s.lyapunov_loss + np.maximum(0.0, np.log(ll))
        return s.res, s.lyapunov_loss
