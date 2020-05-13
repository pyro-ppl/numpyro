import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
from jax import ops
from jax.experimental import loops

@functools.partial(jax.jit, static_argnums=(0,2,3))
def runge_kutta_4(f: Callable[[float, np.ndarray], np.ndarray], 
                  y0: np.ndarray,
                  step_size,
                  num_steps):
    with loops.Scope() as s:
        s.res = np.empty((num_steps, *y0.shape))
        s.y = y0
        for i in s.range(num_steps):
            t = i * step_size
            k1 = step_size * f(t, s.y)
            k2 = step_size * f(t + step_size / 2, s.y + k1 / 2)
            k3 = step_size * f(t + step_size / 2, s.y + k2 / 2)
            k4 = step_size * f(t + step_size, s.y + k3)
            s.y = s.y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            s.res = ops.index_update(s.res, i, s.y)
        return s.res
