import numpy as onp

import jax.numpy as np
from jax.numpy import *  # noqa: F401, F403

tensor = array  # noqa: F405

randn = onp.random.randn


# Provide wrappers to initialize ones/zeros using the pytorch convention
# of using *sizes. e.g. ops.ones(2, 3) as well as ops.ones((2, 3)) can
# be used to initialize an array of ones with shape (2, 3).

def ones(*sizes, **kwargs):
    if len(sizes) == 0:
        raise ValueError('Positional `size` argument not provided.')
    elif len(sizes) == 1:
        if isinstance(sizes[0], (tuple, list)):
            sizes = sizes[0]
    if not all([isinstance(s, int) for s in sizes]):
        raise ValueError('Invalid data type for `size` provided.')
    return np.ones(sizes, **kwargs)


def zeros(*sizes, **kwargs):
    if len(sizes) == 0:
        raise ValueError('Positional `size` argument not provided.')
    elif len(sizes) == 1:
        if isinstance(sizes[0], (tuple, list)):
            sizes = sizes[0]
    if not all([isinstance(s, int) for s in sizes]):
        raise ValueError('Invalid data type for `size` provided.')
    return np.ones(sizes, **kwargs)
