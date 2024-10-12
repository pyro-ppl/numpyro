# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import numpy as np

import jax

ARRAY_TYPE = Union[jax.Array, np.ndarray]  # jax.Array covers tracers
