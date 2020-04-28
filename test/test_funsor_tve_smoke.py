# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import OrderedDict

import pytest
import jax
import jax.numpy as np
import numpy as onp

import funsor
from funsor.einsum import einsum
from funsor.tensor import Tensor
from funsor.testing import assert_close, make_chain_einsum, make_einsum_example, make_hmm_einsum, make_plated_hmm_einsum
from funsor.util import get_backend

funsor.set_backend("jax")


def raw_einsum(operands, equation=None, plates=None, backend='funsor.einsum.numpy_log'):
    funsor_operands = tuple(
        funsor.to_funsor(
            operand, output=funsor.reals(),
            dim_to_name={dim: name for dim, name in zip(range(-np.ndim(operand), 0), inputs)}
        ) for operand, inputs in zip(operands, equation.split("->")[0].split(","))
    )
    return einsum(equation, *funsor_operands, plates=plates, backend=backend).data


OPTIMIZED_EINSUM_EXAMPLES = [
    make_chain_einsum(t) for t in range(2, 50, 10)
] + [
    make_hmm_einsum(t) for t in range(2, 50, 10)
]


PLATED_EINSUM_EXAMPLES = [
    make_plated_hmm_einsum(num_steps, num_obs_plates=b, num_hidden_plates=a)
    for num_steps in range(3, 50, 6)
    for (a, b) in [(0, 1), (0, 2), (0, 0), (1, 1), (1, 2)]
]


@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'funsor.einsum.numpy_log',
])
def test_optimized_plated_einsum_smoke(equation, plates, backend):

    jit_raw_einsum = jax.jit(functools.partial(
        raw_einsum, equation=equation, plates=plates, backend=backend))

    for i in range(3):
        operands = make_einsum_example(equation)[3]
        actual = jit_raw_einsum(operands)
        assert np.ndim(actual) == 0
