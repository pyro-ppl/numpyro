# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import pytest

import jax
import jax.numpy as jnp

import funsor
from funsor.einsum import einsum
from funsor.testing import make_einsum_example, make_hmm_einsum, make_plated_hmm_einsum
from numpyro.contrib.indexing import Vindex

funsor.set_backend("jax")
pytestmark = pytest.mark.skipif('CI' in os.environ, reason="those tests are slow in CI")


def raw_einsum(operands, equation=None, plates=None, backend='funsor.einsum.numpy_log'):
    funsor_operands = tuple(
        funsor.to_funsor(
            operand, output=funsor.reals(),
            dim_to_name={dim: name for dim, name in zip(range(-jnp.ndim(operand), 0), inputs)}
        ) for operand, inputs in zip(operands, equation.split("->")[0].split(","))
    )
    return einsum(equation, *funsor_operands, plates=plates, backend=backend).data


PLATED_EINSUM_EXAMPLES = [
    make_plated_hmm_einsum(num_steps, num_obs_plates=b, num_hidden_plates=a)
    for num_steps in range(2, 102, 10)
    for (a, b) in [(0, 1), (0, 2), (0, 0), (1, 1), (1, 2)]
]


@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['funsor.einsum.numpy_log'])
@pytest.mark.parametrize('sizes', [(2, 3), (16, 17)])
def test_optimized_plated_einsum_smoke(equation, plates, backend, sizes):

    jit_raw_einsum = jax.jit(jax.value_and_grad(functools.partial(
        raw_einsum, equation=equation, plates=plates, backend=backend)))

    for i in range(2):
        operands = make_einsum_example(equation, sizes=sizes)[3]
        actual, grads = jit_raw_einsum(operands)
        assert jnp.ndim(actual) == 0


def raw_hmm_with_indexing(operands, equation=None, backend='funsor.einsum.numpy_log'):
    indexed_operands = []
    for i, operand in enumerate(operands):
        if jnp.ndim(operand) == 1:
            y = jnp.arange(operand.shape[0]).reshape((operand.shape[0], 1))
            indexed_operands.append(Vindex(operand)[y].squeeze())
        elif jnp.ndim(operand) == 2:
            x = jnp.arange(operand.shape[0]).reshape((operand.shape[0], 1, 1))
            y = jnp.arange(operand.shape[1]).reshape((operand.shape[1], 1))
            indexed_operands.append(Vindex(operand)[x, y].squeeze())
        else:
            raise ValueError
    return raw_einsum(tuple(indexed_operands), equation=equation, plates=None, backend=backend)


@pytest.mark.parametrize('equation', [make_hmm_einsum(t) for t in range(2, 102, 10)])
@pytest.mark.parametrize('backend', ['funsor.einsum.numpy_log'])
@pytest.mark.parametrize('sizes', [(2, 3), (16, 17)])
def test_hmm_einsum_smoke(equation, backend, sizes):

    jit_raw_einsum = jax.jit(jax.value_and_grad(functools.partial(
        raw_hmm_with_indexing, equation=equation, backend=backend)))

    for i in range(2):
        operands = make_einsum_example(equation, sizes=sizes)[3]
        actual, grads = jit_raw_einsum(operands)
        assert jnp.ndim(actual) == 0
