import pytest
from numpy.testing import assert_allclose

import jax.numpy as np
from jax import lax
from jax.tree_util import tree_map

from numpyro.util import control_flow_prims_disabled, laxtuple, optional, tscan


@pytest.mark.parametrize('prims_disabled', [True, False])
def test_tscan(prims_disabled):
    def f(tree, yz):
        y, z = yz
        return tree_map(lambda x: (x + y) * z, tree)

    Tree = laxtuple("Tree", ["x", "y", "z"])
    a = Tree(np.array([1., 2.]),
             np.array(3., dtype=np.float32),
             np.array(4., dtype=np.float32))
    bs = (np.array([1., 2., 3., 4.]),
          np.array([4., 3., 2., 1.]))

    expected_tree = lax.scan(f, a, bs)
    with optional(prims_disabled, control_flow_prims_disabled()):
        actual_tree = tscan(f, a, bs, fields=(0, 2))
    assert_allclose(actual_tree.x, expected_tree.x)
    assert_allclose(actual_tree.z, expected_tree.z)
    assert actual_tree.y is None
