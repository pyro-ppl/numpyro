from numpy.testing import assert_allclose

import jax.numpy as np
from jax import lax
from jax.test_util import check_eq
from jax.tree_util import tree_map

from numpyro.util import control_flow_prims_disabled, fori_collect, laxtuple, scan


def test_fori_collect():
    def f(x):
        return {'i': x['i'] + x['j'], 'j': x['i'] - x['j']}

    a = {'i': np.array([0.]), 'j': np.array([1.])}
    scan_tree = lax.scan(lambda x, y: f(x), a, np.arange(3))
    expected_tree = {'i': scan_tree['i']}
    actual_tree = fori_collect(3, f, a, transform=lambda a: {'i': a['i']})
    check_eq(actual_tree, expected_tree)


def test_scan_prims_disabled():
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
    with control_flow_prims_disabled():
        actual_tree = scan(f, a, bs)
    assert_allclose(actual_tree.x, expected_tree.x)
    assert_allclose(actual_tree.y, expected_tree.y)
    assert_allclose(actual_tree.z, expected_tree.z)
