import pytest
from numpy.testing import assert_allclose

import jax.numpy as np
from jax import lax
from jax.test_util import check_eq
from jax.tree_util import tree_map

from numpyro.util import control_flow_prims_disabled, fori_append, fori_collect, laxtuple, optional, scan, tscan


@pytest.mark.parametrize('prims_disabled', [True, False])
def test_tscan(prims_disabled):
    def f(tree, yz):
        y, z = yz
        return tree_map(lambda x: (x + y) * z, tree)

    Tree = laxtuple("Tree", ["x", "y", "z"])
    OutTree = laxtuple("OutTree", ["x", "z"])
    a = Tree(np.array([1., 2.]),
             np.array(3., dtype=np.float32),
             np.array(4., dtype=np.float32))
    bs = (np.array([1., 2., 3., 4.]),
          np.array([4., 3., 2., 1.]))

    scan_tree = lax.scan(f, a, bs)
    expected_tree = OutTree(scan_tree.x, scan_tree.z)
    with optional(prims_disabled, control_flow_prims_disabled()):
        actual_tree = tscan(f, a, bs, transform=lambda a: OutTree(a.x, a.z))
    check_eq(actual_tree, expected_tree)


@pytest.mark.parametrize('prims_disabled', [True, False])
def test_tscan_dict(prims_disabled):
    def f(x, y):
        return {'i': x['i'] + y['i'], 'j': x['j'] - y['j']}

    a = {'i': np.array([0.]), 'j': np.array([1.])}
    bs = {'i': np.array([1., 2., 3.]), 'j': np.array([2., 3., 6.])}
    scan_tree = lax.scan(f, a, bs)
    expected_tree = {'i': scan_tree['i']}
    with optional(prims_disabled, control_flow_prims_disabled()):
        actual_tree = tscan(f, a, bs, transform=lambda a: {'i': a['i']})
    check_eq(actual_tree, expected_tree)


def test_fori_append():
    def f(x):
        return {'i': x['i'] + x['j'], 'j': x['i'] - x['j']}

    a = {'i': np.array([0.]), 'j': np.array([1.])}
    scan_tree = lax.scan(lambda x, y: f(x), a, np.arange(3))
    expected_tree = {'i': scan_tree['i']}
    actual_tree = fori_append(f, a, 3, transform=lambda a: {'i': a['i']})
    check_eq(actual_tree, expected_tree)


def test_fori_collect():
    def f(x):
        return {'i': x['i'] + x['j'], 'j': x['i'] - x['j']}

    a = {'i': np.array([0.]), 'j': np.array([1.])}
    scan_tree = lax.scan(lambda x, y: f(x), a, np.arange(3))
    expected_tree = {'i': scan_tree['i']}
    actual_tree = fori_collect(3, f, a, transform=lambda a: {'i': a['i']})
    check_eq(actual_tree, expected_tree)


@pytest.mark.parametrize('prims_disabled', [True, False])
def test_scan_scalar(prims_disabled):
    def f(x, y):
        return {'i': x['i'] + y['i'], 'j': x['j'] - y['j']}

    a = {'i': np.array(0.), 'j': np.array(1.)}
    bs = {'i': np.array([1., 2., 3.]), 'j': np.array([2., 3., 6.])}
    expected_tree = {'i': np.array([1., 3., 6.])}
    with optional(prims_disabled, control_flow_prims_disabled()):
        actual_tree = tscan(f, a, bs, transform=lambda a: {'i': a['i']})
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
