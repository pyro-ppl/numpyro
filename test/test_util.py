from numpy.testing import assert_allclose
import pytest

from jax import lax
from jax.dtypes import canonicalize_dtype
import jax.numpy as np
from jax.test_util import check_eq
from jax.tree_util import tree_flatten, tree_multimap

from numpyro.util import fori_collect, ravel_pytree


def test_fori_collect():
    def f(x):
        return {'i': x['i'] + x['j'], 'j': x['i'] - x['j']}

    a = {'i': np.array([0.]), 'j': np.array([1.])}
    expected_tree = {'i': np.array([[0.], [2.]])}
    actual_tree = fori_collect(1, 3, f, a, transform=lambda a: {'i': a['i']})
    check_eq(actual_tree, expected_tree)


@pytest.mark.parametrize('pytree', [
    {'a': np.array(0.), 'b': np.array([[1., 2.], [3., 4.]])},
    {'a': np.array(0), 'b': np.array([[1, 2], [3, 4]])},
    {'a': np.array(0), 'b': np.array([[1., 2.], [3., 4.]])},
    {'a': 0., 'b': np.array([[1., 2.], [3., 4.]])},
    {'a': False, 'b': np.array([[1., 2.], [3., 4.]])},
    [False, True, 0., np.array([[1., 2.], [3., 4.]])],
])
def test_ravel_pytree(pytree):
    flat, unravel_fn = ravel_pytree(pytree)
    unravel = unravel_fn(flat)
    tree_flatten(tree_multimap(lambda x, y: assert_allclose(x, y), unravel, pytree))
    assert all(tree_flatten(tree_multimap(lambda x, y:
                                          canonicalize_dtype(lax.dtype(x)) == canonicalize_dtype(lax.dtype(y)),
                                          unravel, pytree))[0])
