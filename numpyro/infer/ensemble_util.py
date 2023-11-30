import warnings

import numpy as np

from jax import lax, vmap
import jax.numpy as jnp

from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_zip, unzip2, HashablePartial

zip = safe_zip


def _get_nondiagonal_pairs(n):
    """
    From https://github.com/dfm/emcee/blob/main/src/emcee/moves/de.py:
    
    Get the indices of a square matrix with size n, excluding the diagonal.
    """
    
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack([np.concatenate([rows, cols]), 
                             np.concatenate([cols, rows])])

    return jnp.asarray(pairs)


def batch_ravel_pytree(pytree):
    """Ravel (flatten) a pytree of arrays with leading batch dimension down to a (batch_size, 1D) array.   
    Args:
      pytree: a pytree of arrays and scalars to ravel.  
    Returns:
      A pair where the first element is a (batch_size, 1D) array representing the flattened and
      concatenated leaf values, with dtype determined by promoting the dtypes of
      leaf values, and the second element is a callable for unflattening a (batch_size, 1D)
      vector of the same length back to a pytree of of the same structure as the
      input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
      a convention a 1D empty array of dtype float32 is returned in the first
      component of the output.  
    For details on dtype promotion, see
    https://jax.readthedocs.io/en/latest/type_promotion.html.   
    """
    
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, HashablePartial(unravel_pytree, treedef, unravel_list)

def unravel_pytree(treedef, unravel_list, flat):
    return tree_unflatten(treedef, unravel_list(flat))

@vmap
def vmapped_ravel(a):
    return jnp.ravel(a)

def _ravel_list(lst):
    if not lst: return jnp.array([], jnp.float32), lambda _: []
    from_dtypes = tuple(dtypes.dtype(l) for l in lst)
    to_dtype = dtypes.result_type(*from_dtypes)
    
    # here 1 is n_leading_batch_dimensions    
    sizes, shapes = unzip2((np.prod(jnp.shape(x)[1:]), jnp.shape(x)[1:]) for x in lst)
    indices = tuple(np.cumsum(sizes))
    
    if all(dt == to_dtype for dt in from_dtypes):
        # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
        # See https://github.com/google/jax/issues/7809.
        del from_dtypes, to_dtype
        
        # axis = n_leading_batch_dimensions
        # vmap n_leading_batch_dimensions times
        raveled = jnp.concatenate([vmapped_ravel(e) for e in lst], axis=1)
        return raveled, HashablePartial(_unravel_list_single_dtype, indices, shapes)
    
    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
    raveled = jnp.concatenate([vmapped_ravel(e) for e in lst])
    unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
    return raveled, unrav
    
    
def _unravel_list_single_dtype(indices, shapes, arr):
    # axis is n_leading_batch_dimensions
    chunks = jnp.split(arr, indices[:-1], axis=1)

    # the number of -1s is the number of leading batch dimensions
    return [chunk.reshape((-1, *shape)) for chunk, shape in zip(chunks, shapes)]


def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
    arr_dtype = dtypes.dtype(arr)
    if arr_dtype != to_dtype:
      raise TypeError(f"unravel function given array of dtype {arr_dtype}, "
                      f"but expected dtype {to_dtype}")
    
    # axis is n_leading_batch_dimensions
    chunks = jnp.split(arr, indices[:-1], axis=1)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
      # the number of -1s is the number of leading batch dimensions
      return [lax.convert_element_type(chunk.reshape((-1, *shape)), dtype)
              for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)]
      
      
      
