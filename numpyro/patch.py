import jax
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu


def patch_dependency(target, root_module):
    parts = target.split('.')
    assert parts[0] == root_module.__name__
    module = root_module
    for part in parts[1:-1]:
        module = getattr(module, part)
    name = parts[-1]
    old_fn = getattr(module, name)
    old_fn = getattr(old_fn, '_pyro_unpatched', old_fn)  # ensure patching is idempotent

    def decorator(new_fn):
        new_fn.__name__ = name
        new_fn._pyro_unpatched = old_fn
        setattr(module, name, new_fn)
        return new_fn

    return decorator


# TODO: Remove with jax v0.1.26
@patch_dependency('jax.interpreters.partial_eval.trace_unwrapped_to_jaxpr', jax)
def _trace_unwrapped_to_jaxpr(fun, pvals, **kwargs):
    return pe.trace_to_jaxpr(lu.wrap_init(fun, kwargs), pvals)
