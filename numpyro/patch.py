import jax


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


@patch_dependency('jax.api._check_inexact_input_vjp', jax)
def _check_inexact_input_vjp(x):
    # allow inexact inputs (int/bool) to reverse-mode differentiation
    pass
