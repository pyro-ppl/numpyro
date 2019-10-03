import numpyro.compat
import numpyro.diagnostics
import numpyro.distributions
import numpyro.infer
import numpyro.infer_util
import numpyro.patch  # noqa: F401
from numpyro.primitives import module, param, plate, sample  # noqa: F401
import numpyro.util
from numpyro.version import __version__  # noqa: F401

util.set_platform('cpu')


__all__ = [
    '__version__',
    'compat',
    'diagnostics',
    'distributions',
    'infer',
    'infer_util',
    'module',
    'param',
    'plate',
    'sample',
    'util',
]
