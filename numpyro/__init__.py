import numpyro.distributions
import numpyro.compat
import numpyro.infer
import numpyro.patch  # noqa: F401
from numpyro.primitives import module, param, plate, sample  # noqa: F401
import numpyro.util as util
from numpyro.version import __version__  # noqa: F401

util.set_platform('cpu')


__all__ = [
    '__version__',
    'compat',
    'distributions',
    'infer',
    'module',
    'param',
    'plate',
    'sample',
    'util',
]
