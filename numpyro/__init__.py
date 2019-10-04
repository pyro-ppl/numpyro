import numpyro.patch  # noqa: F401
from numpyro import compat, diagnostics, distributions, handlers, infer, infer_util, util
from numpyro.primitives import module, param, plate, sample
from numpyro.version import __version__

util.set_platform('cpu')


__all__ = [
    '__version__',
    'compat',
    'diagnostics',
    'distributions',
    'handlers',
    'infer',
    'infer_util',
    'module',
    'param',
    'plate',
    'sample',
    'util',
]
