import numpyro.patch  # noqa: F401
from numpyro import compat, diagnostics, distributions, handlers, infer, infer_util
from numpyro.primitives import module, param, plate, sample
from numpyro.util import set_host_device_count, set_platform
from numpyro.version import __version__

set_platform('cpu')


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
    'set_host_device_count',
    'set_platform',
]
