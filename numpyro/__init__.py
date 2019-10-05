from numpyro import compat, diagnostics, distributions, handlers, infer, infer_util, optim
import numpyro.patch  # noqa: F401
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
    'optim',
    'param',
    'plate',
    'sample',
    'set_host_device_count',
    'set_platform',
]
