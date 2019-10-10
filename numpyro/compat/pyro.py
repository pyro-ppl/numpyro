import warnings

from numpyro.compat.util import UnsupportedAPIWarning
from numpyro.primitives import module, param, plate, sample  # noqa: F401


def get_param_store():
    warnings.warn('NumPyro does not have a parameter store.', category=UnsupportedAPIWarning)
    # Return an empty dict for compatibility
    return {}
