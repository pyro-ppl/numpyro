import pytest

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401

pytestmark = pytest.mark.filterwarnings("ignore::numpyro.compat.util.UnsupportedAPIWarning")


@pytest.yield_fixture
def backend():
    with pyro_backend('numpy'):
        yield
