# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore::numpyro.compat.util.UnsupportedAPIWarning"
)


@pytest.fixture
def backend():
    with pyro_backend("numpy"):
        yield
