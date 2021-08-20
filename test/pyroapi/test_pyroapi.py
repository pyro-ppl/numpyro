# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpyro.infer import (
    RenyiELBO,
    Trace_ELBO,
    TraceMeanField_ELBO,
)
from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401
import pytest

cont_inf_only_cls_names = [
    RenyiELBO.__name__,
    Trace_ELBO.__name__,
    TraceMeanField_ELBO.__name__,
]

pytestmark = pytest.mark.filterwarnings(
    "ignore::numpyro.compat.util.UnsupportedAPIWarning",
    *(
        f"ignore:Currently, SVI with {s_name} loss does not support models with discrete latent variables"
        for s_name in cont_inf_only_cls_names
    ),
)


@pytest.fixture
def backend():
    with pyro_backend("numpy"):
        yield
