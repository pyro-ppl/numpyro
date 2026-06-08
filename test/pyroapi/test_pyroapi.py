# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyroapi import pyro_backend
from pyroapi.dispatch import distributions as dist, infer, ops, pyro
from pyroapi.tests import *  # noqa F401
from pyroapi.tests.test_svi import assert_ok
import pytest

from numpyro.infer import RenyiELBO, Trace_ELBO, TraceMeanField_ELBO

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


# pyroapi's test_constraints inits the simplex param `q` with an unnormalized
# exp(randn(3)); use a valid simplex so it passes with validation enabled.
@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
def test_constraints(backend, jit):
    data = ops.tensor(0.5)

    def model():
        locs = pyro.param("locs", ops.randn(3), constraint=dist.constraints.real)
        scales = pyro.param(
            "scales", ops.exp(ops.randn(3)), constraint=dist.constraints.positive
        )
        p = ops.tensor([0.5, 0.3, 0.2])
        x = pyro.sample("x", dist.Categorical(p))
        pyro.sample("obs", dist.Normal(locs[x], scales[x]), obs=data)

    def guide():
        q = pyro.param(
            "q", ops.tensor([0.4, 0.3, 0.3]), constraint=dist.constraints.simplex
        )
        pyro.sample("x", dist.Categorical(q))

    Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
    elbo = Elbo(ignore_jit_warnings=True)
    assert_ok(model, guide, elbo)
