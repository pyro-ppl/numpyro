# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpy.testing import assert_allclose


@pytest.mark.parametrize('intervene,observe,flip', [
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, True, True),
])
def test_counterfactual_query(intervene, observe, flip):
    # x -> y -> z -> w

    sites = ["x", "y", "z", "w"]
    observations = {"x": 1., "y": None, "z": 1., "w": 1.}
    interventions = {"x": None, "y": 0., "z": 2., "w": 1.}

    def model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        y = numpyro.sample("y", dist.Normal(x, 1))
        z = numpyro.sample("z", dist.Normal(y, 1))
        w = numpyro.sample("w", dist.Normal(z, 1))
        return dict(x=x, y=y, z=z, w=w)

    if not flip:
        if intervene:
            model = handlers.do(model, data=interventions)
        if observe:
            model = handlers.condition(model, data=observations)
    elif flip and intervene and observe:
        model = handlers.do(
            handlers.condition(model, data=observations),
            data=interventions)

    tr = handlers.trace(model).get_trace()
    actual_values = tr.nodes["_RETURN"]["value"]
    for name in sites:
        # case 1: purely observational query like handlers.condition
        if not intervene and observe:
            if observations[name] is not None:
                assert tr.nodes[name]['is_observed']
                assert_allclose(observations[name], actual_values[name])
                assert_allclose(observations[name], tr.nodes[name]['value'])
            if interventions[name] != observations[name]:
                assert_not_equal(interventions[name], actual_values[name])
        # case 2: purely interventional query like old handlers.do
        elif intervene and not observe:
            assert not tr.nodes[name]['is_observed']
            if interventions[name] is not None:
                assert_allclose(interventions[name], actual_values[name])
            assert_not_equal(observations[name], tr.nodes[name]['value'])
            assert_not_equal(interventions[name], tr.nodes[name]['value'])
        # case 3: counterfactual query mixing intervention and observation
        elif intervene and observe:
            if observations[name] is not None:
                assert tr.nodes[name]['is_observed']
                assert_allclose(observations[name], tr.nodes[name]['value'])
            if interventions[name] is not None:
                assert_allclose(interventions[name], actual_values[name])
            if interventions[name] != observations[name]:
                assert_not_equal(interventions[name], tr.nodes[name]['value'])


def assert_not_equal(x, y, prec=1e-5, msg=''):
    try:
        assert_allclose(x, y, atol=prec, msg=msg)
    except AssertionError:
        return
    raise AssertionError("{} \nValues are equal: x={}, y={}, prec={}".format(msg, x, y, prec))
