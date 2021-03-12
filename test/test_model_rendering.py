import pytest

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.util import get_model_relations, generate_graph_specification


def simple(data):
    x = numpyro.sample('x', dist.Normal(0, 1))
    sd = numpyro.sample('sd', dist.LogNormal(x, 1))
    with numpyro.plate('N', len(data)):
        numpyro.sample('obs', dist.Normal(x, sd), obs=data)


@pytest.mark.parametrize('test_model,model_kwargs,expected_graph_spec', [
    (simple, dict(data=jnp.ones(10)), {
        'plate_groups': {'N': ['obs'], None: ['x', 'sd']},
        'plate_data': {'N': {'parent': None}},
        'node_data': {
            'x': {'is_observed': False},
            'sd': {'is_observed': False},
            'obs': {'is_observed': True},
        },
        'edge_list': [('x', 'sd'), ('x', 'obs'), ('sd', 'obs')],
    }),
])
def test_model_transformation(test_model, model_kwargs, expected_graph_spec):
    relations = get_model_relations(test_model, **model_kwargs)
    graph_spec = generate_graph_specification(relations)

    assert graph_spec == expected_graph_spec
