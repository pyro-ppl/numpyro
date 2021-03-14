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


def nested_plates():
    N_plate = numpyro.plate('N', 10, dim=-2)
    M_plate = numpyro.plate('M', 5, dim=-1)
    with N_plate:
        x = numpyro.sample('x', dist.Normal(0, 1))
        with M_plate:
            y = numpyro.sample('y', dist.Normal(0, 1))
    with M_plate:
        z = numpyro.sample('z', dist.Normal(0, 1))


def discrete_to_continuous(probs, locs):
    c = numpyro.sample('c', dist.Categorical(probs))
    numpyro.sample('x', dist.Normal(locs[c], 0.5))


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
    (nested_plates, dict(), {
        'plate_groups': {'N': ['x', 'y'], 'M': ['y'], 'M__CLONE': ['z'], None: []},
        'plate_data': {
            'N': {'parent': None},
            'M': {'parent': 'N'},
            'M__CLONE': {'parent': None},
        },
        'node_data': {
            'x': {'is_observed': False},
            'y': {'is_observed': False},
            'z': {'is_observed': False},
        },
        'edge_list': [],
    }),
    (
        discrete_to_continuous,
        dict(probs=jnp.array([0.15, 0.3, 0.3, 0.25]), locs=jnp.array([-2, 0, 2, 4])),
        {
            'plate_groups': {None: ['c', 'x']},
            'plate_data': {},
            'node_data': {
                'c': {'is_observed': False},
                'x': {'is_observed': False},
            },
            'edge_list': [('c', 'x')],
        }
    ),
])
def test_model_transformation(test_model, model_kwargs, expected_graph_spec):
    relations = get_model_relations(test_model, **model_kwargs)
    graph_spec = generate_graph_specification(relations)

    assert graph_spec == expected_graph_spec
