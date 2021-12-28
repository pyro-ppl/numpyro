import pytest
from jax import random

import numpyro
from numpyro import handlers
from numpyro.contrib.einstein.reinit_guide import WrappedGuide
from numpyro.distributions import Normal, Bernoulli
from numpyro.distributions.constraints import _Real
from numpyro.infer import init_to_feasible, init_to_median, init_to_sample, init_to_uniform
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoLaplaceApproximation, AutoLowRankMultivariateNormal, \
    AutoNormal, AutoDelta, AutoDiagonalNormal


@pytest.mark.parametrize(
    "auto_class",
    [AutoMultivariateNormal,
     AutoLaplaceApproximation,
     AutoLowRankMultivariateNormal,
     AutoNormal,
     AutoDelta,
     AutoDiagonalNormal,
     ]
)
@pytest.mark.parametrize(
    "init_loc_fn",
    [
        init_to_feasible,
        init_to_median,
        init_to_sample,
        init_to_uniform,
    ],
)
@pytest.mark.parametrize(
    "num_particles", [1, 2, 10])
def test_auto_guide(auto_class, init_loc_fn, num_particles):
    def model(obs=None):
        a = numpyro.sample('a', Normal(0, 1))
        numpyro.sample('obs', Bernoulli(logits=a), obs=obs)

    obs = Bernoulli(.5).sample(random.PRNGKey(0), (10,))
    auto_guide = auto_class(model, init_loc_fn=init_loc_fn())
    with handlers.seed(rng_seed=0), handlers.trace() as auto_guide_tr:
        auto_guide(obs)

    # Corresponds to current procedure in `SteinVI.init`
    wrapped_guide = WrappedGuide(auto_guide, init_strategy=init_loc_fn())
    rng_keys = random.split(random.PRNGKey(1), num_particles)
    wrapped_guide.find_params(rng_keys, obs)
    init_params = wrapped_guide.init_params()

    for name, (init_value, constraint) in init_params.items():
        assert name in auto_guide_tr
        auto_param = auto_guide_tr[name]
        assert init_value.shape == (num_particles, *auto_param['value'].shape)

        if 'constraint' in auto_param['kwargs']:
            assert constraint == auto_param['kwargs']['constraint']
        else:
            constraint == _Real()
