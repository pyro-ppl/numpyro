import tempfile

import jax.numpy as jnp
import pytest
from jax import random

from numpyro.contrib.callbacks import (
    Checkpoint)
from numpyro.contrib.einstein.stein import VIState
from numpyro.optim import Adam


########################################
# Variational Interface
########################################
@pytest.mark.parametrize("callback", [])
def test_callback(callback):
    pass


def test_checkpoint():
    params = {"a": jnp.array([1, 2, 3]), "b": jnp.array([1.2, 1.2, .1])}
    state = VIState(Adam(1.).init(params), random.PRNGKey(0))
    with tempfile.NamedTemporaryFile() as tmp_file:
        Checkpoint(tmp_file.name)._checkpoint("", 0.0, 0.0, state)
        tmp_file.seek(0)
        optim_state, rng_key, loss = Checkpoint.load(tmp_file.name)
        loaded_state = VIState(optim_state, rng_key)
        assert state.optim_state[0] == loaded_state.optim_state[0]
        assert state.optim_state[1].subtree_defs == loaded_state.optim_state[1].subtree_defs
        assert state.optim_state[1].tree_def == loaded_state.optim_state[1].tree_def
        for exp, act in zip(state.optim_state[1].packed_state, loaded_state.optim_state[1].packed_state):
            assert all(all(e == a) for e, a in zip(exp, act))
        assert all(state.rng_key == loaded_state.rng_key)
