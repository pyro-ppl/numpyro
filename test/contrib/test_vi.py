from collections import namedtuple
from math import floor
import tempfile

import numpy as np
from numpy.ma.testutils import assert_close
import pytest

from jax import random
import jax.numpy as jnp

from numpyro.contrib.callbacks import (
    Checkpoint,
    EarlyStopping,
    History,
    Progbar,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from numpyro.contrib.einstein.stein import VIState
from numpyro.optim import Adam


########################################
# Variational Interface
########################################
@pytest.mark.parametrize("callback", [])
def test_callback(callback):
    pass


def test_checkpoint():
    params = {"a": jnp.array([1, 2, 3]), "b": jnp.array([1.2, 1.2, 0.1])}
    state = VIState(Adam(1.0).init(params), random.PRNGKey(0))
    with tempfile.NamedTemporaryFile() as tmp_file:
        Checkpoint(tmp_file.name)._checkpoint("", 0.0, 0.0, state)
        tmp_file.seek(0)
        optim_state, rng_key, loss = Checkpoint.load(tmp_file.name)
        loaded_state = VIState(optim_state, rng_key)
        assert state.optim_state[0] == loaded_state.optim_state[0]
        assert (
            state.optim_state[1].subtree_defs
            == loaded_state.optim_state[1].subtree_defs
        )
        assert state.optim_state[1].tree_def == loaded_state.optim_state[1].tree_def
        for exp, act in zip(
            state.optim_state[1].packed_state, loaded_state.optim_state[1].packed_state
        ):
            assert all(all(e == a) for e, a in zip(exp, act))
        assert all(state.rng_key == loaded_state.rng_key)


@pytest.mark.parametrize("smoothing", ["none", "exp", "dexp"])
@pytest.mark.parametrize("patience", [0, 1, 2, 100])
def test_early_stopping(smoothing, patience):
    es = EarlyStopping(
        patience=patience, smoothing=smoothing, loss_mode="training", min_delta=0.5
    )
    num_steps = 101
    curr_loss = 10.0

    train_info = {
        "num_steps": num_steps,
        "state": {},
        "loss": curr_loss,
        "model_args": (),
        "model_kwargs": {},
    }
    es.on_train_begin(train_info)
    assert es.best_loss == es.curr_loss == train_info["loss"]
    es.loss_mode = "validation"
    es.on_train_begin({**train_info, **{"loss": 5.0}})
    assert es.best_loss == es.curr_loss == train_info["loss"]
    es.loss_mode = "training"

    key = random.PRNGKey(0)

    for num_steps in range(num_steps):
        try:
            key, noise_key = random.split(key)
            train_info["loss"] = train_info["loss"] - 0.1 * random.normal(key, ())
            es.on_train_end(train_info)
        except StopIteration:
            assert num_steps == patience
            break


def test_history():
    num_steps = 100

    train_info = lambda loss: {
        "num_steps": 100,
        "state": {},
        "loss": loss,
        "model_args": (),
        "model_kwargs": {},
    }
    h = History()
    key = random.PRNGKey(0)

    key, tkey, vkey = random.split(key, 3)
    tlosses = 10.0 + jnp.cumsum(random.normal(vkey, (num_steps,)))
    vlosses = -3.4 + jnp.cumsum(random.normal(tkey, (num_steps,)))
    for i in range(num_steps):
        h.on_train_step_end(i, train_info(tlosses[i]))
        h.on_validation_end(i, train_info(vlosses[i]))

    assert (jnp.array(h.training_history) == tlosses).all()
    assert (jnp.array(h.validation_history) == vlosses).all()


def test_progbar(capsys):
    num_steps = 100

    train_info = lambda loss: {
        "num_steps": 100,
        "state": {},
        "loss": loss,
        "model_args": (),
        "model_kwargs": {},
    }
    vi = namedtuple("vi", ["name"])

    pb = Progbar()
    pb.vi = vi("vi")
    pb.on_train_begin(train_info(10.0))
    losses = 10.0 + np.cumsum(100 * np.random.normal(size=(num_steps,)))
    for i in pb.progbar:
        pb.on_train_step_end(i, train_info(losses[i]))
        exp_str = " ".join(("vi", format(float(losses[i]), ".5"))) + ": "
        assert pb.progbar.desc == exp_str
        capsys.readouterr()
    assert capsys.readouterr().err.strip().startswith(exp_str)


def test_terminate_on_nan():
    train_info = lambda loss: {
        "num_steps": 100,
        "state": {},
        "loss": loss,
        "model_args": (),
        "model_kwargs": {},
    }

    try:
        TerminateOnNaN().on_train_step_end(
            0, train_info(jnp.array([1, 2, 3, 4.523, 43214]))
        )
    except StopIteration:
        pytest.fail()
    try:
        TerminateOnNaN().on_train_step_end(
            0, train_info(jnp.array([1, 2, jnp.nan, 4.523, 43214]))
        )
        pytest.fail()
    except StopIteration:
        pass


def test_reduce_lr_on_plateau():
    schedule = ReduceLROnPlateau(
        initial_lr=1.0, patience=2, factor=1e-1, min_lr=1e-5, frequency="step"
    )
    schedule.best_loss = 10.0
    for i in range(100):
        schedule._reduce_lr_on_plateau(11.0)
        exp_lr = max(10 ** (-floor(i // 2)), 1e-5)
        assert_close(schedule.lr, exp_lr)
