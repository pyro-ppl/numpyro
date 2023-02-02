# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from flax.training.train_state import TrainState
import jax
from jax import lax, numpy as jnp, random
import optax

from models import cross_entropy_loss  # isort:skip


def create_train_state(model, x, learning_rate_fn):
    params = model.init(random.PRNGKey(0), x)
    tx = optax.adam(learning_rate_fn)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def train_step(state, x_batched, y_batched):
    def loss_fn(params):
        y_pred = state.apply_fn(params, x_batched)
        loss = cross_entropy_loss(y_pred, y_batched)
        return jnp.sum(loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def train_epoch(state, train_fetch, num_train, train_idx, epoch_rng):
    def _fn(i, val):
        state, loss_sum = val
        x_batched, y_batched = train_fetch(i, train_idx)
        state, loss = train_step(state, x_batched, y_batched)
        loss_sum += loss
        return state, loss_sum

    return lax.fori_loop(0, num_train, _fn, (state, 0.0))


def eval_epoch(state, test_fetch, num_test, test_idx, epoch_rng):
    def _fn(i, loss_sum):
        x_batched, y_batched = test_fetch(i, test_idx)
        y_pred = state.apply_fn(state.params, x_batched)
        loss = cross_entropy_loss(y_pred, y_batched)
        loss_sum += jnp.sum(loss)
        return loss_sum

    loss = lax.fori_loop(0, num_test, _fn, 0.0)
    loss = loss / num_test
    return loss


def train_baseline(
    model,
    num_train,
    train_idx,
    train_fetch,
    num_test,
    test_idx,
    test_fetch,
    n_epochs=100,
):
    state = create_train_state(model, train_fetch(0, train_idx)[0], 0.003)

    rng = random.PRNGKey(0)
    best_val_loss = jnp.inf
    best_state = state
    for i in range(n_epochs):
        epoch_rng = jax.random.fold_in(rng, i)
        state, train_loss = train_epoch(
            state, train_fetch, num_train, train_idx, epoch_rng
        )
        val_loss = eval_epoch(state, test_fetch, num_test, test_idx, epoch_rng)
        print(f"Epoch loss - train loss: {train_loss}, validation loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state

    return best_state.params
