# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from flax import traverse_util
import jax
from jax import lax, numpy as jnp, random
import optax

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIState


def flattened_traversal(fn):
    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def create_train_state(
    rng, model, guide, train_fetch, baseline_params, learning_rate_fn
):
    label_fn = flattened_traversal(
        lambda path, _: "adam" if not path[0].startswith("baseline") else "none"
    )
    tx = optax.multi_transform(
        {"adam": optax.adam(learning_rate_fn), "none": optax.set_to_zero()}, label_fn
    )

    svi = SVI(model, guide, tx, loss=Trace_ELBO())
    x_batched, y_batched = train_fetch(0)
    state = svi.init(rng, x=x_batched, y=y_batched)

    svi_params = state.optim_state[1][0]
    svi_params["baseline$params"] = baseline_params.unfreeze()["params"]
    state = SVIState(
        optim_state=(state.optim_state[0], (svi_params, state.optim_state[1][1])),
        mutable_state=state.mutable_state,
        rng_key=state.rng_key,
    )
    return svi, state


def train_epoch(svi, state, train_fetch, num_train, train_idx, epoch_rng):
    def _fn(i, val):
        state, loss_sum = val
        x_batched, y_batched = train_fetch(i, train_idx)
        state, loss = svi.update(state, x=x_batched, y=y_batched)
        loss_sum += loss
        return state, loss_sum

    return lax.fori_loop(0, num_train, _fn, (state, 0.0))


def eval_epoch(svi, state, test_fetch, num_test, test_idx, epoch_rng):
    def _fn(i, loss_sum):
        x_batched, y_batched = test_fetch(i, test_idx)
        loss = svi.evaluate(state, x=x_batched, y=y_batched)
        loss_sum += loss
        return loss_sum

    loss = lax.fori_loop(0, num_test, _fn, 0.0)
    loss = loss / num_test
    return loss


def train_cvae(
    model,
    guide,
    baseline_params,
    num_train,
    train_idx,
    train_fetch,
    num_test,
    test_idx,
    test_fetch,
    n_epochs=100,
):
    svi, state = create_train_state(
        random.PRNGKey(23), model, guide, train_fetch, baseline_params, 0.003
    )

    p1 = baseline_params.unfreeze()["params"]["Dense_0"]["kernel"]
    p2 = state.optim_state[1][0]["baseline$params"]["Dense_0"]["kernel"]
    assert jnp.all(p1 == p2)

    rng = random.PRNGKey(0)
    best_val_loss = jnp.inf
    best_state = state
    for i in range(n_epochs):
        epoch_rng = jax.random.fold_in(rng, i)
        state, train_loss = train_epoch(
            svi, state, train_fetch, num_train, train_idx, epoch_rng
        )
        val_loss = eval_epoch(svi, state, test_fetch, num_test, test_idx, epoch_rng)
        print(f"Epoch loss - train loss: {train_loss}, validation loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state

    p2 = best_state.optim_state[1][0]["baseline$params"]["Dense_0"]["kernel"]
    assert jnp.all(p1 == p2)
    return svi.get_params(best_state)
