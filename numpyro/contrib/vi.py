from abc import abstractmethod
from collections import namedtuple
from typing import List

import jax

from numpyro import handlers
from numpyro.contrib import callbacks
from numpyro.util import fori_collect


class VI:
    CurrentState = namedtuple("CurrentState", ["optim_state", "rng_key"])

    def __init__(self, model, guide, optim, loss, name, **static_kwargs):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optim = optim
        self.name = name
        self.static_kwargs = static_kwargs

    @abstractmethod
    def get_params(self, state):
        raise NotImplementedError

    @abstractmethod
    def init(self, rng_key, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, state, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, state, *args, num_samples=1, **kwargs):
        raise NotImplementedError

    def run(
            self,
            rng_key,
            num_steps,
            *args,
            callbacks: List[callbacks.Callback] = None,
            batch_fun=None,
            validation_rate=5,
            validation_fun=None,
            restore=False,
            restore_path=None,
            jit_compile=True,
            **kwargs
    ):
        def bodyfn(_i, info):
            body_state = info[0]
            return *self.update(body_state, *info[2:], **kwargs), *info[2:]

        if batch_fun is not None:
            batch_args, batch_kwargs, _, _ = batch_fun(0)
        else:
            batch_args = ()
            batch_kwargs = {}

        state = self.init(rng_key, *args, *batch_args, **kwargs, **batch_kwargs)
        if restore and restore_path:
            opt_state, rng_key, loss = callbacks.Checkpoint.load(restore_path)
            state = VI.CurrentState(opt_state, rng_key)

            num_steps -= state[0][0]
        else:
            loss = self.evaluate(state, *args, *batch_args, **kwargs, **batch_kwargs)
        if (
                not callbacks
                and batch_fun is None
                and validation_fun is None
                and jit_compile
        ):
            losses, last_res = fori_collect(
                0,
                num_steps,
                lambda info: bodyfn(0, info),
                (state, loss, *args),
                progbar=False,
                transform=lambda val: val[0],
                return_last_val=True
            )
            state = last_res[0]
        else:
            losses = []
            try:
                train_info = {
                    "num_steps": num_steps,
                    "state": state,
                    "loss": loss,
                    "model_args": args,
                    "model_kwargs": kwargs,
                }
                if jit_compile:
                    bodyfn = jax.jit(bodyfn)
                for callback in callbacks:
                    callback.vi = self
                    callback.on_train_begin(train_info)
                epoch_begin = True
                for i in range(num_steps):
                    epoch = 1000, 0
                    is_last = False
                    if batch_fun is not None:
                        batch_args, batch_kwargs, epoch, is_last = batch_fun(i)
                        if epoch_begin:
                            epoch_begin = False
                            for callback in callbacks:
                                callback.on_train_epoch_begin(epoch, train_info)
                    else:
                        batch_args = args
                    for callback in callbacks:
                        callback.on_train_step_begin(i, train_info)
                    res = bodyfn(
                        i, (state, loss, *batch_args)  # , **kwargs, **batch_kwargs
                    )
                    state, loss = res[:2]
                    losses.append(loss)
                    train_info["state"] = state
                    train_info["loss"] = loss
                    for callback in callbacks:
                        callback.on_train_step_end(i, train_info)
                    if batch_fun is not None and is_last:
                        for callback in callbacks:
                            callback.on_train_epoch_end(epoch, train_info)
                        epoch_begin = True
                    if (i + 1) % validation_rate == 0 and validation_fun is not None:
                        val_step = (i + 1) // validation_rate
                        val_args, val_kwargs = validation_fun(val_step)
                        val_info = {
                            "model_args": [*args, *val_args],
                            "model_kwargs": {**kwargs, **val_kwargs},
                        }
                        for callback in callbacks:
                            callback.on_validation_begin(val_step, val_info)
                        val_loss = self.evaluate(
                            state, *args, *val_args, **kwargs, **val_kwargs
                        )
                        val_info["loss"] = val_loss
                        for callback in callbacks:
                            callback.on_validation_end(val_step, val_info)
                for callback in callbacks:
                    callback.on_train_end(train_info)
                    callback.vi = None
            except StopIteration:
                return state, losses
        return state, losses

    def _predict_model(self, rng_key, params, *args, **kwargs):
        guide_trace = handlers.trace(
            handlers.substitute(handlers.seed(self.guide, rng_key), params)
        ).get_trace(*args, **kwargs)
        model_trace = handlers.trace(
            handlers.replay(
                handlers.substitute(handlers.seed(self.model, rng_key), params),
                guide_trace,
            )
        ).get_trace(*args, **kwargs)
        return {
            name: site["value"]
            for name, site in model_trace.items()
            if ("is_observed" not in site) or not site["is_observed"]
        }
