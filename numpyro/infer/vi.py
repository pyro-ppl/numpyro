from abc import ABC, abstractmethod
from typing import List

import jax
from jax.lax import fori_loop

import numpyro.callbacks.callback as ncallback
from numpyro import handlers


class VI(ABC):
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

    def train(self, rng_key, num_steps, *args, callbacks: List[ncallback.Callback] = None, batch_fun=None,
              validation_rate=5, validation_fun=None, **kwargs):
        def bodyfn(_i, info, *args, **kwargs):
            body_state, _ = info
            return self.update(body_state, *args, **kwargs)
        if batch_fun is not None:
            batch_args, batch_kwargs, _, _ = batch_fun(0)
        else:
            batch_args = ()
            batch_kwargs = {}

        state = self.init(rng_key, *args, *batch_args, **kwargs, **batch_kwargs)
        loss = self.evaluate(state, *args, *batch_args, **kwargs, **batch_kwargs)
        if not callbacks:
            state, loss = fori_loop(0, num_steps, lambda i, info: bodyfn(i, info, *args, **kwargs), (state, loss))
        else:
            try:
                train_info = {
                    'num_steps': num_steps,
                    'state': state,
                    'loss': loss,
                    'model_args': args,
                    'model_kwargs': kwargs
                }
                bodyfn = jax.jit(bodyfn)
                for callback in callbacks:
                    callback.vi = self
                    callback.on_train_begin(train_info)
                epoch_begin = True
                for i in range(num_steps):
                    epoch = 0
                    is_last = False
                    if batch_fun is not None:
                        batch_args, batch_kwargs, epoch, is_last = batch_fun(i)
                        if epoch_begin:
                            epoch_begin = False
                            for callback in callbacks:
                                callback.on_train_epoch_begin(epoch, train_info)
                    for callback in callbacks:
                        callback.on_train_step_begin(i, train_info)
                    state, loss = bodyfn(i, (state, loss), *args, *batch_args, **kwargs, **batch_kwargs)
                    train_info['state'] = state
                    train_info['loss'] = loss
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
                            'model_args': [*args, *val_args],
                            'model_kwargs': {**kwargs, **val_kwargs}
                        }
                        for callback in callbacks:
                            callback.on_validation_begin(val_step, val_info)
                        val_loss = self.evaluate(state, *args, *val_args, **kwargs, **val_kwargs)
                        val_info['loss'] = val_loss
                        for callback in callbacks:
                            callback.on_validation_end(val_step, val_info)
                for callback in callbacks:
                    callback.on_train_end(train_info)
                    callback.vi = None
            except StopIteration:
                return state, loss
        return state, loss

    def _predict_model(self, rng_key, params, *args, **kwargs):
        guide_trace = handlers.trace(handlers.substitute(handlers.seed(self.guide, rng_key), params)
                                     ).get_trace(*args, **kwargs)
        model_trace = handlers.trace(handlers.replay(
            handlers.substitute(handlers.seed(self.model, rng_key), params), guide_trace)
        ).get_trace(*args, **kwargs)
        return {name: site['value'] for name, site in model_trace.items()
                if ('is_observed' not in site) or not site['is_observed']}
