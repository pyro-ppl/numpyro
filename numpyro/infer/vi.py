from abc import ABC, abstractmethod
from typing import List

import jax
from jax.lax import fori_loop

import numpyro.callbacks.callback as ncallback


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

    def train(self, rng_key, num_steps, *args, callbacks: List[ncallback.Callback] = None, **kwargs):
        def bodyfn(_i, info):
            body_state, _ = info
            return self.update(body_state, *args, **kwargs)

        state = self.init(rng_key, *args, **kwargs)
        loss = self.evaluate(state, *args, **kwargs)
        if not callbacks:
            state, loss = fori_loop(0, num_steps, bodyfn, (state, loss))
        else:
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
            for i in range(num_steps):
                for callback in callbacks: callback.on_train_step_begin(i, train_info)
                state, loss = bodyfn(i, (state, loss))
                train_info['state'] = state
                train_info['loss'] = loss
                for callback in callbacks: callback.on_train_step_end(i, train_info)
            for callback in callbacks:
                callback.on_train_end(train_info)
                callback.vi = None
        return state, loss
