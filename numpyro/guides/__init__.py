from abc import ABC, abstractmethod
from numpyro import handlers
from numpyro.infer.util import find_valid_initial_params, init_to_uniform
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import biject_to
import jax.random

__all__ = ['ReinitGuide', 'WrappedGuide']

class ReinitGuide(ABC):
    @abstractmethod
    def init_params(self):
        raise NotImplementedError

    @abstractmethod
    def find_params(self, rng_keys, *args, **kwargs):
        raise NotImplementedError

# TODO Actually Test This out
class WrappedGuide(ReinitGuide):
    def __init__(self, fn, init_strategy=init_to_uniform()):
        self.fn = fn
        self._init_params = None
        self.init_strategy = init_strategy

    def init_params(self):
        return self._init_params

    def find_params(self, rng_keys, *args, **kwargs):
        guide_trace = handlers.trace(handlers.seed(self.fn, rng_keys[0])).get_trace(*args, **kwargs)
        init_params, _ = handlers.block(find_valid_initial_params)(rng_keys, self.fn,
                                                                   init_strategy=self.init_strategy,
                                                                   param_as_improper=True, # To get new values for existing parameters
                                                                   model_args=args,
                                                                   model_kwargs=kwargs)
        params = {}
        for name, site in guide_trace.items():
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', real)
                param_val = biject_to(constraint)(init_params[name])
                params[name] = (name, param_val, constraint)
        self._init_params = {param: (val, constr) for param, val, constr in params.values()}
