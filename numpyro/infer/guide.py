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


class WrappedGuide(ReinitGuide):
    def __init__(self, fn, reinit_hide_fn=lambda site: site['name'].endswith('$params'), init_strategy=init_to_uniform):
        self.fn = fn
        self._init_params = None
        self.init_strategy = init_strategy(reinit_param=lambda site: not reinit_hide_fn(site))
        self._reinit_hide_fn = reinit_hide_fn

    def init_params(self):
        return self._init_params

    def find_params(self, rng_keys, *args, **kwargs):
        guide_trace = handlers.trace(handlers.seed(self.fn, rng_keys[0])).get_trace(*args, **kwargs)

        def _find_valid_params(rng_key):
            k1, k2 = jax.random.split(rng_key)
            guide = handlers.seed(handlers.block(self.fn, self._reinit_hide_fn), k2)
            guide_trace = handlers.trace(handlers.seed(self.fn, rng_key)).get_trace(*args, **kwargs)
            (mapped_params, _, _), _ = handlers.block(find_valid_initial_params)(k1, guide, init_strategy=self.init_strategy,
                                                                         model_args=args,
                                                                         model_kwargs=kwargs)
            hidden_params = {name: site['value'] for name, site in guide_trace.items()
                             if site['type'] == 'param' and self._reinit_hide_fn(site)}
            res_params = {**mapped_params, **hidden_params}
            return res_params

        init_params = jax.vmap(_find_valid_params)(rng_keys)
        params = {}
        for name, site in guide_trace.items():
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', real)
                param_val = biject_to(constraint)(init_params[name])
                params[name] = (name, param_val, constraint)
        self._init_params = {param: (val, constr) for param, val, constr in params.values()}

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
