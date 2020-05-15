from jax import lax, random

import numpyro
from numpyro import handlers
from numpyro.infer.util import log_density


class ScanDistribution():
    def __init__(self, fn, init, xs):
        self.fn = fn
        self.init = init
        self.xs = xs

    def sample(self, key, sample_shape=()):
        if sample_shape:
            raise NotImplementedError  # reshape and map/vmap?

        def body_fn(wrapped_val, x):
            rng_key, val = wrapped_val
            rng_key, subkey = random.split(rng_key)
            traced_fn = handlers.trace(handlers.seed(fn, subkey))
            val, y = traced_fn(val, x)
            site_values = {name: site["value"] for name, site in traced_fn.trace.items()}
            return (rng_key, val), (site_values, y)

        (_, last_val), (site_values, ys) = lax.scan(body_fn, (key, self.init), self.xs)
        self._last_val = last_val
        self._ys = ys
        return site_values

    def log_prob(value):
        def body_fn(carry, wrapped_x):
            site_values, x = wrapped_x
            log_joint, (carry, y) = log_density(
                fn, model_args=(carry, x), model_kwargs={}, param=site_values)
            return carry, (log_joint, y)

        last_val, (log_joint, ys) = lax.scan(body_fn, self.init_value, (value, self.xs))
        self._last_val = last_val
        self._ys = ys
        return log_joint.sum()


def scan(name, fn, init_value, xs):
    scan_dist = ScanDistribution(fn, init_value, xs)
    site_values = numpyro.sample(name, scan_dist)
    for name, value in site_values.items():
        numpyro.deterministic(name, value)
    scan_result = scan_dist._last_val, scan_dist._ys
    return scan_result
