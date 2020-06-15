# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This provides a small set of effect handlers in NumPyro that are modeled
after Pyro's `poutine <http://docs.pyro.ai/en/stable/poutine.html>`_ module.
For a tutorial on effect handlers more generally, readers are encouraged to
read `Poutine: A Guide to Programming with Effect Handlers in Pyro
<http://pyro.ai/examples/effect_handlers.html>`_. These simple effect handlers
can be composed together or new ones added to enable implementation of custom
inference utilities and algorithms.

**Example**

As an example, we are using :class:`~numpyro.handlers.seed`, :class:`~numpyro.handlers.trace`
and :class:`~numpyro.handlers.substitute` handlers to define the `log_likelihood` function below.
We first create a logistic regression model and sample from the posterior distribution over
the regression parameters using :func:`~numpyro.infer.MCMC`. The `log_likelihood` function
uses effect handlers to run the model by substituting sample sites with values from the posterior
distribution and computes the log density for a single data point. The `log_predictive_density`
function computes the log likelihood for each draw from the joint posterior and aggregates the
results for all the data points, but does so by using JAX's auto-vectorize transform called
`vmap` so that we do not need to loop over all the data points.



.. doctest::

   >>> import jax.numpy as jnp
   >>> from jax import random, vmap
   >>> from jax.scipy.special import logsumexp
   >>> import numpyro
   >>> import numpyro.distributions as dist
   >>> from numpyro import handlers
   >>> from numpyro.infer import MCMC, NUTS

   >>> N, D = 3000, 3
   >>> def logistic_regression(data, labels):
   ...     coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(D), jnp.ones(D)))
   ...     intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
   ...     logits = jnp.sum(coefs * data + intercept, axis=-1)
   ...     return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

   >>> data = random.normal(random.PRNGKey(0), (N, D))
   >>> true_coefs = jnp.arange(1., D + 1.)
   >>> logits = jnp.sum(true_coefs * data, axis=-1)
   >>> labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

   >>> num_warmup, num_samples = 1000, 1000
   >>> mcmc = MCMC(NUTS(model=logistic_regression), num_warmup, num_samples)
   >>> mcmc.run(random.PRNGKey(2), data, labels)  # doctest: +SKIP
   sample: 100%|██████████| 1000/1000 [00:00<00:00, 1252.39it/s, 1 steps of size 5.83e-01. acc. prob=0.85]
   >>> mcmc.print_summary()  # doctest: +SKIP


                      mean         sd       5.5%      94.5%      n_eff       Rhat
       coefs[0]       0.96       0.07       0.85       1.07     455.35       1.01
       coefs[1]       2.05       0.09       1.91       2.20     332.00       1.01
       coefs[2]       3.18       0.13       2.96       3.37     320.27       1.00
      intercept      -0.03       0.02      -0.06       0.00     402.53       1.00

   >>> def log_likelihood(rng_key, params, model, *args, **kwargs):
   ...     model = handlers.substitute(handlers.seed(model, rng_key), params)
   ...     model_trace = handlers.trace(model).get_trace(*args, **kwargs)
   ...     obs_node = model_trace['obs']
   ...     return obs_node['fn'].log_prob(obs_node['value'])

   >>> def log_predictive_density(rng_key, params, model, *args, **kwargs):
   ...     n = list(params.values())[0].shape[0]
   ...     log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
   ...     log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
   ...     return jnp.sum(logsumexp(log_lk_vals, 0) - jnp.log(n))

   >>> print(log_predictive_density(random.PRNGKey(2), mcmc.get_samples(),
   ...       logistic_regression, data, labels))  # doctest: +SKIP
   -874.89813
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import warnings

from jax import lax, random
import jax.numpy as jnp

from numpyro.primitives import Messenger
from numpyro.util import not_jax_tracer

__all__ = [
    'block',
    'condition',
    'replay',
    'scale',
    'seed',
    'substitute',
    'trace',
]


class trace(Messenger):
    """
    Returns a handler that records the inputs and outputs at primitive calls
    inside `fn`.

    **Example**

    .. doctest::

       >>> from jax import random
       >>> import numpyro
       >>> import numpyro.distributions as dist
       >>> from numpyro.handlers import seed, trace
       >>> import pprint as pp

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> exec_trace = trace(seed(model, random.PRNGKey(0))).get_trace()
       >>> pp.pprint(exec_trace)  # doctest: +SKIP
       OrderedDict([('a',
                     {'args': (),
                      'fn': <numpyro.distributions.continuous.Normal object at 0x7f9e689b1eb8>,
                      'is_observed': False,
                      'kwargs': {'rng_key': DeviceArray([0, 0], dtype=uint32)},
                      'name': 'a',
                      'type': 'sample',
                      'value': DeviceArray(-0.20584235, dtype=float32)})])
    """
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        assert not(msg['type'] == 'sample' and msg['name'] in self.trace), 'all sites must have unique names'
        self.trace[msg['name']] = msg.copy()

    def get_trace(self, *args, **kwargs):
        """
        Run the wrapped callable and return the recorded trace.

        :param `*args`: arguments to the callable.
        :param `**kwargs`: keyword arguments to the callable.
        :return: `OrderedDict` containing the execution trace.
        """
        self(*args, **kwargs)
        return self.trace


class replay(Messenger):
    """
    Given a callable `fn` and an execution trace `guide_trace`,
    return a callable which substitutes `sample` calls in `fn` with
    values from the corresponding site names in `guide_trace`.

    :param fn: Python callable with NumPyro primitives.
    :param guide_trace: an OrderedDict containing execution metadata.

    **Example**

    .. doctest::

       >>> from jax import random
       >>> import numpyro
       >>> import numpyro.distributions as dist
       >>> from numpyro.handlers import replay, seed, trace

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> exec_trace = trace(seed(model, random.PRNGKey(0))).get_trace()
       >>> print(exec_trace['a']['value'])  # doctest: +SKIP
       -0.20584235
       >>> replayed_trace = trace(replay(model, exec_trace)).get_trace()
       >>> print(exec_trace['a']['value'])  # doctest: +SKIP
       -0.20584235
       >>> assert replayed_trace['a']['value'] == exec_trace['a']['value']
    """
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg['name'] in self.guide_trace and msg['type'] in ('sample', 'plate'):
            msg['value'] = self.guide_trace[msg['name']]['value']


class block(Messenger):
    """
    Given a callable `fn`, return another callable that selectively hides
    primitive sites  where `hide_fn` returns True from other effect handlers
    on the stack.

    :param fn: Python callable with NumPyro primitives.
    :param hide_fn: function which when given a dictionary containing
        site-level metadata returns whether it should be blocked.

    **Example:**

    .. doctest::

       >>> from jax import random
       >>> import numpyro
       >>> from numpyro.handlers import block, seed, trace
       >>> import numpyro.distributions as dist

       >>> def model():
       ...     a = numpyro.sample('a', dist.Normal(0., 1.))
       ...     return numpyro.sample('b', dist.Normal(a, 1.))

       >>> model = seed(model, random.PRNGKey(0))
       >>> block_all = block(model)
       >>> block_a = block(model, lambda site: site['name'] == 'a')
       >>> trace_block_all = trace(block_all).get_trace()
       >>> assert not {'a', 'b'}.intersection(trace_block_all.keys())
       >>> trace_block_a =  trace(block_a).get_trace()
       >>> assert 'a' not in trace_block_a
       >>> assert 'b' in trace_block_a
    """
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg['stop'] = True


class condition(Messenger):
    """
    Conditions unobserved sample sites to values from `param_map` or `condition_fn`.
    Similar to :class:`~numpyro.handlers.substitute` except that it only affects
    `sample` sites and changes the `is_observed` property to `True`.

    :param fn: Python callable with NumPyro primitives.
    :param dict param_map: dictionary of `numpy.ndarray` values keyed by
       site names.
    :param condition_fn: callable that takes in a site dict and returns
       a numpy array or `None` (in which case the handler has no side
       effect).

    **Example:**

    .. doctest::

       >>> from jax import random
       >>> import numpyro
       >>> from numpyro.handlers import condition, seed, substitute, trace
       >>> import numpyro.distributions as dist

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> model = seed(model, random.PRNGKey(0))
       >>> exec_trace = trace(condition(model, {'a': -1})).get_trace()
       >>> assert exec_trace['a']['value'] == -1
       >>> assert exec_trace['a']['is_observed']
    """
    def __init__(self, fn=None, param_map=None, condition_fn=None):
        self.condition_fn = condition_fn
        self.param_map = param_map
        if sum((x is not None for x in (param_map, condition_fn))) != 1:
            raise ValueError('Only one of `param_map` or `condition_fn` '
                             'should be provided.')
        super(condition, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] != 'sample':
            return

        if self.param_map is not None:
            value = self.param_map.get(msg['name'])
        else:
            value = self.condition_fn(msg)

        if value is not None:
            msg['value'] = value
            if msg['is_observed']:
                raise ValueError("Cannot condition an already observed site: {}.".format(msg['name']))
            msg['is_observed'] = True


class mask(Messenger):
    """
    This messenger masks out some of the sample statements elementwise.

    :param mask_array: a DeviceArray with `bool` dtype for masking elementwise masking
        of sample sites.
    """
    def __init__(self, fn=None, mask_array=True):
        if lax.dtype(mask_array) != 'bool':
            raise ValueError("`mask` should be a bool array.")
        self.mask = mask_array
        super(mask, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] != 'sample':
            return

        msg['mask'] = self.mask if msg['mask'] is None else self.mask & msg['mask']


class reparam(Messenger):
    """
    Reparametrizes each affected sample site into one or more auxiliary sample
    sites followed by a deterministic transformation [1].

    To specify reparameterizers, pass a ``config`` dict or callable to the
    constructor.  See the :mod:`numpyro.infer.reparam` module for available
    reparameterizers.

    Note some reparameterizers can examine the ``*args,**kwargs`` inputs of
    functions they affect; these reparameterizers require using
    ``handlers.reparam`` as a decorator rather than as a context manager.

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf

    :param config: Configuration, either a dict mapping site name to
        :class:`~numpyro.infer.reparam.Reparam` ,
        or a function mapping site to
        :class:`~numpyro.infer.reparam.Reparam` or None.
    :type config: dict or callable
    """
    def __init__(self, fn=None, config=None):
        assert isinstance(config, dict) or callable(config)
        self.config = config
        super().__init__(fn)

    def process_message(self, msg):
        if msg["type"] != "sample":
            return

        if isinstance(self.config, dict):
            reparam = self.config.get(msg["name"])
        else:
            reparam = self.config(msg)
        if reparam is None:
            return

        new_fn, value = reparam(msg["name"], msg["fn"], msg["value"])

        if value is not None:
            if new_fn is None:
                msg['type'] = 'deterministic'
                msg['value'] = value
                for key in list(msg.keys()):
                    if key not in ('type', 'name', 'value'):
                        del msg[key]
                return

            if msg["value"] is None:
                msg["is_observed"] = True
            msg["value"] = value
        msg["fn"] = new_fn


class scale(Messenger):
    """
    This messenger rescales the log probability score.

    This is typically used for data subsampling or for stratified sampling of data
    (e.g. in fraud detection where negatives vastly outnumber positives).

    :param float scale_factor: a positive scaling factor
    """
    def __init__(self, fn=None, scale_factor=1.):
        if not_jax_tracer(scale_factor):
            if scale_factor <= 0:
                raise ValueError("scale factor should be a positive number.")
        self.scale = scale_factor
        super(scale, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] not in ('sample', 'plate'):
            return

        msg["scale"] = self.scale if msg.get('scale') is None else self.scale * msg['scale']


class seed(Messenger):
    """
    JAX uses a functional pseudo random number generator that requires passing
    in a seed :func:`~jax.random.PRNGKey` to every stochastic function. The
    `seed` handler allows us to initially seed a stochastic function with a
    :func:`~jax.random.PRNGKey`. Every call to the :func:`~numpyro.handlers.sample`
    primitive inside the function results in a splitting of this initial seed
    so that we use a fresh seed for each subsequent call without having to
    explicitly pass in a `PRNGKey` to each `sample` call.

    :param fn: Python callable with NumPyro primitives.
    :param rng_seed: a random number generator seed.
    :type rng_seed: int, jnp.ndarray scalar, or jax.random.PRNGKey

    .. note::

        Unlike in Pyro, `numpyro.sample` primitive cannot be used without wrapping
        it in seed handler since there is no global random state. As such,
        users need to use `seed` as a contextmanager to generate samples from
        distributions or as a decorator for their model callable (See below).

    **Example:**

    .. doctest::

       >>> from jax import random
       >>> import numpyro
       >>> import numpyro.handlers
       >>> import numpyro.distributions as dist

       >>> # as context manager
       >>> with handlers.seed(rng_seed=1):
       ...     x = numpyro.sample('x', dist.Normal(0., 1.))

       >>> def model():
       ...     return numpyro.sample('y', dist.Normal(0., 1.))

       >>> # as function decorator (/modifier)
       >>> y = handlers.seed(model, rng_seed=1)()
       >>> assert x == y
    """
    def __init__(self, fn=None, rng_seed=None, rng=None):
        if rng is not None:
            warnings.warn('`rng` argument is deprecated and renamed to `rng_seed` instead.', DeprecationWarning)
            rng_seed = rng
        if isinstance(rng_seed, int) or (isinstance(rng_seed, jnp.ndarray) and not jnp.shape(rng_seed)):
            rng_seed = random.PRNGKey(rng_seed)
        if not (isinstance(rng_seed, jnp.ndarray) and rng_seed.dtype == jnp.uint32 and rng_seed.shape == (2,)):
            raise TypeError('Incorrect type for rng_seed: {}'.format(type(rng_seed)))
        self.rng_key = rng_seed
        super(seed, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] == 'sample' and not msg['is_observed'] and \
                msg['kwargs']['rng_key'] is None:
            self.rng_key, rng_key_sample = random.split(self.rng_key)
            msg['kwargs']['rng_key'] = rng_key_sample


class substitute(Messenger):
    """
    Given a callable `fn` and a dict `param_map` keyed by site names
    (alternatively, a callable `substitute_fn`), return a callable
    which substitutes all primitive calls in `fn` with values from
    `param_map` whose key matches the site name. If the site name
    is not present in `param_map`, there is no side effect.

    If a `substitute_fn` is provided, then the value at the site is
    replaced by the value returned from the call to `substitute_fn`
    for the given site.

    :param fn: Python callable with NumPyro primitives.
    :param dict param_map: dictionary of `numpy.ndarray` values keyed by
        site names.
    :param substitute_fn: callable that takes in a site dict and returns
        a numpy array or `None` (in which case the handler has no side
        effect).

    **Example:**

    .. doctest::

       >>> from jax import random
       >>> import numpyro
       >>> from numpyro.handlers import seed, substitute, trace
       >>> import numpyro.distributions as dist

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> model = seed(model, random.PRNGKey(0))
       >>> exec_trace = trace(substitute(model, {'a': -1})).get_trace()
       >>> assert exec_trace['a']['value'] == -1
    """
    def __init__(self, fn=None, param_map=None, substitute_fn=None):
        self.substitute_fn = substitute_fn
        self.param_map = param_map
        if sum((x is not None for x in (param_map, substitute_fn))) != 1:
            raise ValueError('Only one of `param_map` or `substitute_fn` '
                             'should be provided.')
        super(substitute, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] not in ('sample', 'param'):
            return

        if self.param_map is not None:
            value = self.param_map.get(msg['name'])
        else:
            value = self.substitute_fn(msg)

        if value is not None:
            msg['value'] = value
