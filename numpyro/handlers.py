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


.. testsetup::

   import jax.numpy as np
   from jax import random, vmap
   from jax.scipy.special import logsumexp
   import numpyro
   import numpyro.distributions as dist
   from numpyro import handlers
   from numpyro.infer import MCMC, NUTS

.. doctest::

   >>> N, D = 3000, 3
   >>> def logistic_regression(data, labels):
   ...     coefs = numpyro.sample('coefs', dist.Normal(np.zeros(D), np.ones(D)))
   ...     intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
   ...     logits = np.sum(coefs * data + intercept, axis=-1)
   ...     return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

   >>> data = random.normal(random.PRNGKey(0), (N, D))
   >>> true_coefs = np.arange(1., D + 1.)
   >>> logits = np.sum(true_coefs * data, axis=-1)
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
   ...     return np.sum(logsumexp(log_lk_vals, 0) - np.log(n))

   >>> print(log_predictive_density(random.PRNGKey(2), mcmc.get_samples(),
   ...       logistic_regression, data, labels))  # doctest: +SKIP
   -874.89813
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import warnings

from jax import random
import jax.numpy as np

from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import ComposeTransform, biject_to
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

    .. testsetup::

       from jax import random
       import numpyro
       import numpyro.distributions as dist
       from numpyro.handlers import seed, trace
       import pprint as pp

    .. doctest::

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> exec_trace = trace(seed(model, random.PRNGKey(0))).get_trace()
       >>> pp.pprint(exec_trace)  # doctest: +SKIP
       OrderedDict([('a',
                     {'args': (),
                      'fn': <numpyro.distributions.continuous.Normal object at 0x7f9e689b1eb8>,
                      'is_observed': False,
                      'kwargs': {'random_state': DeviceArray([0, 0], dtype=uint32)},
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

    .. testsetup::

       from jax import random
       import numpyro
       import numpyro.distributions as dist
       from numpyro.handlers import replay, seed, trace

    .. doctest::

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
        if msg['name'] in self.guide_trace:
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

    .. testsetup::

       from jax import random
       import numpyro
       from numpyro.handlers import block, seed, trace
       import numpyro.distributions as dist

    .. doctest::

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

     .. testsetup::

       from jax import random
       import numpyro
       from numpyro.handlers import condition, seed, substitute, trace
       import numpyro.distributions as dist

    .. doctest::

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> model = seed(model, random.PRNGKey(0))
       >>> exec_trace = trace(condition(model, {'a': -1})).get_trace()
       >>> assert exec_trace['a']['value'] == -1
       >>> assert exec_trace['a']['is_observed']
    """
    def __init__(self, fn=None, param_map=None, substitute_fn=None):
        self.substitute_fn = substitute_fn
        self.param_map = param_map
        super(condition, self).__init__(fn)

    def process_message(self, msg):
        site_name = msg['name']
        if msg['type'] == 'sample':
            value = None
            if self.param_map is not None:
                if site_name in self.param_map:
                    value = self.param_map[site_name]
            else:
                value = self.substitute_fn(msg)
            if value is not None:
                msg['value'] = value
                if msg['is_observed']:
                    raise ValueError("Cannot condition an already observed site: {}.".format(site_name))
                msg['is_observed'] = True


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
        msg["scale"] = self.scale * msg.get('scale', 1)


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
    :type rng_seed: int, np.ndarray scalar, or jax.random.PRNGKey

    .. note::

        Unlike in Pyro, `numpyro.sample` primitive cannot be used without wrapping
        it in seed handler since there is no global random state. As such,
        users need to use `seed` as a contextmanager to generate samples from
        distributions or as a decorator for their model callable (See below).

    **Example:**

    .. testsetup::

      from jax import random
      import numpyro
      import numpyro.handlers
      import numpyro.distributions as dist

    .. doctest::

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
        if isinstance(rng_seed, int) or (isinstance(rng_seed, np.ndarray) and not np.shape(rng_seed)):
            rng_seed = random.PRNGKey(rng_seed)
        if not (isinstance(rng_seed, np.ndarray) and rng_seed.dtype == np.uint32 and rng_seed.shape == (2,)):
            raise TypeError('Incorrect type for rng_seed: {}'.format(type(rng_seed)))
        self.rng_key = rng_seed
        super(seed, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] == 'sample' and not msg['is_observed'] and \
                msg['kwargs']['random_state'] is None:
            self.rng_key, rng_key_sample = random.split(self.rng_key)
            msg['kwargs']['random_state'] = rng_key_sample


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
    :param dict base_param_map: similar to `param_map` but only holds samples
        from base distributions.
    :param substitute_fn: callable that takes in a site dict and returns
        a numpy array or `None` (in which case the handler has no side
        effect).

    **Example:**

     .. testsetup::

       from jax import random
       import numpyro
       from numpyro.handlers import seed, substitute, trace
       import numpyro.distributions as dist

    .. doctest::

       >>> def model():
       ...     numpyro.sample('a', dist.Normal(0., 1.))

       >>> model = seed(model, random.PRNGKey(0))
       >>> exec_trace = trace(substitute(model, {'a': -1})).get_trace()
       >>> assert exec_trace['a']['value'] == -1
    """
    def __init__(self, fn=None, param_map=None, base_param_map=None, substitute_fn=None):
        self.substitute_fn = substitute_fn
        self.param_map = param_map
        self.base_param_map = base_param_map
        if sum((x is not None for x in (param_map, base_param_map, substitute_fn))) != 1:
            raise ValueError('Only one of `param_map`, `base_param_map`, or `substitute_fn` '
                             'should be provided.')
        super(substitute, self).__init__(fn)

    def process_message(self, msg):
        if self.param_map is not None:
            if msg['name'] in self.param_map:
                msg['value'] = self.param_map[msg['name']]
        else:
            base_value = self.substitute_fn(msg) if self.substitute_fn \
                else self.base_param_map.get(msg['name'], None)
            if base_value is not None:
                if msg['type'] == 'sample':
                    msg['value'], msg['intermediates'] = msg['fn'].transform_with_intermediates(
                        base_value)
                else:
                    constraint = msg['kwargs'].pop('constraint', real)
                    transform = biject_to(constraint)
                    if isinstance(transform, ComposeTransform):
                        # No need to apply the first transform since the base value
                        # should have the same support as the first part's co-domain.
                        msg['value'] = ComposeTransform(transform.parts[1:])(base_value)
                    else:
                        msg['value'] = base_value
