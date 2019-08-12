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
the regression parameters using :func:`~numpyro.mcmc.mcmc`. The `log_likelihood` function
uses effect handlers to run the model by substituting sample sites with values from the posterior
distribution and computes the log density for a single data point. The `expected_log_likelihood`
function computes the log likelihood for each draw from the joint posterior and aggregates the
results, but does so by using JAX's auto-vectorize transform called `vmap` so that we do not
need to loop over all the data points.


.. testsetup::

   import jax.numpy as np
   from jax import random, vmap
   from jax.scipy.special import logsumexp
   import numpyro.distributions as dist
   from numpyro.handlers import sample, seed, substitute, trace
   from numpyro.hmc_util import initialize_model
   from numpyro.mcmc import mcmc

.. doctest::

   >>> N, D = 3000, 3
   >>> def logistic_regression(data, labels):
   ...     coefs = sample('coefs', dist.Normal(np.zeros(D), np.ones(D)))
   ...     intercept = sample('intercept', dist.Normal(0., 10.))
   ...     logits = np.sum(coefs * data + intercept, axis=-1)
   ...     return sample('obs', dist.Bernoulli(logits=logits), obs=labels)

   >>> data = random.normal(random.PRNGKey(0), (N, D))
   >>> true_coefs = np.arange(1., D + 1.)
   >>> logits = np.sum(true_coefs * data, axis=-1)
   >>> labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

   >>> init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(2), logistic_regression, data, labels)
   >>> num_warmup, num_samples = 1000, 1000
   >>> samples = mcmc(num_warmup, num_samples, init_params,
   ...                potential_fn=potential_fn,
   ...                constrain_fn=constrain_fn)  # doctest: +SKIP
   warmup: 100%|██████████| 1000/1000 [00:09<00:00, 109.40it/s, 1 steps of size 5.83e-01. acc. prob=0.79]
   sample: 100%|██████████| 1000/1000 [00:00<00:00, 1252.39it/s, 1 steps of size 5.83e-01. acc. prob=0.85]


                      mean         sd       5.5%      94.5%      n_eff       Rhat
       coefs[0]       0.96       0.07       0.85       1.07     455.35       1.01
       coefs[1]       2.05       0.09       1.91       2.20     332.00       1.01
       coefs[2]       3.18       0.13       2.96       3.37     320.27       1.00
      intercept      -0.03       0.02      -0.06       0.00     402.53       1.00

   >>> def log_likelihood(rng, params, model, *args, **kwargs):
   ...     model = substitute(seed(model, rng), params)
   ...     model_trace = trace(model).get_trace(*args, **kwargs)
   ...     obs_node = model_trace['obs']
   ...     return np.sum(obs_node['fn'].log_prob(obs_node['value']))

   >>> def expected_log_likelihood(rng, params, model, *args, **kwargs):
   ...     n = list(params.values())[0].shape[0]
   ...     log_lk_fn = vmap(lambda rng, params: log_likelihood(rng, params, model, *args, **kwargs))
   ...     log_lk_vals = log_lk_fn(random.split(rng, n), params)
   ...     return logsumexp(log_lk_vals) - np.log(n)

   >>> print(expected_log_likelihood(random.PRNGKey(2), samples, logistic_regression, data, labels))  # doctest: +SKIP
   -876.172
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from jax import random

from numpyro.distributions.constraints import biject_to, real, ComposeTransform

_PYRO_STACK = []


class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        _PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert _PYRO_STACK[-1] is self
        _PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class trace(Messenger):
    """
    Returns a handler that records the inputs and outputs at primitive calls
    inside `fn`.

    **Example**

    .. testsetup::

       from jax import random
       import numpyro.distributions as dist
       from numpyro.handlers import sample, seed, trace
       import pprint as pp

    .. doctest::

       >>> def model():
       ...     sample('a', dist.Normal(0., 1.))

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
        assert msg['name'] not in self.trace, 'all sites must have unique names'
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
       import numpyro.distributions as dist
       from numpyro.handlers import replay, sample, seed, trace

    .. doctest::

       >>> def model():
       ...     sample('a', dist.Normal(0., 1.))

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
       from numpyro.handlers import block, sample, seed, trace
       import numpyro.distributions as dist

    .. doctest::

       >>> def model():
       ...     a = sample('a', dist.Normal(0., 1.))
       ...     return sample('b', dist.Normal(a, 1.))

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
       from numpyro.handlers import sample, seed, substitute, trace
       import numpyro.distributions as dist

    .. doctest::

       >>> def model():
       ...     sample('a', dist.Normal(0., 1.))

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
    """
    def __init__(self, fn, rng):
        self.rng = rng
        super(seed, self).__init__(fn)

    def process_message(self, msg):
        if msg['type'] == 'sample':
            self.rng, rng_sample = random.split(self.rng)
            msg['kwargs']['random_state'] = rng_sample


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

     .. testsetup::

       from jax import random
       from numpyro.handlers import sample, seed, substitute, trace
       import numpyro.distributions as dist

    .. doctest::

       >>> def model():
       ...     sample('a', dist.Normal(0., 1.))

       >>> model = seed(model, random.PRNGKey(0))
       >>> exec_trace = trace(substitute(model, {'a': -1})).get_trace()
       >>> assert exec_trace['a']['value'] == -1
    """
    def __init__(self, fn=None, param_map=None, base_param_map=None, substitute_fn=None):
        self.substitute_fn = substitute_fn
        self.param_map = param_map
        self.base_param_map = base_param_map
        super(substitute, self).__init__(fn)

    def process_message(self, msg):
        if self.param_map is not None:
            if msg['name'] in self.param_map:
                msg['value'] = self.param_map[msg['name']]
        elif self.base_param_map is not None:
            if msg['name'] in self.base_param_map:
                if msg['type'] == 'sample':
                    msg['value'], msg['intermediates'] = msg['fn'].transform_with_intermediates(
                        self.base_param_map[msg['name']])
                else:
                    base_value = self.base_param_map[msg['name']]
                    constraint = msg['kwargs'].pop('constraint', real)
                    transform = biject_to(constraint)
                    if isinstance(transform, ComposeTransform):
                        msg['value'] = ComposeTransform(transform.parts[1:])(base_value)
                    else:
                        msg['value'] = self.base_param_map[msg['name']]
        elif self.substitute_fn is not None:
            base_value = self.substitute_fn(msg)
            if base_value is not None:
                if msg['type'] == 'sample':
                    msg['value'], msg['intermediates'] = msg['fn'].transform_with_intermediates(
                        base_value)
                else:
                    constraint = msg['kwargs'].pop('constraint', real)
                    transform = biject_to(constraint)
                    if isinstance(transform, ComposeTransform):
                        msg['value'] = ComposeTransform(transform.parts[1:])(base_value)
                    else:
                        msg['value'] = base_value
        else:
            raise ValueError("Neither `param_map`, `base_param_map`, nor `substitute_fn`"
                             "provided to substitute handler.")


def apply_stack(msg):
    pointer = 0
    for pointer, handler in enumerate(reversed(_PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg['value'] is None:
        if msg['type'] == 'sample':
            msg['value'], msg['intermediates'] = msg['fn'](*msg['args'],
                                                           sample_intermediates=True,
                                                           **msg['kwargs'])
        else:
            msg['value'] = msg['fn'](*msg['args'], **msg['kwargs'])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in _PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


def sample(name, fn, obs=None, sample_shape=()):
    """
    Returns a random sample from the stochastic function `fn`. This can have
    additional side effects when wrapped inside effect handlers like
    :class:`~numpyro.handlers.substitute`.

    :param str name: name of the sample site
    :param fn: Python callable
    :param numpy.ndarray obs: observed value
    :param sample_shape: Shape of samples to be drawn.
    :return: sample from the stochastic `fn`.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return fn(sample_shape=sample_shape)

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'sample',
        'name': name,
        'fn': fn,
        'args': (),
        'kwargs': {'sample_shape': sample_shape},
        'value': obs,
        'is_observed': obs is not None,
        'intermediates': [],
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']


def identity(x, *args, **kwargs):
    return x


def param(name, init_value, **kwargs):
    """
    Annotate the given site as an optimizable parameter for use with
    :mod:`jax.experimental.optimizers`. For an example of how `param` statements
    can be used in inference algorithms, refer to :func:`~numpyro.svi.svi`.

    :param str name: name of site.
    :param numpy.ndarray init_value: initial value specified by the user. Note that
        the onus of using this to initialize the optimizer is on the user /
        inference algorithm, since there is no global parameter store in
        NumPyro.
    :return: value for the parameter. Unless wrapped inside a
        handler like :class:`~numpyro.handlers.substitute`, this will simply
        return the initial value.
    """
    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not _PYRO_STACK:
        return init_value

    # Otherwise, we initialize a message...
    initial_msg = {
        'type': 'param',
        'name': name,
        'fn': identity,
        'args': (init_value,),
        'kwargs': kwargs,
        'value': None,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg['value']
