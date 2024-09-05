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
   >>> mcmc = MCMC(NUTS(model=logistic_regression), num_warmup=num_warmup, num_samples=num_samples)
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

from collections import OrderedDict
import warnings

import numpy as np

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.distributions.distribution import COERCIONS
from numpyro.primitives import (
    _PYRO_STACK,
    CondIndepStackFrame,
    Messenger,
    apply_stack,
    plate,
)
from numpyro.util import find_stack_level, is_prng_key, not_jax_tracer

__all__ = [
    "block",
    "collapse",
    "condition",
    "infer_config",
    "lift",
    "mask",
    "reparam",
    "replay",
    "scale",
    "scope",
    "seed",
    "substitute",
    "trace",
    "do",
]


class trace(Messenger):
    """
    Returns a handler that records the inputs and outputs at primitive calls
    inside `fn`.

    **Example:**

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
                      'kwargs': {'rng_key': Array([0, 0], dtype=uint32)},
                      'name': 'a',
                      'type': 'sample',
                      'value': Array(-0.20584235, dtype=float32)})])
    """

    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        if "name" not in msg:
            # skip recording helper messages e.g. `control_flow`, `to_data`, `to_funsor`
            # which has no name
            return
        assert not (
            msg["type"] in ("sample", "deterministic") and msg["name"] in self.trace
        ), "all sites must have unique names but got `{}` duplicated".format(
            msg["name"]
        )
        self.trace[msg["name"]] = msg.copy()

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
    Given a callable `fn` and an execution trace `trace`,
    return a callable which substitutes `sample` calls in `fn` with
    values from the corresponding site names in `trace`.

    :param fn: Python callable with NumPyro primitives.
    :param trace: an OrderedDict containing execution metadata.

    **Example:**

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

    def __init__(self, fn=None, trace=None):
        assert trace is not None
        self.trace = trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] in ("sample", "plate") and msg["name"] in self.trace:
            name = msg["name"]
            guide_msg = self.trace[name]
            if msg["type"] == "plate":
                if guide_msg["type"] != "plate":
                    raise RuntimeError(f"Site {name} must be a plate in trace.")
                msg["value"] = guide_msg["value"]
                return None
            if msg["is_observed"]:
                return None
            if guide_msg["type"] != "sample" or guide_msg["is_observed"]:
                raise RuntimeError(f"Site {name} must be sampled in trace.")
            msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"].copy()
        if msg["type"] == "control_flow":
            msg["kwargs"]["substitute_stack"].append(("replay", self.trace))


class block(Messenger):
    """
    Given a callable `fn`, return another callable that selectively hides
    primitive sites from other effect handlers on the stack. In the absence
    of parameters, all primitive sites are blocked. `hide_fn` takes precedence
    over `hide`, which has higher priority than `expose_types` followed by `expose`.
    Only the parameter with the precedence is considered.

    :param callable fn: Python callable with NumPyro primitives.
    :param callable hide_fn: function which when given a dictionary containing
        site-level metadata returns whether it should be blocked.
    :param list hide: list of site names to hide.
    :param list expose_types: list of site types to expose, e.g. `['param']`.
    :param list expose: list of site names to expose.
    :returns: Python callable with NumPyro primitives.

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

    def __init__(
        self,
        fn=None,
        hide_fn=None,
        hide=None,
        expose_types=None,
        expose=None,
    ):
        if hide_fn is not None:
            self.hide_fn = hide_fn
        elif hide is not None:
            self.hide_fn = lambda msg: msg.get("name") in hide
        elif expose_types is not None:
            self.hide_fn = lambda msg: msg.get("type") not in expose_types
        elif expose is not None:
            self.hide_fn = lambda msg: msg.get("name") not in expose
        else:
            self.hide_fn = lambda msg: True
        super(block, self).__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True


class collapse(trace):
    """
    EXPERIMENTAL Collapses all sites in the context by lazily sampling and
    attempting to use conjugacy relations. If no conjugacy is known this will
    fail. Code using the results of sample sites must be written to accept
    Funsors rather than Tensors. This requires ``funsor`` to be installed.
    """

    _coerce = None

    def __init__(self, *args, **kwargs):
        if collapse._coerce is None:
            import funsor
            from funsor.distribution import CoerceDistributionToFunsor

            funsor.set_backend("jax")
            collapse._coerce = CoerceDistributionToFunsor("jax")
        super().__init__(*args, **kwargs)

    def process_message(self, msg):
        from funsor.terms import Funsor

        if msg["type"] == "sample":
            if msg["value"] is None:
                msg["value"] = msg["name"]

            if isinstance(msg["fn"], Funsor) or isinstance(msg["value"], (str, Funsor)):
                msg["stop"] = True

    def __enter__(self):
        self.preserved_plates = frozenset(
            h.name for h in _PYRO_STACK if isinstance(h, plate)
        )
        COERCIONS.append(self._coerce)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        import funsor

        _coerce = COERCIONS.pop()
        assert _coerce is self._coerce
        super().__exit__(exc_type, exc_value, traceback)

        if exc_type is not None:
            return

        # Convert delayed statements to pyro.factor()
        reduced_vars = []
        log_prob_terms = []
        plates = frozenset()
        for name, site in self.trace.items():
            if site["type"] != "sample":
                continue
            if not site["is_observed"]:
                reduced_vars.append(name)
            dim_to_name = {f.dim: f.name for f in site["cond_indep_stack"]}
            fn = funsor.to_funsor(site["fn"], funsor.Real, dim_to_name)
            value = site["value"]
            if not isinstance(value, str):
                value = funsor.to_funsor(site["value"], fn.inputs["value"], dim_to_name)
            log_prob_terms.append(fn(value=value))
            plates |= frozenset(f.name for f in site["cond_indep_stack"])
        assert log_prob_terms, "nothing to collapse"
        reduced_plates = plates - self.preserved_plates
        log_prob = funsor.sum_product.sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            log_prob_terms,
            eliminate=frozenset(reduced_vars) | reduced_plates,
            plates=plates,
        )
        name = reduced_vars[0]
        numpyro.factor(name, log_prob.data)


class condition(Messenger):
    """
    Conditions unobserved sample sites to values from `data` or `condition_fn`.
    Similar to :class:`~numpyro.handlers.substitute` except that it only affects
    `sample` sites and changes the `is_observed` property to `True`.

    :param fn: Python callable with NumPyro primitives.
    :param dict data: dictionary of `numpy.ndarray` values keyed by
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

    def __init__(self, fn=None, data=None, condition_fn=None):
        self.condition_fn = condition_fn
        self.data = data
        if sum((x is not None for x in (data, condition_fn))) != 1:
            raise ValueError(
                "Only one of `data` or `condition_fn` " "should be provided."
            )
        super(condition, self).__init__(fn)

    def process_message(self, msg):
        if (msg["type"] != "sample") or msg.get("_control_flow_done", False):
            if msg["type"] == "control_flow":
                if self.data is not None:
                    msg["kwargs"]["substitute_stack"].append(("condition", self.data))
                if self.condition_fn is not None:
                    msg["kwargs"]["substitute_stack"].append(
                        ("condition", self.condition_fn)
                    )
            return

        if self.data is not None:
            value = self.data.get(msg["name"])
        else:
            value = self.condition_fn(msg)

        if value is not None:
            msg["value"] = value
            msg["is_observed"] = True


class infer_config(Messenger):
    """
    Given a callable `fn` that contains NumPyro primitive calls
    and a callable `config_fn` taking a trace site and returning a dictionary,
    updates the value of the infer kwarg at a sample site to config_fn(site).

    :param fn: a stochastic function (callable containing NumPyro primitive calls)
    :param config_fn: a callable taking a site and returning an infer dict
    """

    def __init__(self, fn=None, config_fn=None):
        super().__init__(fn)
        self.config_fn = config_fn

    def process_message(self, msg):
        if msg["type"] in ("sample",):
            msg["infer"].update(self.config_fn(msg))


class lift(Messenger):
    """
    Given a stochastic function with ``param`` calls and a prior distribution,
    create a stochastic function where all param calls are replaced by sampling from prior.
    Prior should be a distribution or a dict of names to distributions.

    Consider the following NumPyro program:

        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.handlers import lift
        >>>
        >>> def model(x):
        ...     s = numpyro.param("s", 0.5)
        ...     z = numpyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2
        >>> lifted_model = lift(model, prior={"s": dist.Exponential(0.3)})

    ``lift`` makes ``param`` statements behave like ``sample`` statements
    using the distributions in ``prior``.  In this example, site `s` will now behave
    as if it was replaced with ``s = numpyro.sample("s", dist.Exponential(0.3))``.

    :param fn: function whose parameters will be lifted to random values
    :param prior: prior function in the form of a Distribution or a dict of Distributions
    """

    def __init__(self, fn=None, prior=None):
        super().__init__(fn)
        self.prior = prior
        self._samples_cache = {}

    def __enter__(self):
        self._samples_cache = {}
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        self._samples_cache = {}
        return super().__exit__(*args, **kwargs)

    def process_message(self, msg):
        if msg["type"] != "param":
            return

        name = msg["name"]
        fn = self.prior.get(name) if isinstance(self.prior, dict) else self.prior
        if isinstance(fn, numpyro.distributions.Distribution):
            msg["type"] = "sample"
            msg["fn"] = fn
            msg["args"] = ()
            msg["kwargs"] = {
                "rng_key": msg["kwargs"].get("rng_key", None),
                "sample_shape": msg["kwargs"].get("sample_shape", ()),
            }
            msg["intermediates"] = []
            msg["infer"] = msg.get("infer", {})
        else:
            # otherwise leave as is
            return

        if name in self._samples_cache:
            # Multiple pyro.param statements with the same
            # name. Block the site and fix the value.
            msg["value"] = self._samples_cache[name]["value"]
            msg["is_observed"] = True
            msg["stop"] = True
        else:
            self._samples_cache[name] = msg
            msg["is_observed"] = False


class mask(Messenger):
    """
    This messenger masks out some of the sample statements elementwise.

    :param mask: a boolean or a boolean-valued array for masking elementwise log
        probability of sample sites (`True` includes a site, `False` excludes a site).
    """

    def __init__(self, fn=None, mask=True):
        if jnp.result_type(mask) != "bool":
            raise ValueError("`mask` should be a bool array.")
        self.mask = mask
        super().__init__(fn)

    def process_message(self, msg):
        if msg["type"] != "sample":
            if msg["type"] == "inspect":
                msg["mask"] = (
                    self.mask if msg["mask"] is None else (self.mask & msg["mask"])
                )
            return

        msg["fn"] = msg["fn"].mask(self.mask)


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
                msg["type"] = "deterministic"
                msg["value"] = value
                for key in list(msg.keys()):
                    if key not in ("type", "name", "value", "cond_indep_stack"):
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

    :param scale: a positive scaling factor that is broadcastable to the shape
        of log probability.
    :type scale: float or numpy.ndarray
    """

    def __init__(self, fn=None, scale=1.0):
        if not_jax_tracer(scale):
            if np.any(np.less_equal(scale, 0)):
                raise ValueError("'scale' argument should be positive.")
        self.scale = scale
        super().__init__(fn)

    def process_message(self, msg):
        if msg["type"] not in ("param", "sample", "plate"):
            return

        msg["scale"] = (
            self.scale if msg.get("scale") is None else self.scale * msg["scale"]
        )
        plate_to_scale = msg.setdefault("plate_to_scale", {})
        scale = (
            self.scale
            if plate_to_scale.get(None) is None
            else self.scale * plate_to_scale[None]
        )
        plate_to_scale[None] = scale


class scope(Messenger):
    """
    This handler prepend a prefix followed by a divider to the name of sample sites.

    **Example:**

    .. doctest::

        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.handlers import scope, seed, trace

        >>> def model():
        ...     with scope(prefix="a"):
        ...         with scope(prefix="b", divider="."):
        ...             return numpyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/b.x" in trace(seed(model, 0)).get_trace()

    :param fn: Python callable with NumPyro primitives.
    :param str prefix: a string to prepend to sample names
    :param str divider: a string to join the prefix and sample name; default to `'/'`
    :param list hide_types: an optional list of side types to skip renaming.
    """

    def __init__(self, fn=None, prefix="", divider="/", *, hide_types=None):
        self.prefix = prefix
        self.divider = divider
        self.hide_types = [] if hide_types is None else hide_types
        super().__init__(fn)

    def process_message(self, msg):
        if msg.get("name") and msg["type"] not in self.hide_types:
            msg["name"] = f"{self.prefix}{self.divider}{msg['name']}"

        if msg.get("cond_indep_stack") and "plate" not in self.hide_types:
            msg["cond_indep_stack"] = [
                CondIndepStackFrame(
                    f"{self.prefix}{self.divider}{i.name}", i.dim, i.size
                )
                for i in msg["cond_indep_stack"]
            ]


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
    :param list hide_types: an optional list of side types to skip seeding, e.g. ['plate'].

    .. note::

        Unlike in Pyro, `numpyro.sample` primitive cannot be used without wrapping
        it in seed handler since there is no global random state. As such,
        users need to use `seed` as a contextmanager to generate samples from
        distributions or as a decorator for their model callable (See below).

    .. note::

       The seed handler has a mutable attribute `rng_key` which keeps changing after
       each `sample` call. Hence an instance of this class (e.g. `seed(model, rng_seed=0)`)
       might create tracer-leaks when jitted. A solution is to close the instance in a
       function, e.g., `seeded_model = lambda *args: seed(model, rng_seed=0)(*args)`.
       This `seeded_model` can be jitted.

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

    def __init__(self, fn=None, rng_seed=None, hide_types=None):
        if not is_prng_key(rng_seed) and (
            isinstance(rng_seed, int)
            or (
                isinstance(rng_seed, (np.ndarray, jnp.ndarray))
                and not jnp.shape(rng_seed)
            )
        ):
            rng_seed = random.PRNGKey(rng_seed)
        if not is_prng_key(rng_seed):
            raise TypeError("Incorrect type for rng_seed: {}".format(type(rng_seed)))
        self.rng_key = rng_seed
        self.hide_types = [] if hide_types is None else hide_types
        super(seed, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] in self.hide_types:
            return
        if msg["type"] not in ["sample", "prng_key", "plate", "control_flow"]:
            return
        if (msg["kwargs"]["rng_key"] is not None) or (msg["value"] is not None):
            # no need to create a new key when value is available
            return
        self.rng_key, rng_key_sample = random.split(self.rng_key)
        msg["kwargs"]["rng_key"] = rng_key_sample


class substitute(Messenger):
    """
    Given a callable `fn` and a dict `data` keyed by site names
    (alternatively, a callable `substitute_fn`), return a callable
    which substitutes all primitive calls in `fn` with values from
    `data` whose key matches the site name. If the site name
    is not present in `data`, there is no side effect.

    If a `substitute_fn` is provided, then the value at the site is
    replaced by the value returned from the call to `substitute_fn`
    for the given site.

    .. note:: This handler is mainly used for internal algorithms.
        For conditioning a generative model on observed data, please
        use the :class:`condition` handler.

    :param fn: Python callable with NumPyro primitives.
    :param dict data: dictionary of `numpy.ndarray` values keyed by
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

    def __init__(self, fn=None, data=None, substitute_fn=None):
        self.substitute_fn = substitute_fn
        self.data = data
        if sum((x is not None for x in (data, substitute_fn))) != 1:
            raise ValueError(
                "Only one of `data` or `substitute_fn` " "should be provided."
            )
        super(substitute, self).__init__(fn)

    def process_message(self, msg):
        if (
            msg["type"] not in ("sample", "param", "mutable", "plate", "deterministic")
        ) or msg.get("_control_flow_done", False):
            if msg["type"] == "control_flow":
                if self.data is not None:
                    msg["kwargs"]["substitute_stack"].append(("substitute", self.data))
                if self.substitute_fn is not None:
                    msg["kwargs"]["substitute_stack"].append(
                        ("substitute", self.substitute_fn)
                    )
            return

        if self.data is not None:
            value = self.data.get(msg.get("name"))
        else:
            value = self.substitute_fn(msg)

        if value is not None:
            msg["value"] = value


class do(Messenger):
    """
    Given a stochastic function with some sample statements and a dictionary
    of values at names, set the return values of those sites equal to the values
    as if they were hard-coded to those values and introduce fresh sample sites
    with the same names whose values do not propagate.

    Composes freely with :func:`~numpyro.handlers.condition` to represent
    counterfactual distributions over potential outcomes. See Single World
    Intervention Graphs [1] for additional details and theory.

    This is equivalent to replacing `z = numpyro.sample("z", ...)` with `z = 1.`
    and introducing a fresh sample site `numpyro.sample("z", ...)` whose value is
    not used elsewhere.

    **References:**

    1. *Single World Intervention Graphs: A Primer*,
       Thomas Richardson, James Robins

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param data: a ``dict`` mapping sample site names to interventions

    **Example:**

    .. doctest::

      >>> import jax.numpy as jnp
      >>> import numpyro
      >>> from numpyro.handlers import do, trace, seed
      >>> import numpyro.distributions as dist
      >>> def model(x):
      ...     s = numpyro.sample("s", dist.LogNormal())
      ...     z = numpyro.sample("z", dist.Normal(x, s))
      ...     return z ** 2
      >>> intervened_model = handlers.do(model, data={"z": 1.})
      >>> with trace() as exec_trace:
      ...     z_square = seed(intervened_model, 0)(1)
      >>> assert exec_trace['z']['value'] != 1.
      >>> assert not exec_trace['z']['is_observed']
      >>> assert not exec_trace['z'].get('stop', None)
      >>> assert z_square == 1
    """

    def __init__(self, fn=None, data=None):
        self.data = data
        self._intervener_id = str(id(self))
        super(do, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] != "sample":
            return
        if (
            msg.get("_intervener_id", None) != self._intervener_id
            and self.data.get(msg["name"]) is not None
        ):
            if msg.get("_intervener_id", None) is not None:
                warnings.warn(
                    "Attempting to intervene on variable {} multiple times,"
                    "this is almost certainly incorrect behavior".format(msg["name"]),
                    RuntimeWarning,
                    stacklevel=find_stack_level(),
                )
            msg["_intervener_id"] = self._intervener_id

            # split node, avoid reapplying self recursively to new node
            new_msg = msg.copy()
            apply_stack(new_msg)

            intervention = self.data.get(msg["name"])
            msg["name"] = msg["name"] + "__CF"  # mangle old name
            msg["value"] = intervention
            msg["is_observed"] = True
            msg["stop"] = True
