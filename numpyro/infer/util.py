# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from contextlib import contextmanager
from functools import partial
import warnings

import numpy as np

from jax import device_get, jacfwd, lax, random, value_and_grad
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpyro
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.distributions.util import is_identically_one, sum_rightmost
from numpyro.handlers import condition, replay, seed, substitute, trace
from numpyro.infer.initialization import init_to_uniform, init_to_value
from numpyro.util import not_jax_tracer, soft_vmap, while_loop

__all__ = [
    "find_valid_initial_params",
    "get_potential_fn",
    "log_density",
    "log_likelihood",
    "potential_energy",
    "initialize_model",
    "Predictive",
]

ModelInfo = namedtuple(
    "ModelInfo", ["param_info", "potential_fn", "postprocess_fn", "model_trace"]
)
ParamInfo = namedtuple("ParamInfo", ["z", "potential_energy", "z_grad"])


def log_density(model, model_args, model_kwargs, params):
    """
    (EXPERIMENTAL INTERFACE) Computes log of joint density for the model given
    latent values ``params``.

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :return: log of joint density and a corresponding model trace
    """
    model = substitute(model, data=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = jnp.zeros(())
    for site in model_trace.values():
        if site["type"] == "sample":
            value = site["value"]
            intermediates = site["intermediates"]
            scale = site["scale"]
            if intermediates:
                log_prob = site["fn"].log_prob(value, intermediates)
            else:
                log_prob = site["fn"].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            log_prob = jnp.sum(log_prob)
            log_joint = log_joint + log_prob
    return log_joint, model_trace


class _without_rsample_stop_gradient(numpyro.primitives.Messenger):
    """
    Stop gradient for samples at latent sample sites for which has_rsample=False.
    """

    def postprocess_message(self, msg):
        if (
            msg["type"] == "sample"
            and (not msg["is_observed"])
            and (not msg["fn"].has_rsample)
        ):
            msg["value"] = lax.stop_gradient(msg["value"])
            # TODO: reconsider this logic
            # here we clear all the cached value so that gradients of log_prob(value) w.r.t.
            # all parameters of the transformed distributions match the behavior of
            # TransformedDistribution(d, transform) in Pyro with transform.cache_size == 0
            msg["intermediates"] = None


def get_importance_trace(model, guide, args, kwargs, params):
    """
    (EXPERIMENTAL) Returns traces from the guide and the model that is run against it.
    The returned traces also store the log probability at each site.

    .. note:: Gradients are blocked at latent sites which do not have reparametrized samplers.
    """
    guide = substitute(guide, data=params)
    with _without_rsample_stop_gradient():
        guide_trace = trace(guide).get_trace(*args, **kwargs)
    model = substitute(replay(model, guide_trace), data=params)
    model_trace = trace(model).get_trace(*args, **kwargs)
    for tr in (guide_trace, model_trace):
        for site in tr.values():
            if site["type"] == "sample":
                if "log_prob" not in site:
                    value = site["value"]
                    intermediates = site["intermediates"]
                    scale = site["scale"]
                    if intermediates:
                        log_prob = site["fn"].log_prob(value, intermediates)
                    else:
                        log_prob = site["fn"].log_prob(value)

                    if (scale is not None) and (not is_identically_one(scale)):
                        log_prob = scale * log_prob
                    site["log_prob"] = log_prob
    return model_trace, guide_trace


def transform_fn(transforms, params, invert=False):
    """
    (EXPERIMENTAL INTERFACE) Callable that applies a transformation from the `transforms`
    dict to values in the `params` dict and returns the transformed values keyed on
    the same names.

    :param transforms: Dictionary of transforms keyed by names. Names in
        `transforms` and `params` should align.
    :param params: Dictionary of arrays keyed by names.
    :param invert: Whether to apply the inverse of the transforms.
    :return: `dict` of transformed params.
    """
    if invert:
        transforms = {k: v.inv for k, v in transforms.items()}
    return {k: transforms[k](v) if k in transforms else v for k, v in params.items()}


def constrain_fn(model, model_args, model_kwargs, params, return_deterministic=False):
    """
    (EXPERIMENTAL INTERFACE) Gets value at each latent site in `model` given
    unconstrained parameters `params`. The `transforms` is used to transform these
    unconstrained parameters to base values of the corresponding priors in `model`.
    If a prior is a transformed distribution, the corresponding base value lies in
    the support of base distribution. Otherwise, the base value lies in the support
    of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: dictionary of unconstrained values keyed by site
        names.
    :param bool return_deterministic: whether to return the value of `deterministic`
        sites from the model. Defaults to `False`.
    :return: `dict` of transformed params.
    """

    def substitute_fn(site):
        if site["name"] in params:
            if site["type"] == "sample":
                with helpful_support_errors(site):
                    return biject_to(site["fn"].support)(params[site["name"]])
            else:
                return params[site["name"]]

    substituted_model = substitute(model, substitute_fn=substitute_fn)
    model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
    return {
        k: v["value"]
        for k, v in model_trace.items()
        if (k in params) or (return_deterministic and (v["type"] == "deterministic"))
    }


def _unconstrain_reparam(params, site):
    name = site["name"]
    if name in params:
        p = params[name]
        support = site["fn"].support
        with helpful_support_errors(site):
            t = biject_to(support)
        # in scan, we might only want to substitute an item at index i, rather than the whole sequence
        i = site["infer"].get("_scan_current_index", None)
        if i is not None:
            event_dim_shift = t.codomain.event_dim - t.domain.event_dim
            expected_unconstrained_dim = len(site["fn"].shape()) - event_dim_shift
            # check if p has additional time dimension
            if jnp.ndim(p) > expected_unconstrained_dim:
                p = p[i]

        if support in [constraints.real, constraints.real_vector]:
            return p
        value = t(p)

        log_det = t.log_abs_det_jacobian(p, value)
        log_det = sum_rightmost(
            log_det, jnp.ndim(log_det) - jnp.ndim(value) + len(site["fn"].event_shape)
        )
        if site["scale"] is not None:
            log_det = site["scale"] * log_det
        numpyro.factor("_{}_log_det".format(name), log_det)
        return value


def potential_energy(model, model_args, model_kwargs, params, enum=False):
    """
    (EXPERIMENTAL INTERFACE) Computes potential energy of a model given unconstrained params.
    Under the hood, we will transform these unconstrained parameters to the values
    belong to the supports of the corresponding priors in `model`.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict params: unconstrained parameters of `model`.
    :param bool enum: whether to enumerate over discrete latent sites.
    :return: potential energy given unconstrained parameters.
    """
    if enum:
        from numpyro.contrib.funsor import log_density as log_density_
    else:
        log_density_ = log_density

    substituted_model = substitute(
        model, substitute_fn=partial(_unconstrain_reparam, params)
    )
    # no param is needed for log_density computation because we already substitute
    log_joint, model_trace = log_density_(
        substituted_model, model_args, model_kwargs, {}
    )
    return -log_joint


def _init_to_unconstrained_value(site=None, values={}):
    if site is None:
        return partial(_init_to_unconstrained_value, values=values)


def find_valid_initial_params(
    rng_key,
    model,
    *,
    init_strategy=init_to_uniform,
    enum=False,
    model_args=(),
    model_kwargs=None,
    prototype_params=None,
    forward_mode_differentiation=False,
    validate_grad=True,
):
    """
    (EXPERIMENTAL INTERFACE) Given a model with Pyro primitives, returns an initial
    valid unconstrained value for all the parameters. This function also returns
    the corresponding potential energy, the gradients, and an
    `is_valid` flag to say whether the initial parameters are valid. Parameter values
    are considered valid if the values and the gradients for the log density have
    finite values.

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param callable init_strategy: a per-site initialization function.
    :param bool enum: whether to enumerate over discrete latent sites.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param dict prototype_params: an optional prototype parameters, which is used
        to define the shape for initial parameters.
    :param bool forward_mode_differentiation: whether to use forward-mode differentiation
        or reverse-mode differentiation. Defaults to False.
    :param bool validate_grad: whether to validate gradient of the initial params.
        Defaults to True.
    :return: tuple of `init_params_info` and `is_valid`, where `init_params_info` is the tuple
        containing the initial params, their potential energy, and their gradients.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    init_strategy = (
        init_strategy if isinstance(init_strategy, partial) else init_strategy()
    )
    # handle those init strategies differently to save computation
    if init_strategy.func is init_to_uniform:
        radius = init_strategy.keywords.get("radius")
        init_values = {}
    elif init_strategy.func is _init_to_unconstrained_value:
        radius = 2
        init_values = init_strategy.keywords.get("values")
    else:
        radius = None

    def cond_fn(state):
        i, _, _, is_valid = state
        return (i < 100) & (~is_valid)

    def body_fn(state):
        i, key, _, _ = state
        key, subkey = random.split(key)

        if radius is None or prototype_params is None:
            # XXX: we don't want to apply enum to draw latent samples
            model_ = model
            if enum:
                from numpyro.contrib.funsor import enum as enum_handler

                if isinstance(model, substitute) and isinstance(model.fn, enum_handler):
                    model_ = substitute(model.fn.fn, data=model.data)
                elif isinstance(model, enum_handler):
                    model_ = model.fn

            # Wrap model in a `substitute` handler to initialize from `init_loc_fn`.
            seeded_model = substitute(seed(model_, subkey), substitute_fn=init_strategy)
            model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
            constrained_values, inv_transforms = {}, {}
            for k, v in model_trace.items():
                if (
                    v["type"] == "sample"
                    and not v["is_observed"]
                    and not v["fn"].is_discrete
                ):
                    constrained_values[k] = v["value"]
                    with helpful_support_errors(v):
                        inv_transforms[k] = biject_to(v["fn"].support)
            params = transform_fn(
                inv_transforms,
                {k: v for k, v in constrained_values.items()},
                invert=True,
            )
        else:  # this branch doesn't require tracing the model
            params = {}
            for k, v in prototype_params.items():
                if k in init_values:
                    params[k] = init_values[k]
                else:
                    params[k] = random.uniform(
                        subkey, jnp.shape(v), minval=-radius, maxval=radius
                    )
                    key, subkey = random.split(key)

        potential_fn = partial(
            potential_energy, model, model_args, model_kwargs, enum=enum
        )
        if validate_grad:
            if forward_mode_differentiation:
                pe = potential_fn(params)
                z_grad = jacfwd(potential_fn)(params)
            else:
                pe, z_grad = value_and_grad(potential_fn)(params)
            z_grad_flat = ravel_pytree(z_grad)[0]
            is_valid = jnp.isfinite(pe) & jnp.all(jnp.isfinite(z_grad_flat))
        else:
            pe = potential_fn(params)
            is_valid = jnp.isfinite(pe)
            z_grad = None

        return i + 1, key, (params, pe, z_grad), is_valid

    def _find_valid_params(rng_key, exit_early=False):
        init_state = (0, rng_key, (prototype_params, 0.0, prototype_params), False)
        if exit_early and not_jax_tracer(rng_key):
            # Early return if valid params found. This is only helpful for single chain,
            # where we can avoid compiling body_fn in while_loop.
            _, _, (init_params, pe, z_grad), is_valid = init_state = body_fn(init_state)
            if not_jax_tracer(is_valid):
                if device_get(is_valid):
                    return (init_params, pe, z_grad), is_valid

        # XXX: this requires compiling the model, so for multi-chain, we trace the model 2-times
        # even if the init_state is a valid result
        _, _, (init_params, pe, z_grad), is_valid = while_loop(
            cond_fn, body_fn, init_state
        )
        return (init_params, pe, z_grad), is_valid

    # Handle possible vectorization
    if rng_key.ndim == 1:
        (init_params, pe, z_grad), is_valid = _find_valid_params(
            rng_key, exit_early=True
        )
    else:
        (init_params, pe, z_grad), is_valid = lax.map(_find_valid_params, rng_key)
    return (init_params, pe, z_grad), is_valid


def _get_model_transforms(model, model_args=(), model_kwargs=None):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    inv_transforms = {}
    # model code may need to be replayed in the presence of deterministic sites
    replay_model = False
    has_enumerate_support = False
    for k, v in model_trace.items():
        if v["type"] == "sample" and not v["is_observed"]:
            if v["fn"].is_discrete:
                has_enumerate_support = True
                if not v["fn"].has_enumerate_support:
                    raise RuntimeError(
                        "MCMC only supports continuous sites or discrete sites "
                        f"with enumerate support, but got {type(v['fn']).__name__}."
                    )
            else:
                support = v["fn"].support
                with helpful_support_errors(v):
                    inv_transforms[k] = biject_to(support)
                # XXX: the following code filters out most situations with dynamic supports
                args = ()
                if isinstance(support, constraints._GreaterThan):
                    args = ("lower_bound",)
                elif isinstance(support, constraints._Interval):
                    args = ("lower_bound", "upper_bound")
                for arg in args:
                    if not isinstance(getattr(support, arg), (int, float)):
                        replay_model = True
        elif v["type"] == "deterministic":
            replay_model = True
    return inv_transforms, replay_model, has_enumerate_support, model_trace


def get_potential_fn(
    model,
    inv_transforms,
    *,
    enum=False,
    replay_model=False,
    dynamic_args=False,
    model_args=(),
    model_kwargs=None,
):
    """
    (EXPERIMENTAL INTERFACE) Given a model with Pyro primitives, returns a
    function which, given unconstrained parameters, evaluates the potential
    energy (negative log joint density). In addition, this returns a
    function to transform unconstrained values at sample sites to constrained
    values within their respective support.

    :param model: Python callable containing Pyro primitives.
    :param dict inv_transforms: dictionary of transforms keyed by names.
    :param bool enum: whether to enumerate over discrete latent sites.
    :param bool replay_model: whether we need to replay model in
        `postprocess_fn` to obtain `deterministic` sites.
    :param bool dynamic_args: if `True`, the `potential_fn` and
        `constraints_fn` are themselves dependent on model arguments.
        When provided a `*model_args, **model_kwargs`, they return
        `potential_fn` and `constraints_fn` callables, respectively.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :return: tuple of (`potential_fn`, `postprocess_fn`). The latter is used
        to constrain unconstrained samples (e.g. those returned by HMC)
        to values that lie within the site's support, and return values at
        `deterministic` sites in the model.
    """
    if dynamic_args:

        def potential_fn(*args, **kwargs):
            return partial(potential_energy, model, args, kwargs, enum=enum)

        def postprocess_fn(*args, **kwargs):
            if replay_model:
                # XXX: we seed to sample discrete sites (but not collect them)
                model_ = seed(model.fn, 0) if enum else model
                return partial(
                    constrain_fn, model_, args, kwargs, return_deterministic=True
                )
            else:
                return partial(transform_fn, inv_transforms)

    else:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        potential_fn = partial(
            potential_energy, model, model_args, model_kwargs, enum=enum
        )
        if replay_model:
            model_ = seed(model.fn, 0) if enum else model
            postprocess_fn = partial(
                constrain_fn,
                model_,
                model_args,
                model_kwargs,
                return_deterministic=True,
            )
        else:
            postprocess_fn = partial(transform_fn, inv_transforms)

    return potential_fn, postprocess_fn


def _guess_max_plate_nesting(model_trace):
    """
    Guesses max_plate_nesting by using model trace.
    This optimistically assumes static model
    structure.
    """
    sites = [site for site in model_trace.values() if site["type"] == "sample"]

    dims = [
        frame.dim
        for site in sites
        for frame in site["cond_indep_stack"]
        if frame.dim is not None
    ]
    max_plate_nesting = -min(dims) if dims else 0
    return max_plate_nesting


# TODO: follow pyro.util.check_site_shape logics for more complete validation
def _validate_model(model_trace):
    # XXX: this validates plate statements under `enum`
    sites = [site for site in model_trace.values() if site["type"] == "sample"]

    for site in sites:
        batch_dims = len(site["fn"].batch_shape)
        if site.get("_control_flow_done", False):
            batch_dims = batch_dims - 1  # remove time dimension under scan
        plate_dims = -min([0] + [frame.dim for frame in site["cond_indep_stack"]])
        assert (
            plate_dims >= batch_dims
        ), "Missing plate statement for batch dimensions at site {}".format(
            site["name"]
        )


def initialize_model(
    rng_key,
    model,
    *,
    init_strategy=init_to_uniform,
    dynamic_args=False,
    model_args=(),
    model_kwargs=None,
    forward_mode_differentiation=False,
    validate_grad=True,
):
    """
    (EXPERIMENTAL INTERFACE) Helper function that calls :func:`~numpyro.infer.util.get_potential_fn`
    and :func:`~numpyro.infer.util.find_valid_initial_params` under the hood
    to return a tuple of (`init_params_info`, `potential_fn`, `postprocess_fn`, `model_trace`).

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param bool dynamic_args: if `True`, the `potential_fn` and
        `constraints_fn` are themselves dependent on model arguments.
        When provided a `*model_args, **model_kwargs`, they return
        `potential_fn` and `constraints_fn` callables, respectively.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs: kwargs provided to the model.
    :param bool forward_mode_differentiation: whether to use forward-mode differentiation
        or reverse-mode differentiation. By default, we use reverse mode but the forward
        mode can be useful in some cases to improve the performance. In addition, some
        control flow utility on JAX such as `jax.lax.while_loop` or `jax.lax.fori_loop`
        only supports forward-mode differentiation. See
        `JAX's The Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_
        for more information.
    :param bool validate_grad: whether to validate gradient of the initial params.
        Defaults to True.
    :return: a namedtupe `ModelInfo` which contains the fields
        (`param_info`, `potential_fn`, `postprocess_fn`, `model_trace`), where
        `param_info` is a namedtuple `ParamInfo` containing values from the prior
        used to initiate MCMC, their corresponding potential energy, and their gradients;
        `postprocess_fn` is a callable that uses inverse transforms
        to convert unconstrained HMC samples to constrained values that
        lie within the site's support, in addition to returning values
        at `deterministic` sites in the model.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    substituted_model = substitute(
        seed(model, rng_key if jnp.ndim(rng_key) == 1 else rng_key[0]),
        substitute_fn=init_strategy,
    )
    (
        inv_transforms,
        replay_model,
        has_enumerate_support,
        model_trace,
    ) = _get_model_transforms(substituted_model, model_args, model_kwargs)
    # substitute param sites from model_trace to model so
    # we don't need to generate again parameters of `numpyro.module`
    model = substitute(
        model,
        data={
            k: site["value"]
            for k, site in model_trace.items()
            if site["type"] in ["param"]
        },
    )
    constrained_values = {
        k: v["value"]
        for k, v in model_trace.items()
        if v["type"] == "sample" and not v["is_observed"] and not v["fn"].is_discrete
    }

    if has_enumerate_support:
        from numpyro.contrib.funsor import config_enumerate, enum

        if not isinstance(model, enum):
            max_plate_nesting = _guess_max_plate_nesting(model_trace)
            _validate_model(model_trace)
            model = enum(config_enumerate(model), -max_plate_nesting - 1)

    potential_fn, postprocess_fn = get_potential_fn(
        model,
        inv_transforms,
        replay_model=replay_model,
        enum=has_enumerate_support,
        dynamic_args=dynamic_args,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )

    init_strategy = (
        init_strategy if isinstance(init_strategy, partial) else init_strategy()
    )
    if (init_strategy.func is init_to_value) and not replay_model:
        init_values = init_strategy.keywords.get("values")
        unconstrained_values = transform_fn(inv_transforms, init_values, invert=True)
        init_strategy = _init_to_unconstrained_value(values=unconstrained_values)
    prototype_params = transform_fn(inv_transforms, constrained_values, invert=True)
    (init_params, pe, grad), is_valid = find_valid_initial_params(
        rng_key,
        substitute(
            model,
            data={
                k: site["value"]
                for k, site in model_trace.items()
                if site["type"] in ["plate"]
            },
        ),
        init_strategy=init_strategy,
        enum=has_enumerate_support,
        model_args=model_args,
        model_kwargs=model_kwargs,
        prototype_params=prototype_params,
        forward_mode_differentiation=forward_mode_differentiation,
        validate_grad=validate_grad,
    )

    if not_jax_tracer(is_valid):
        if device_get(~jnp.all(is_valid)):
            with numpyro.validation_enabled(), trace() as tr:
                # validate parameters
                substituted_model(*model_args, **model_kwargs)
                # validate values
                for site in tr.values():
                    if site["type"] == "sample":
                        with warnings.catch_warnings(record=True) as ws:
                            site["fn"]._validate_sample(site["value"])
                        if len(ws) > 0:
                            for w in ws:
                                # at site information to the warning message
                                w.message.args = (
                                    "Site {}: {}".format(
                                        site["name"], w.message.args[0]
                                    ),
                                ) + w.message.args[1:]
                                warnings.showwarning(
                                    w.message,
                                    w.category,
                                    w.filename,
                                    w.lineno,
                                    file=w.file,
                                    line=w.line,
                                )
            raise RuntimeError(
                "Cannot find valid initial parameters. Please check your model again."
            )
    return ModelInfo(
        ParamInfo(init_params, pe, grad), potential_fn, postprocess_fn, model_trace
    )


def _predictive(
    rng_key,
    model,
    posterior_samples,
    batch_shape,
    return_sites=None,
    infer_discrete=False,
    parallel=True,
    model_args=(),
    model_kwargs={},
):
    masked_model = numpyro.handlers.mask(model, mask=False)
    if infer_discrete:
        # inspect the model to get some structure
        rng_key, subkey = random.split(rng_key)
        batch_ndim = len(batch_shape)
        prototype_sample = tree_map(
            lambda x: jnp.reshape(x, (-1,) + jnp.shape(x)[batch_ndim:])[0],
            posterior_samples,
        )
        prototype_trace = trace(
            seed(substitute(masked_model, prototype_sample), subkey)
        ).get_trace(*model_args, **model_kwargs)
        first_available_dim = -_guess_max_plate_nesting(prototype_trace) - 1

    def single_prediction(val):
        rng_key, samples = val
        if infer_discrete:
            from numpyro.contrib.funsor import config_enumerate
            from numpyro.contrib.funsor.discrete import _sample_posterior

            model_trace = prototype_trace
            temperature = 1
            pred_samples = _sample_posterior(
                config_enumerate(condition(model, samples)),
                first_available_dim,
                temperature,
                rng_key,
                *model_args,
                **model_kwargs,
            )
        else:
            model_trace = trace(
                seed(substitute(masked_model, samples), rng_key)
            ).get_trace(*model_args, **model_kwargs)
            pred_samples = {name: site["value"] for name, site in model_trace.items()}

        if return_sites is not None:
            if return_sites == "":
                sites = {
                    k for k, site in model_trace.items() if site["type"] != "plate"
                }
            else:
                sites = return_sites
        else:
            sites = {
                k
                for k, site in model_trace.items()
                if (site["type"] == "sample" and k not in samples)
                or (site["type"] == "deterministic")
            }
        return {name: value for name, value in pred_samples.items() if name in sites}

    num_samples = int(np.prod(batch_shape))
    if num_samples > 1:
        rng_key = random.split(rng_key, num_samples)
    rng_key = rng_key.reshape(batch_shape + (2,))
    chunk_size = num_samples if parallel else 1
    return soft_vmap(
        single_prediction, (rng_key, posterior_samples), len(batch_shape), chunk_size
    )


class Predictive(object):
    """
    This class is used to construct predictive distribution. The predictive distribution is obtained
    by running model conditioned on latent samples from `posterior_samples`.

    .. warning::
        The interface for the `Predictive` class is experimental, and
        might change in the future.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param callable guide: optional guide to get posterior samples of sites not present
        in `posterior_samples`.
    :param dict params: dictionary of values for param sites of model/guide.
    :param int num_samples: number of samples
    :param list return_sites: sites to return; by default only sample sites not present
        in `posterior_samples` are returned.
    :param bool infer_discrete: whether or not to sample discrete sites from the
        posterior, conditioned on observations and other latent values in
        ``posterior_samples``. Under the hood, those sites will be marked with
        ``site["infer"]["enumerate"] = "parallel"``. See how `infer_discrete` works at
        the `Pyro enumeration tutorial <https://pyro.ai/examples/enumeration.html>`_.
        Note that this requires ``funsor`` installation.
    :param bool parallel: whether to predict in parallel using JAX vectorized map :func:`jax.vmap`.
        Defaults to False.
    :param batch_ndims: the number of batch dimensions in posterior samples. Some usages:

        + set `batch_ndims=0` to get prediction for 1 single sample

        + set `batch_ndims=1` to get prediction for `posterior_samples`
          with shapes `(num_samples x ...)`

        + set `batch_ndims=2` to get prediction for `posterior_samples`
          with shapes `(num_chains x N x ...)`. Note that if `num_samples`
          argument is not None, its value should be equal to `num_chains x N`.

    :return: dict of samples from the predictive distribution.
    """

    def __init__(
        self,
        model,
        posterior_samples=None,
        *,
        guide=None,
        params=None,
        num_samples=None,
        return_sites=None,
        infer_discrete=False,
        parallel=False,
        batch_ndims=1,
    ):
        if posterior_samples is None and num_samples is None:
            raise ValueError(
                "Either posterior_samples or num_samples must be specified."
            )

        posterior_samples = {} if posterior_samples is None else posterior_samples

        prototype_site = batch_shape = batch_size = None
        for name, sample in posterior_samples.items():
            if batch_shape is not None and sample.shape[:batch_ndims] != batch_shape:
                raise ValueError(
                    f"Batch shapes at site {name} and {prototype_site} "
                    f"should be the same, but got "
                    f"{sample.shape[:batch_ndims]} and {batch_shape}"
                )
            else:
                prototype_site = name
                batch_shape = sample.shape[:batch_ndims]
                batch_size = int(np.prod(batch_shape))
                if (num_samples is not None) and (num_samples != batch_size):
                    warnings.warn(
                        "Sample's batch dimension size {} is different from the "
                        "provided {} num_samples argument. Defaulting to {}.".format(
                            batch_size, num_samples, batch_size
                        ),
                        UserWarning,
                    )
                num_samples = batch_size

        if num_samples is None:
            raise ValueError(
                "No sample sites in posterior samples to infer `num_samples`."
            )

        if batch_shape is None:
            batch_shape = (1,) * (batch_ndims - 1) + (num_samples,)

        if return_sites is not None:
            assert isinstance(return_sites, (list, tuple, set))

        self.model = model
        self.posterior_samples = {} if posterior_samples is None else posterior_samples
        self.num_samples = num_samples
        self.guide = guide
        self.params = {} if params is None else params
        self.infer_discrete = infer_discrete
        self.return_sites = return_sites
        self.parallel = parallel
        self.batch_ndims = batch_ndims
        self._batch_shape = batch_shape

    def __call__(self, rng_key, *args, **kwargs):
        """
        Returns dict of samples from the predictive distribution. By default, only sample sites not
        contained in `posterior_samples` are returned. This can be modified by changing the
        `return_sites` keyword argument of this :class:`Predictive` instance.

        :param jax.random.PRNGKey rng_key: random key to draw samples.
        :param args: model arguments.
        :param kwargs: model kwargs.
        """
        posterior_samples = self.posterior_samples
        if self.guide is not None:
            rng_key, guide_rng_key = random.split(rng_key)
            # use return_sites='' as a special signal to return all sites
            guide = substitute(self.guide, self.params)
            posterior_samples = _predictive(
                guide_rng_key,
                guide,
                posterior_samples,
                self._batch_shape,
                return_sites="",
                parallel=self.parallel,
                model_args=args,
                model_kwargs=kwargs,
            )
        model = substitute(self.model, self.params)
        return _predictive(
            rng_key,
            model,
            posterior_samples,
            self._batch_shape,
            return_sites=self.return_sites,
            infer_discrete=self.infer_discrete,
            parallel=self.parallel,
            model_args=args,
            model_kwargs=kwargs,
        )


def log_likelihood(
    model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs
):
    """
    (EXPERIMENTAL INTERFACE) Returns log likelihood at observation nodes of model,
    given samples of all latent variables.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param batch_ndims: the number of batch dimensions in posterior samples. Some usages:

        + set `batch_ndims=0` to get log likelihoods for 1 single sample

        + set `batch_ndims=1` to get log likelihoods for `posterior_samples`
          with shapes `(num_samples x ...)`

        + set `batch_ndims=2` to get log likelihoods for `posterior_samples`
          with shapes `(num_chains x num_samples x ...)`

    :param kwargs: model kwargs.
    :return: dict of log likelihoods at observation sites.
    """

    def single_loglik(samples):
        substituted_model = (
            substitute(model, samples) if isinstance(samples, dict) else model
        )
        model_trace = trace(substituted_model).get_trace(*args, **kwargs)
        return {
            name: site["fn"].log_prob(site["value"])
            for name, site in model_trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

    prototype_site = batch_shape = None
    for name, sample in posterior_samples.items():
        if batch_shape is not None and jnp.shape(sample)[:batch_ndims] != batch_shape:
            raise ValueError(
                f"Batch shapes at site {name} and {prototype_site} "
                f"should be the same, but got "
                f"{sample.shape[:batch_ndims]} and {batch_shape}"
            )
        else:
            prototype_site = name
            batch_shape = jnp.shape(sample)[:batch_ndims]

    if batch_shape is None:  # posterior_samples is an empty dict
        batch_shape = (1,) * batch_ndims
        posterior_samples = np.zeros(batch_shape)

    batch_size = int(np.prod(batch_shape))
    chunk_size = batch_size if parallel else 1
    return soft_vmap(single_loglik, posterior_samples, len(batch_shape), chunk_size)


@contextmanager
def helpful_support_errors(site):
    try:
        yield
    except NotImplementedError as e:
        name = site["name"]
        support_name = repr(site["fn"].support).lower()
        if "integer" in support_name or "boolean" in support_name:
            # TODO: mention enumeration when it is supported in SVI
            raise ValueError(
                f"Continuous inference cannot handle discrete sample site '{name}'."
            )
        if "sphere" in support_name:
            raise ValueError(
                f"Continuous inference cannot handle spherical sample site '{name}'. "
                "Consider using ProjectedNormal distribution together with "
                "a reparameterizer, e.g. "
                f"numpyro.handlers.reparam(config={{'{name}': ProjectedNormalReparam()}})."
            )
        raise e from None
