# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import functools
from functools import partial
from itertools import chain
import operator
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random
from jax.tree_util import tree_map

from numpyro import handlers
from numpyro.contrib.einstein.kernels import SteinKernel
from numpyro.contrib.einstein.util import batch_ravel_pytree, get_parameter_transform
from numpyro.contrib.funsor import config_enumerate, enum
from numpyro.distributions import Distribution, Normal
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import IdentityTransform
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.util import _guess_max_plate_nesting, transform_fn
from numpyro.util import fori_collect, ravel_pytree

SteinVIState = namedtuple("SteinVIState", ["optim_state", "rng_key"])
SteinVIRunResult = namedtuple("SteinRunResult", ["params", "state", "losses"])


def _numel(shape):
    return functools.reduce(operator.mul, shape, 1)


class SteinVI:
    """Stein variational inference for stein mixtures.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param kernel_fn: Function that produces a logarithm of the statistical kernel to use with Stein inference
    :param num_particles: number of particles for Stein inference.
        (More particles capture more of the posterior distribution)
    :param loss_temperature: scaling of loss factor
    :param repulsion_temperature: scaling of repulsive forces (Non-linear Stein)
    :param enum: whether to apply automatic marginalization of discrete variables
    :param classic_guide_param_fn: predicate on names of parameters in guide which should be optimized classically
                                   without Stein (E.g. parameters for large normal networks or other transformation)
    :param static_kwargs: Static keyword arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    """

    def __init__(
        self,
        model,
        guide,
        optim,
        loss,
        kernel_fn: SteinKernel,
        num_particles: int = 10,
        loss_temperature: float = 1.0,
        repulsion_temperature: float = 1.0,
        classic_guide_params_fn: Callable[[str], bool] = lambda name: False,
        enum=True,
        **static_kwargs,
    ):
        self._inference_model = model
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss
        self.kernel_fn = kernel_fn
        self.static_kwargs = static_kwargs
        self.num_particles = num_particles
        self.loss_temperature = loss_temperature
        self.repulsion_temperature = repulsion_temperature
        self.enum = enum
        self.classic_guide_params_fn = classic_guide_params_fn
        self.guide_param_names = None
        self.constrain_fn = None
        self.uconstrain_fn = None
        self.particle_transform_fn = None
        self.particle_transforms = None

    def _apply_kernel(self, kernel, x, y, v):
        if self.kernel_fn.mode == "norm" or self.kernel_fn.mode == "vector":
            return kernel(x, y) * v
        else:
            return kernel(x, y) @ v

    def _kernel_grad(self, kernel, x, y):
        if self.kernel_fn.mode == "norm":
            return jax.grad(lambda x: kernel(x, y))(x)
        elif self.kernel_fn.mode == "vector":
            return jax.vmap(lambda i: jax.grad(lambda x: kernel(x, y)[i])(x)[i])(
                jnp.arange(x.shape[0])
            )
        else:
            return jax.vmap(
                lambda a: jnp.sum(
                    jax.vmap(lambda b: jax.grad(lambda x: kernel(x, y)[a, b])(x)[b])(
                        jnp.arange(x.shape[0])
                    )
                )
            )(jnp.arange(x.shape[0]))

    def _param_size(self, param):
        if isinstance(param, tuple) or isinstance(param, list):
            return sum(map(self._param_size, param))
        return param.size

    def _calc_particle_info(self, uparams, num_particles, start_index=0):
        uparam_keys = list(uparams.keys())
        uparam_keys.sort()
        res = {}
        end_index = start_index
        for k in uparam_keys:
            if isinstance(uparams[k], dict):
                res_sub, end_index = self._calc_particle_info(
                    uparams[k], num_particles, start_index
                )
                res[k] = res_sub
            else:
                end_index = start_index + self._param_size(uparams[k]) // num_particles
                res[k] = (start_index, end_index)
            start_index = end_index
        return res, end_index

    def _find_init_params(self, particle_seed, inner_guide, inner_guide_trace):
        def extract_info(site):
            nonlocal particle_seed
            name = site["name"]
            value = site["value"]
            constraint = site["kwargs"].get("constraint", real)
            transform = get_parameter_transform(site)
            if (
                isinstance(inner_guide, AutoGuide)
                and "_".join((inner_guide.prefix, "loc")) in name
            ):
                site_key, particle_seed = jax.random.split(particle_seed)
                unconstrained_shape = transform.inverse_shape(value.shape)
                init_value = jnp.expand_dims(
                    transform.inv(value), 0
                ) + Normal(  # Add gaussian noise
                    scale=0.1
                ).sample(
                    particle_seed, (self.num_particles, *unconstrained_shape)
                )
                init_value = transform(init_value)

            else:
                site_fn = site["fn"]
                site_args = site["args"]
                site_key, particle_seed = jax.random.split(particle_seed)

                def _reinit(seed):
                    with handlers.seed(rng_seed=seed):
                        return site_fn(*site_args)

                init_value = jax.vmap(_reinit)(
                    jax.random.split(particle_seed, self.num_particles)
                )
            return init_value, constraint

        init_params = {
            name: extract_info(site)
            for name, site in inner_guide_trace.items()
            if site.get("type") == "param"
        }
        return init_params

    def _svgd_loss_and_grads(self, rng_key, unconstr_params, *args, **kwargs):
        # 0. Separate model and guide parameters, since only guide parameters are updated using Stein
        classic_uparams = {
            p: v
            for p, v in unconstr_params.items()
            if p not in self.guide_param_names or self.classic_guide_params_fn(p)
        }
        stein_uparams = {
            p: v for p, v in unconstr_params.items() if p not in classic_uparams
        }
        # 1. Collect each guide parameter into monolithic particles that capture correlations
        # between parameter values across each individual particle
        stein_particles, unravel_pytree, unravel_pytree_batched = batch_ravel_pytree(
            stein_uparams, nbatch_dims=1
        )
        particle_info, _ = self._calc_particle_info(
            stein_uparams, stein_particles.shape[0]
        )

        # 2. Calculate loss and gradients for each parameter
        def scaled_loss(rng_key, classic_params, stein_params):
            params = {**classic_params, **stein_params}
            loss_val = self.loss.loss(
                rng_key,
                params,
                handlers.scale(self._inference_model, self.loss_temperature),
                self.guide,
                *args,
                **kwargs,
            )
            return -loss_val

        def kernel_particle_loss_fn(ps):
            return scaled_loss(
                rng_key,
                self.constrain_fn(classic_uparams),
                self.constrain_fn(unravel_pytree(ps)),
            )

        def particle_transform_fn(particle):
            params = unravel_pytree(particle)

            tparams = self.particle_transform_fn(params)
            tparticle, _ = ravel_pytree(tparams)
            return tparticle

        tstein_particles = jax.vmap(particle_transform_fn)(stein_particles)

        loss, particle_ljp_grads = jax.vmap(
            jax.value_and_grad(kernel_particle_loss_fn)
        )(tstein_particles)
        classic_param_grads = jax.vmap(
            lambda ps: jax.grad(
                lambda cps: scaled_loss(
                    rng_key,
                    self.constrain_fn(cps),
                    self.constrain_fn(unravel_pytree(ps)),
                )
            )(classic_uparams)
        )(stein_particles)
        classic_param_grads = tree_map(partial(jnp.mean, axis=0), classic_param_grads)

        # 3. Calculate kernel on monolithic particle
        kernel = self.kernel_fn.compute(
            stein_particles, particle_info, kernel_particle_loss_fn
        )

        # 4. Calculate the attractive force and repulsive force on the monolithic particles
        attractive_force = jax.vmap(
            lambda y: jnp.sum(
                jax.vmap(
                    lambda x, x_ljp_grad: self._apply_kernel(kernel, x, y, x_ljp_grad)
                )(tstein_particles, particle_ljp_grads),
                axis=0,
            )
        )(tstein_particles)
        repulsive_force = jax.vmap(
            lambda y: jnp.sum(
                jax.vmap(
                    lambda x: self.repulsion_temperature
                    * self._kernel_grad(kernel, x, y)
                )(tstein_particles),
                axis=0,
            )
        )(tstein_particles)

        def single_particle_grad(particle, attr_forces, rep_forces):
            def _nontrivial_jac(var_name, var):
                if isinstance(self.particle_transforms[var_name], IdentityTransform):
                    return None
                return jax.jacfwd(self.particle_transforms[var_name].inv)(var)

            def _update_force(attr_force, rep_force, jac):
                force = attr_force.reshape(-1) + rep_force.reshape(-1)
                if jac is not None:
                    force = force @ jac.reshape(
                        (_numel(jac.shape[: len(jac.shape) // 2]), -1)
                    )
                return force.reshape(attr_force.shape)

            reparam_jac = {
                name: tree_map(lambda var: _nontrivial_jac(name, var), variables)
                for name, variables in unravel_pytree(particle).items()
            }
            jac_params = tree_map(
                _update_force,
                unravel_pytree(attr_forces),
                unravel_pytree(rep_forces),
                reparam_jac,
            )
            jac_particle, _ = ravel_pytree(jac_params)
            return jac_particle

        particle_grads = (
            jax.vmap(single_particle_grad)(
                stein_particles, attractive_force, repulsive_force
            )
            / self.num_particles
        )

        # 5. Decompose the monolithic particle forces back to concrete parameter values
        stein_param_grads = unravel_pytree_batched(particle_grads)

        # 6. Return loss and gradients (based on parameter forces)
        res_grads = tree_map(lambda x: -x, {**classic_param_grads, **stein_param_grads})
        return -jnp.mean(loss), res_grads

    def init(self, rng_key, *args, **kwargs):
        """
        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: initial :data:`SteinVIState`
        """
        rng_key, kernel_seed, model_seed, guide_seed = jax.random.split(rng_key, 4)
        model_init = handlers.seed(self.model, model_seed)
        guide_init = handlers.seed(self.guide, guide_seed)
        guide_trace = handlers.trace(guide_init).get_trace(
            *args, **kwargs, **self.static_kwargs
        )
        model_trace = handlers.trace(model_init).get_trace(
            *args, **kwargs, **self.static_kwargs
        )
        rng_key, particle_seed = jax.random.split(rng_key)
        guide_init_params = self._find_init_params(
            particle_seed, self.guide, guide_trace
        )
        params = {}
        transforms = {}
        inv_transforms = {}
        particle_transforms = {}
        guide_param_names = set()
        should_enum = False
        for site in model_trace.values():
            if (
                "fn" in site
                and site["type"] == "sample"
                and not site["is_observed"]
                and isinstance(site["fn"], Distribution)
                and site["fn"].is_discrete
            ):
                if site["fn"].has_enumerate_support and self.enum:
                    should_enum = True
                else:
                    raise Exception(
                        "Cannot enumerate model with discrete variables without enumerate support"
                    )
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in chain(model_trace.values(), guide_trace.values()):
            if site["type"] == "param":
                transform = get_parameter_transform(site)
                inv_transforms[site["name"]] = transform
                transforms[site["name"]] = transform.inv
                particle_transforms[site["name"]] = site.get(
                    "particle_transform", IdentityTransform()
                )
                if site["name"] in guide_init_params:
                    pval, _ = guide_init_params[site["name"]]
                    if self.classic_guide_params_fn(site["name"]):
                        pval = tree_map(lambda x: x[0], pval)
                else:
                    pval = site["value"]
                params[site["name"]] = transform.inv(pval)
                if site["name"] in guide_trace:
                    guide_param_names.add(site["name"])

        if should_enum:
            mpn = _guess_max_plate_nesting(model_trace)
            self._inference_model = enum(config_enumerate(self.model), -mpn - 1)
        self.guide_param_names = guide_param_names
        self.constrain_fn = partial(transform_fn, inv_transforms)
        self.uconstrain_fn = partial(transform_fn, transforms)
        self.particle_transforms = particle_transforms
        self.particle_transform_fn = partial(transform_fn, particle_transforms)
        stein_particles, _, _ = batch_ravel_pytree(
            {
                k: params[k]
                for k, site in guide_trace.items()
                if site["type"] == "param" and site["name"] in guide_init_params
            },
            nbatch_dims=1,
        )

        self.kernel_fn.init(kernel_seed, stein_particles.shape)
        return SteinVIState(self.optim.init(params), rng_key)

    def get_params(self, state: SteinVIState):
        """
        Gets values at `param` sites of the `model` and `guide`.
        :param state: current state of the optimizer.
        """
        params = self.constrain_fn(self.optim.get_params(state.optim_state))
        return params

    def update(self, state: SteinVIState, *args, **kwargs):
        """
        Take a single step of Stein (possibly on a batch / minibatch of data),
        using the optimizer.
        :param state: current state of Stein.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(state, loss)`.
        """
        rng_key, rng_key_mcmc, rng_key_step = jax.random.split(state.rng_key, num=3)
        params = self.optim.get_params(state.optim_state)
        optim_state = state.optim_state
        loss_val, grads = self._svgd_loss_and_grads(
            rng_key_step, params, *args, **kwargs, **self.static_kwargs
        )
        optim_state = self.optim.update(grads, optim_state)
        return SteinVIState(optim_state, rng_key), loss_val

    def run(
        self,
        rng_key,
        num_steps,
        *args,
        progress_bar=True,
        init_state=None,
        collect_fn=lambda val: val[1],  # TODO: refactor
        **kwargs,
    ):
        def bodyfn(_i, info):
            body_state = info[0]
            return (*self.update(body_state, *info[2:], **kwargs), *info[2:])

        if init_state is None:
            state = self.init(rng_key, *args, **kwargs)
        else:
            state = init_state
        loss = self.evaluate(state, *args, **kwargs)
        auxiliaries, last_res = fori_collect(
            0,
            num_steps,
            lambda info: bodyfn(0, info),
            (state, loss, *args),
            progbar=progress_bar,
            transform=collect_fn,
            return_last_val=True,
        )
        state = last_res[0]
        return SteinVIRunResult(self.get_params(state), state, auxiliaries)

    def evaluate(self, state, *args, **kwargs):
        """
        Take a single step of Stein (possibly on a batch / minibatch of data).
        :param state: current state of Stein.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate loss given the current parameter values (held within `state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given a state
        _, _, rng_key_eval = jax.random.split(state.rng_key, num=3)
        params = self.optim.get_params(state.optim_state)
        loss_val, _ = self._svgd_loss_and_grads(
            rng_key_eval, params, *args, **kwargs, **self.static_kwargs
        )
        return loss_val
