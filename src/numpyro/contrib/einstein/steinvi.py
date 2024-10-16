# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from collections.abc import Callable
from copy import deepcopy
import functools
from functools import partial
from itertools import chain
import operator

import jax
from jax import grad, jacfwd, numpy as jnp, random, vmap
from jax.flatten_util import ravel_pytree

from numpyro import handlers
from numpyro.contrib.einstein.stein_kernels import SteinKernel
from numpyro.contrib.einstein.stein_loss import SteinLoss
from numpyro.contrib.einstein.stein_util import (
    batch_ravel_pytree,
    get_parameter_transform,
)
from numpyro.contrib.funsor import config_enumerate, enum
from numpyro.distributions import Distribution
from numpyro.distributions.transforms import IdentityTransform
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.util import _guess_max_plate_nesting, transform_fn
from numpyro.optim import _NumPyroOptim
from numpyro.util import fori_collect

SteinVIState = namedtuple("SteinVIState", ["optim_state", "rng_key"])
SteinVIRunResult = namedtuple("SteinRunResult", ["params", "state", "losses"])


def _numel(shape):
    return functools.reduce(operator.mul, shape, 1)


class SteinVI:
    """Variational inference with Stein mixtures.


    **Example:**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.distributions import constraints
        >>> from numpyro.contrib.einstein import MixtureGuidePredictive, SteinVI, RBFKernel

        >>> def model(data):
        ...     f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
        ...     with numpyro.plate("N", data.shape[0] if data is not None else 10):
        ...         numpyro.sample("obs", dist.Bernoulli(f), obs=data)

        >>> def guide(data):
        ...     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
        ...     beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
        ...                            constraint=constraints.positive)
        ...     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
        >>> optimizer = numpyro.optim.Adam(step_size=0.0005)
        >>> stein = SteinVI(model, guide, optimizer, kernel_fn=RBFKernel())
        >>> stein_result = stein.run(random.PRNGKey(0), 2000, data)
        >>> params = stein_result.params
        >>> # use guide to make predictive
        >>> predictive = MixtureGuidePredictive(model, guide, params, num_samples=1000, guide_sites=stein.guide_sites)
        >>> samples = predictive(random.PRNGKey(1), data=None)

    :param Callable model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param _NumPyroOptim optim: An instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param SteinKernel kernel_fn: Function that produces a logarithm of the statistical kernel to use with Stein mixture
        inference.
    :param num_stein_particles: Number of particles (i.e., mixture components) in the Stein mixture.
    :param num_elbo_particles: Number of Monte Carlo draws used to approximate the attractive force gradient.
        (More particles give better gradient approximations)
    :param Float loss_temperature: Scaling factor of the attractive force.
    :param Float repulsion_temperature: Scaling factor of the repulsive force (Non-linear Stein)
    :param Callable non_mixture_guide_param_fn: predicate on names of parameters in guide which should be optimized
        classically without Stein (E.g. parameters for large normal networks or other transformation)
    :param static_kwargs: Static keyword arguments for the model / guide, i.e. arguments that remain constant
        during inference.
    """

    def __init__(
        self,
        model: Callable,
        guide: Callable,
        optim: _NumPyroOptim,
        kernel_fn: SteinKernel,
        num_stein_particles: int = 10,
        num_elbo_particles: int = 10,
        loss_temperature: float = 1.0,
        repulsion_temperature: float = 1.0,
        non_mixture_guide_params_fn: Callable[[str], bool] = lambda name: False,
        enum=True,
        **static_kwargs,
    ):
        if isinstance(guide, AutoGuide):
            not_comptaible_guides = [
                "AutoIAFNormal",
                "AutoBNAFNormal",
                "AutoDAIS",
                "AutoSemiDAIS",
                "AutoSurrogateLikelihoodDAIS",
            ]
            guide_name = guide.__class__.__name__
            assert guide_name not in not_comptaible_guides, (
                f"SteinVI currently not compatible with {guide_name}. "
                f"If you have a use case, feel free to open an issue."
            )

            init_loc_error_message = (
                "SteinVI is not compatible with init_to_feasible, init_to_value, "
                "and init_to_uniform with radius=0. If you have a use case, "
                "feel free to open an issue."
            )
            if isinstance(guide.init_loc_fn, partial):
                init_fn_name = guide.init_loc_fn.func.__name__
                if init_fn_name == "init_to_uniform":
                    assert (
                        guide.init_loc_fn.keywords.get("radius", None) != 0
                    ), init_loc_error_message
            else:
                init_fn_name = guide.init_loc_fn.__name__
            assert init_fn_name not in [
                "init_to_feasible",
                "init_to_value",
            ], init_loc_error_message

        self._inference_model = model
        self.model = model
        self.guide = guide
        self._init_guide = deepcopy(guide)
        self.optim = optim
        self.stein_loss = SteinLoss(  # TODO: @OlaRonning handle enum
            elbo_num_particles=num_elbo_particles,
            stein_num_particles=num_stein_particles,
        )
        self.kernel_fn = kernel_fn
        self.static_kwargs = static_kwargs
        self.num_stein_particles = num_stein_particles
        self.loss_temperature = loss_temperature
        self.repulsion_temperature = repulsion_temperature
        self.enum = enum
        self.non_mixture_params_fn = non_mixture_guide_params_fn
        self.guide_sites = None
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
            return grad(lambda x: kernel(x, y))(x)
        elif self.kernel_fn.mode == "vector":
            return vmap(lambda i: grad(lambda x: kernel(x, y)[i])(x)[i])(
                jnp.arange(x.shape[0])
            )
        else:
            return vmap(
                lambda a: jnp.sum(
                    vmap(lambda b: grad(lambda x: kernel(x, y)[a, b])(x)[b])(
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

    def _find_init_params(self, particle_seed, inner_guide, model_args, model_kwargs):
        def local_trace(key):
            guide = deepcopy(inner_guide)

            with handlers.seed(rng_seed=key), handlers.trace() as mixture_trace:
                guide(*model_args, **model_kwargs)

            init_params = {
                name: site["value"]
                for name, site in mixture_trace.items()
                if site.get("type") == "param"
            }
            return init_params

        return vmap(local_trace)(random.split(particle_seed, self.num_stein_particles))

    def _svgd_loss_and_grads(self, rng_key, unconstr_params, *args, **kwargs):
        # 0. Separate model and guide parameters, since only guide parameters are updated using Stein
        non_mixture_uparams = {  # Includes any marked guide parameters and all model parameters
            p: v
            for p, v in unconstr_params.items()
            if p not in self.guide_sites or self.non_mixture_params_fn(p)
        }
        stein_uparams = {
            p: v for p, v in unconstr_params.items() if p not in non_mixture_uparams
        }

        # 1. Collect each guide parameter into monolithic particles that capture correlations
        # between parameter values across each individual particle
        stein_particles, unravel_pytree, unravel_pytree_batched = batch_ravel_pytree(
            stein_uparams, nbatch_dims=1
        )
        particle_info, _ = self._calc_particle_info(
            stein_uparams, stein_particles.shape[0]
        )
        attractive_key, classic_key = random.split(rng_key)

        # 2. Calculate gradients for each particle
        def kernel_particles_loss_fn(
            rng_key, particles
        ):  # TODO: rewrite using def to utilize jax caching
            particle_keys = random.split(rng_key, self.stein_loss.stein_num_particles)
            grads = vmap(
                lambda i: grad(
                    lambda particle: (
                        vmap(
                            lambda elbo_key: self.stein_loss.single_particle_loss(
                                rng_key=elbo_key,
                                model=handlers.scale(
                                    self._inference_model, self.loss_temperature
                                ),
                                guide=self.guide,
                                selected_particle=unravel_pytree(particle),
                                unravel_pytree=unravel_pytree,
                                flat_particles=particles,
                                select_index=i,
                                model_args=args,
                                model_kwargs=kwargs,
                                param_map=self.constrain_fn(non_mixture_uparams),
                            )
                        )(
                            random.split(
                                particle_keys[i], self.stein_loss.elbo_num_particles
                            )
                        )
                    ).mean()
                )(particles[i])
            )(jnp.arange(self.stein_loss.stein_num_particles))

            return grads

        def particle_transform_fn(particle):
            params = unravel_pytree(particle)

            tparams = self.particle_transform_fn(params)
            ctparams = self.constrain_fn(tparams)
            tparticle, _ = ravel_pytree(tparams)
            ctparticle, _ = ravel_pytree(ctparams)
            return tparticle, ctparticle

        # 2.1 Lift particles to constraint space
        tstein_particles, ctstein_particles = vmap(particle_transform_fn)(
            stein_particles
        )

        # 2.2 Compute particle gradients (for attractive force)
        particle_ljp_grads = kernel_particles_loss_fn(attractive_key, ctstein_particles)

        # 2.2 Compute non-mixture parameter gradients
        non_mixture_param_grads = grad(
            lambda cps: -self.stein_loss.loss(
                classic_key,
                self.constrain_fn(cps),
                handlers.scale(self._inference_model, self.loss_temperature),
                self.guide,
                unravel_pytree_batched(ctstein_particles),
                *args,
                **kwargs,
            )
        )(non_mixture_uparams)

        # 3. Calculate kernel of particles
        kernel = self.kernel_fn.compute(
            stein_particles, particle_info, kernel_particles_loss_fn
        )

        # 4. Calculate the attractive force and repulsive force on the particles
        attractive_force = vmap(
            lambda y: jnp.sum(
                vmap(
                    lambda x, x_ljp_grad: self._apply_kernel(kernel, x, y, x_ljp_grad)
                )(tstein_particles, particle_ljp_grads),
                axis=0,
            )
        )(tstein_particles)
        repulsive_force = vmap(
            lambda y: jnp.sum(
                vmap(
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
                return jacfwd(self.particle_transforms[var_name].inv)(var)

            def _update_force(attr_force, rep_force, jac):
                force = attr_force.reshape(-1) + rep_force.reshape(-1)
                if jac is not None:
                    force = force @ jac.reshape(
                        (_numel(jac.shape[: len(jac.shape) // 2]), -1)
                    )
                return force.reshape(attr_force.shape)

            reparam_jac = {
                name: jax.tree.map(lambda var: _nontrivial_jac(name, var), variables)
                for name, variables in unravel_pytree(particle).items()
            }
            jac_params = jax.tree.map(
                _update_force,
                unravel_pytree(attr_forces),
                unravel_pytree(rep_forces),
                reparam_jac,
            )
            jac_particle, _ = ravel_pytree(jac_params)
            return jac_particle

        particle_grads = (
            vmap(single_particle_grad)(
                stein_particles, attractive_force, repulsive_force
            )
            / self.num_stein_particles
        )

        # 5. Decompose the monolithic particle forces back to concrete parameter values
        stein_param_grads = unravel_pytree_batched(particle_grads)

        # 6. Return loss and gradients (based on parameter forces)
        res_grads = jax.tree.map(
            lambda x: -x, {**non_mixture_param_grads, **stein_param_grads}
        )
        return jnp.linalg.norm(particle_grads), res_grads

    def init(self, rng_key, *args, **kwargs):
        """Register random variable transformations, constraints and determine initialize positions of the particles.

        :param rng_key: Random number generator seed.
        :param args: Arguments to the model / guide.
        :param kwargs: Keyword arguments to the model / guide.
        :return: initial :data:`SteinVIState`
        """

        rng_key, kernel_seed, model_seed, guide_seed, particle_seed = random.split(
            rng_key, 5
        )

        model_init = handlers.seed(self.model, model_seed)
        model_trace = handlers.trace(model_init).get_trace(
            *args, **kwargs, **self.static_kwargs
        )

        guide_init_params = self._find_init_params(
            particle_seed, self._init_guide, args, kwargs
        )

        guide_init = handlers.seed(self.guide, guide_seed)
        guide_trace = handlers.trace(guide_init).get_trace(
            *args, **kwargs, **self.static_kwargs
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
                    pval = guide_init_params[site["name"]]
                    if self.non_mixture_params_fn(site["name"]):
                        pval = jax.tree.map(lambda x: x[0], pval)
                else:
                    pval = site["value"]
                params[site["name"]] = transform.inv(pval)
                if site["name"] in guide_trace:
                    guide_param_names.add(site["name"])

        if should_enum:
            mpn = _guess_max_plate_nesting(model_trace)
            self._inference_model = enum(config_enumerate(self.model), -mpn - 1)
        self.guide_sites = guide_param_names
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
        rng_key, rng_key_mcmc, rng_key_step = random.split(state.rng_key, num=3)
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
            diagnostics_fn=lambda state: f"norm Stein force: {state[1]:.3f}"
            if progress_bar
            else None,
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
        :return: normed stein force given the current parameter values (held within `state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given a state
        _, _, rng_key_eval = random.split(state.rng_key, num=3)
        params = self.optim.get_params(state.optim_state)
        normed_stein_force, _ = self._svgd_loss_and_grads(
            rng_key_eval, params, *args, **kwargs, **self.static_kwargs
        )
        return normed_stein_force
