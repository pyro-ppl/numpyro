from collections import namedtuple
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random
from jax import ops
from jax.tree_util import tree_map

from numpyro import handlers
from numpyro.contrib.funsor import enum, config_enumerate
from numpyro.distributions import Distribution
from numpyro.distributions.transforms import IdentityTransform
from numpyro.infer import NUTS, MCMC
from numpyro.infer.einstein.kernels import SteinKernel
from numpyro.infer.einstein.reinit_guide import ReinitGuide
from numpyro.infer.einstein.utils import get_parameter_transform
from numpyro.infer.util import transform_fn, _guess_max_plate_nesting
from numpyro.util import ravel_pytree


# TODO
# Fix MCMC updates to work reasonably with optimizer


# Lots of code based on SVI interface and commonalities should be refactored
class Stein(object):
    CurrentState = namedtuple('CurrentState', ['optim_state', 'rng_key'])

    def __init__(self, model, guide: ReinitGuide, optim, loss, kernel_fn: SteinKernel, num_particles: int = 10,
                 loss_temperature: float = 1.0, repulsion_temperature: float = 1.0,
                 classic_guide_params_fn: Callable[[str], bool] = lambda name: False,
                 enum=True, sp_mcmc_crit='infl',
                 sp_mode='local', num_mcmc_particles: int = 0, num_mcmc_warmup: int = 100, num_mcmc_updates: int = 10,
                 sampler_fn=NUTS, sampler_kwargs=None, mcmc_kwargs=None, **static_kwargs):
        """
        Stein Variational Gradient Descent for Non-parametric Inference.
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
        :param classic_guide_param_fn: predicate on names of parameters in guide which should be optimized classically without Stein (E.g., parameters for large normal networks or other transformation)
        :param sp_mcmc_crit: Stein Point MCMC update selection criterion, either 'infl' for most influential or 'rand' for random (EXPERIMENTAL)
        :param sp_mode: Stein Point MCMC mode for calculating Kernelized Stein Discrepancy. Either 'local' for only the updated MCMC particles or 'global' for all particles. (EXPERIMENTAL)
        :param num_mcmc_particles: Number of particles that should be updated with Stein Point MCMC (should be a subset of number of Stein particles) (EXPERIMENTAL)
        :param num_mcmc_warmup: Number of warmup steps for the MCMC sampler (EXPERIMENTAL)
        :param num_mcmc_updates: Number of MCMC update steps at each iteration (EXPERIMENTAL)
        :param sampler_fn: The MCMC sampling kernel used for the Stein Point MCMC updates (EXPERIMENTAL)
        :param sampler_kwargs: Keyword arguments provided to the MCMC sampling kernel (EXPERIMENTAL)
        :param mcmc_kwargs: Keyword arguments provided to the MCMC interface (EXPERIMENTAL)
        :param static_kwargs: Static keyword arguments for the model / guide, i.e. arguments
            that remain constant during fitting.
        """
        super().__init__(model, guide, optim, loss, name='Stein', **static_kwargs)
        assert sp_mcmc_crit == 'infl' or sp_mcmc_crit == 'rand'
        assert sp_mode == 'local' or sp_mode == 'global'
        assert 0 <= num_mcmc_particles <= num_particles

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
        self.sp_mcmc_crit = sp_mcmc_crit
        self.sp_mode = sp_mode
        self.num_mcmc_particles = num_mcmc_particles
        self.num_mcmc_warmup = num_mcmc_warmup
        self.num_mcmc_updates = num_mcmc_updates
        self.sampler_fn = sampler_fn
        self.sampler_kwargs = sampler_kwargs or dict()
        self.mcmc_kwargs = mcmc_kwargs or dict()
        self.mcmc: MCMC = None
        self.guide_param_names = None
        self.constrain_fn = None
        self.uconstrain_fn = None
        self.particle_transform_fn = None

    def _apply_kernel(self, kernel, x, y, v):
        if self.kernel_fn.mode == 'norm' or self.kernel_fn.mode == 'vector':
            return kernel(x, y) * v
        else:
            return kernel(x, y) @ v

    def _kernel_grad(self, kernel, x, y):
        if self.kernel_fn.mode == 'norm':
            return jax.grad(lambda x: kernel(x, y))(x)
        elif self.kernel_fn.mode == 'vector':
            return jax.vmap(lambda i: jax.grad(lambda x: kernel(x, y)[i])(x)[i])(jnp.arange(x.shape[0]))
        else:
            return jax.vmap(lambda l: jnp.sum(jax.vmap(lambda m: jax.grad(lambda x: kernel(x, y)[l, m])(x)[m])
                                              (jnp.arange(x.shape[0]))))(jnp.arange(x.shape[0]))

    def _param_size(self, param):
        if isinstance(param, tuple) or isinstance(param, list):
            return sum(map(self._param_size, param))
        return param.size

    def _calc_particle_info(self, uparams, num_particles):
        uparam_keys = list(uparams.keys())
        uparam_keys.sort()
        start_index = 0
        res = {}
        for k in uparam_keys:
            end_index = start_index + self._param_size(uparams[k]) // num_particles
            res[k] = (start_index, end_index)
            start_index = end_index
        return res

    def _svgd_loss_and_grads(self, rng_key, unconstr_params, *args, **kwargs):
        # 0. Separate model and guide parameters, since only guide parameters are updated using Stein
        classic_uparams = {p: v for p, v in unconstr_params.items() if
                           p not in self.guide_param_names or self.classic_guide_params_fn(p)}
        stein_uparams = {p: v for p, v in unconstr_params.items() if p not in classic_uparams}
        # 1. Collect each guide parameter into monolithic particles that capture correlations
        # between parameter values across each individual particle
        stein_particles, unravel_pytree = ravel_pytree(stein_uparams, batch_dims=1)
        unravel_pytree_batched = jax.vmap(unravel_pytree)
        particle_info = self._calc_particle_info(stein_uparams, stein_particles.shape[0])

        # 2. Calculate loss and gradients for each parameter
        def scaled_loss(rng_key, classic_params, stein_params):
            params = {**classic_params, **stein_params}
            loss_val = self.loss.loss(rng_key, params, handlers.scale(self._inference_model, self.loss_temperature),
                                      self.guide, *args, **kwargs)
            return - loss_val

        def kernel_particle_loss_fn(ps):
            return scaled_loss(rng_key, self.constrain_fn(classic_uparams),
                               self.constrain_fn(unravel_pytree(ps)))

        def particle_transform_fn(particle):
            params = unravel_pytree(particle)
            tparams = self.particle_transform_fn(params)
            tparticle, _ = ravel_pytree(tparams)
            return tparticle

        tstein_particles = jax.vmap(particle_transform_fn)(stein_particles)

        loss, particle_ljp_grads = jax.vmap(jax.value_and_grad(kernel_particle_loss_fn))(tstein_particles)
        classic_param_grads = jax.vmap(lambda ps: jax.grad(lambda cps:
                                                           scaled_loss(rng_key, self.constrain_fn(cps),
                                                                       self.constrain_fn(unravel_pytree(ps))))(
            classic_uparams))(stein_particles)
        classic_param_grads = tree_map(partial(jnp.mean, axis=0), classic_param_grads)

        # 3. Calculate kernel on monolithic particle
        kernel = self.kernel_fn.compute(stein_particles, particle_info, kernel_particle_loss_fn)

        # 4. Calculate the attractive force and repulsive force on the monolithic particles
        attractive_force = jax.vmap(lambda y: jnp.sum(
            jax.vmap(lambda x, x_ljp_grad: self._apply_kernel(kernel, x, y, x_ljp_grad))(tstein_particles,
                                                                                         particle_ljp_grads), axis=0))(
            tstein_particles)
        repulsive_force = jax.vmap(lambda y: jnp.sum(
            jax.vmap(lambda x: self.repulsion_temperature * self._kernel_grad(kernel, x, y))(tstein_particles),
            axis=0))(
            tstein_particles)

        def single_particle_grad(particle, att_force, rep_force):
            reparam_jac = jax.jacfwd(particle_transform_fn)(particle)
            return (att_force + rep_force) @ reparam_jac

        particle_grads = jax.vmap(single_particle_grad)(stein_particles, attractive_force,
                                                        repulsive_force) / self.num_particles

        # 5. Decompose the monolithic particle forces back to concrete parameter values
        stein_param_grads = unravel_pytree_batched(particle_grads)

        # 6. Return loss and gradients (based on parameter forces)
        res_grads = tree_map(lambda x: -x, {**classic_param_grads, **stein_param_grads})
        return -jnp.mean(loss), res_grads

    def _score_sp_mcmc(self, rng_key, subset_idxs, stein_uparams, sp_mcmc_subset_uparams, classic_uparams,
                       *args, **kwargs):
        if self.sp_mode == 'local':
            _, ksd = self._svgd_loss_and_grads(rng_key, {**sp_mcmc_subset_uparams, **classic_uparams}, *args, **kwargs)
        else:
            stein_uparams = {p: ops.index_update(v, subset_idxs, sp_mcmc_subset_uparams[p]) for p, v in
                             stein_uparams.items()}
            _, ksd = self._svgd_loss_and_grads(rng_key, {**stein_uparams, **classic_uparams}, *args, **kwargs)
        ksd_res = jnp.sum(jnp.concatenate([jnp.ravel(v) for v in ksd.values()]))
        return ksd_res

    def _sp_mcmc(self, rng_key, unconstr_params, *args, **kwargs):
        # 0. Separate classical and stein parameters
        classic_uparams = {p: v for p, v in unconstr_params.items() if
                           p not in self.guide_param_names or self.classic_guide_params_fn(p)}
        stein_uparams = {p: v for p, v in unconstr_params.items() if p not in classic_uparams}

        # 1. Run warmup on a subset of particles to tune the MCMC state
        warmup_key, mcmc_key = jax.random.split(rng_key)
        sampler = self.sampler_fn(
            potential_fn=lambda params: self.loss.loss(warmup_key, {**params, **self.constrain_fn(classic_uparams)},
                                                       self._inference_model, self.guide, *args, **kwargs))
        mcmc = MCMC(sampler, self.num_mcmc_warmup, self.num_mcmc_updates, num_chains=self.num_mcmc_particles,
                    progress_bar=False, chain_method='vectorized',
                    **self.mcmc_kwargs)
        stein_params = self.constrain_fn(stein_uparams)
        stein_subset_params = {p: v[0:self.num_mcmc_particles] for p, v in stein_params.items()}
        mcmc.warmup(warmup_key, *args, init_params=stein_subset_params, **kwargs)

        # 2. Choose MCMC particles
        mcmc_key, choice_key = jax.random.split(mcmc_key)
        if self.num_mcmc_particles == self.num_particles:
            idxs = jnp.arange(self.num_particles)
        else:
            if self.sp_mcmc_crit == 'rand':
                idxs = jax.random.shuffle(choice_key, jnp.arange(self.num_particles))[:self.num_mcmc_particles]
            elif self.sp_mcmc_crit == 'infl':
                _, grads = self._svgd_loss_and_grads(choice_key, unconstr_params, *args, **kwargs)
                ksd = jnp.linalg.norm(
                    jnp.concatenate([jnp.reshape(grads[p], (self.num_particles, -1)) for p in stein_uparams.keys()],
                                    axis=-1),
                    ord=2, axis=-1)
                idxs = jnp.argsort(ksd)[:self.num_mcmc_particles]
            else:
                assert False, "Unsupported SP MCMC criterion: {}".format(self.sp_mcmc_crit)

        # 3. Run MCMC on chosen particles
        stein_params = self.constrain_fn(stein_uparams)
        stein_subset_params = {p: v[idxs] for p, v in stein_params.items()}
        mcmc.run(mcmc_key, *args, init_params=stein_subset_params, **kwargs)
        samples_subset_stein_params = mcmc.get_samples(group_by_chain=True)
        sss_uparams = self.uconstrain_fn(samples_subset_stein_params)

        # 4. Select best MCMC iteration to update particles
        scores = jax.vmap(
            lambda i: self._score_sp_mcmc(mcmc_key, idxs, stein_uparams, {p: v[:, i] for p, v in sss_uparams.items()},
                                          classic_uparams, *args, **kwargs))(jnp.arange(self.num_mcmc_particles))
        mcmc_idx = jnp.argmax(scores)
        stein_uparams = {p: ops.index_update(v, idxs, sss_uparams[p][:, mcmc_idx]) for p, v in stein_uparams.items()}
        return {**stein_uparams, **classic_uparams}

    def init(self, rng_key, *args, **kwargs):
        """
        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: initial :data:`CurrentState`
        """
        rng_key, model_seed, guide_seed = jax.random.split(rng_key, 3)
        model_init = handlers.seed(self.model, model_seed)
        guide_init = handlers.seed(self.guide, guide_seed)
        guide_trace = handlers.trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = handlers.trace(model_init).get_trace(*args, **kwargs, **self.static_kwargs)
        rng_key, particle_seed = jax.random.split(rng_key)
        particle_seeds = jax.random.split(particle_seed, num=self.num_particles)
        self.guide.find_params(particle_seeds, *args, **kwargs,
                               **self.static_kwargs)  # Get parameter values for each particle
        guide_init_params = self.guide.init_params()
        params = {}
        transforms = {}
        inv_transforms = {}
        particle_transforms = {}
        guide_param_names = set()
        should_enum = False
        for site in model_trace.values():
            if isinstance(site['fn'], Distribution) and site['fn'].is_discrete:
                if site['fn'].has_enumerate_support and self.enum:
                    should_enum = True
                else:
                    raise Exception("Cannot enumerate model with discrete variables without enumerate support")
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                transform = get_parameter_transform(site)
                inv_transforms[site['name']] = transform
                transforms[site['name']] = transform.inv
                particle_transforms[site['name']] = site.get('particle_transform', IdentityTransform())
                if site['name'] in guide_init_params:
                    pval, _ = guide_init_params[site['name']]
                    if self.classic_guide_params_fn(site['name']):
                        pval = tree_map(lambda x: x[0], pval)
                else:
                    pval = site['value']
                params[site['name']] = transform.inv(pval)
                if site['name'] in guide_trace:
                    guide_param_names.add(site['name'])

        if should_enum:
            mpn = _guess_max_plate_nesting(model_trace)
            self._inference_model = enum(config_enumerate(self.model), - mpn - 1)
        self.guide_param_names = guide_param_names
        self.constrain_fn = partial(transform_fn, inv_transforms)
        self.uconstrain_fn = partial(transform_fn, transforms)
        self.particle_transform_fn = partial(transform_fn, particle_transforms)
        return Stein.CurrentState(self.optim.init(params), rng_key)

    def get_params(self, state):
        """
        Gets values at `param` sites of the `model` and `guide`.
        :param svi_state: current state of the optimizer.
        """
        params = self.constrain_fn(self.optim.get_params(state.optim_state))
        return params

    def update(self, state, *args, **kwargs):
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
        # Run Stein Point MCMC
        if self.num_mcmc_particles > 0:
            new_params = self._sp_mcmc(rng_key_mcmc, params, *args, **kwargs, **self.static_kwargs)
            grads = {p: new_params[p] - params[p] for p in params}
            optim_state = self.optim.update(grads, state.optim_state)
            params = self.optim.get_params(state.optim_state)
        else:
            optim_state = state.optim_state
        loss_val, grads = self._svgd_loss_and_grads(rng_key_step, params,
                                                    *args, **kwargs, **self.static_kwargs)
        optim_state = self.optim.update(grads, optim_state)
        return Stein.CurrentState(optim_state, rng_key), loss_val

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
        _, rng_key_eval = jax.random.split(state.rng_key)
        params = self.optim.get_params(state.optim_state)
        loss_val, _ = self._svgd_loss_and_grads(rng_key_eval, params,
                                                *args, **kwargs, **self.static_kwargs)
        return loss_val

    def predict(self, state, *args, num_samples=1, **kwargs):
        _, rng_key_predict = jax.random.split(state.rng_key)
        params = self.get_params(state)
        classic_params = {p: v for p, v in params.items() if
                          p not in self.guide_param_names or self.classic_guide_params_fn(p)}
        stein_params = {p: v for p, v in params.items() if p not in classic_params}
        if num_samples == 1:
            return jax.vmap(lambda sp: self._predict_model(rng_key_predict, {**sp, **classic_params}, *args, **kwargs)
                            )(stein_params)
        else:
            return jax.vmap(lambda rk: jax.vmap(lambda sp: self._predict_model(rk, {**sp, **classic_params},
                                                                               *args, **kwargs)
                                                )(stein_params))(jax.random.split(rng_key_predict, num_samples))
