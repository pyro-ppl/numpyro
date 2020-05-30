import _pickle as c_pickle
import bz2
import pathlib
import time
from datetime import datetime
from functools import namedtuple
from typing import Callable

import jax
import jax.numpy as np
import jax.random
import tqdm
from jax import ops
from jax.tree_util import tree_map

from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.guides import ReinitGuide
from numpyro.infer import NUTS, MCMC
from numpyro.infer.kernels import SteinKernel
from numpyro.infer.util import transform_fn
from numpyro.util import fori_loop, ravel_pytree

# TODO
# Fix MCMC updates to work reasonably with optimizer

SVGDState = namedtuple('SVGDState', ['optim_state', 'rng_key'])


# Lots of code based on SVI interface and commonalities should be refactored
class SVGD:
    STRFTIME = "%H%M%S_%d%m%Y"
    STATE_FILE = 'state.pbz2'
    TRANSFORMS_FILE = 'transforms.pbz2'
    INV_TRANSFORMS_FILE = 'inv_transforms.pbz2'
    GUIDE_PARAM_NAMES_FILE = 'guide_param_names.pbz2'

    def __init__(self, model, guide: ReinitGuide, optim, loss, kernel_fn: SteinKernel,
                 num_particles: int = 10, loss_temperature: float = 1.0, repulsion_temperature: float = 1.0,
                 classic_guide_params_fn: Callable[[str], bool] = lambda name: False,
                 sp_mcmc_crit='infl', sp_mode='local',
                 num_mcmc_particles: int = 0, num_mcmc_warmup: int = 100, num_mcmc_updates: int = 10,
                 sampler_fn=NUTS, sampler_kwargs=None, mcmc_kwargs=None, checkpoint_dir_path='checkpoints',
                 **static_kwargs):
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
        :param repulsion_temperature: scaling of repulsive forces (Non-linear SVGD)
        :param classic_guide_param_fn: predicate on names of parameters in guide which should be optimized classically without Stein
                (E.g., parameters for large normal networks or other transformation)
        :param sp_mcmc_crit: Stein Point MCMC update selection criterion, either 'infl' for most influential or 'rand' for random
        :param sp_mode: Stein Point MCMC mode for calculating Kernelized Stein Discrepancy. Either 'local' for only the updated MCMC particles or 'global' for all particles.
        :param num_mcmc_particles: Number of particles that should be updated with Stein Point MCMC (should be a subset of number of Stein particles)
        :param num_mcmc_warmup: Number of warmup steps for the MCMC sampler
        :param num_mcmc_updates: Number of MCMC update steps at each iteration
        :param sampler_fn: The MCMC sampling kernel used for the Stein Point MCMC updates
        :param sampler_kwargs: Keyword arguments provided to the MCMC sampling kernel
        :param mcmc_kwargs: Keyword arguments provided to the MCMC interface
        :param checkpoint_dir_path: path to checkout directory
        :param static_kwargs: Static keyword arguments for the model / guide, i.e. arguments
            that remain constant during fitting.
        """
        assert sp_mcmc_crit == 'infl' or sp_mcmc_crit == 'rand'
        assert sp_mode == 'local' or sp_mode == 'global'
        assert 0 <= num_mcmc_particles <= num_particles

        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss
        self.kernel_fn = kernel_fn
        self.static_kwargs = static_kwargs
        self.num_particles = num_particles
        self.loss_temperature = loss_temperature
        self.repulsion_temperature = repulsion_temperature
        self.classic_guide_params_fn = classic_guide_params_fn
        self.sp_mcmc_crit = sp_mcmc_crit
        self.sp_mode = sp_mode
        self.num_mcmc_particles = num_mcmc_particles
        self.num_mcmc_warmup = num_mcmc_warmup
        self.num_mcmc_updates = num_mcmc_updates
        self.sampler_fn = sampler_fn
        self.sampler_kwargs = sampler_kwargs or dict()
        self.mcmc_kwargs = mcmc_kwargs or dict()
        self.checkpoint_dir_path = pathlib.Path(pathlib.Path(checkpoint_dir_path).absolute())
        self.mcmc: MCMC = None
        self.guide_param_names = None
        self.constrain_fn = None
        self.uconstrain_fn = None
        self.transform = None
        self.inv_transform = None

    def _apply_kernel(self, kernel, x, y, v):
        if self.kernel_fn.mode == 'norm' or self.kernel_fn.mode == 'vector':
            return kernel(x, y) * v
        else:
            return kernel(x, y) @ v

    def _kernel_grad(self, kernel, x, y):
        if self.kernel_fn.mode == 'norm':
            return jax.grad(lambda x: kernel(x, y))(x)
        elif self.kernel_fn.mode == 'vector':
            return jax.vmap(lambda i: jax.grad(lambda xi: kernel(xi, y[i])[i])(x[i]))(np.arange(x.shape[0]))
        else:
            return jax.vmap(lambda l: np.sum(jax.vmap(lambda m: jax.grad(lambda x: kernel(x, y)[l, m])(x)[m])
                                             (np.arange(x.shape[0]))))(np.arange(x.shape[0]))

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
        # 1. Collect each guide parameter into monolithic particles that capture correlations between parameter values across each individual particle
        stein_particles, unravel_pytree, unravel_pytree_batched = ravel_pytree(stein_uparams, batch_dims=1)
        particle_info = self._calc_particle_info(stein_uparams, stein_particles.shape[0])

        # 2. Calculate loss and gradients for each parameter
        def scaled_loss(rng_key, classic_params, stein_params):
            params = {**classic_params, **stein_params}
            loss_val = self.loss.loss(rng_key, params, handlers.scale(self.model, self.loss_temperature), self.guide,
                                      *args, **kwargs, **self.static_kwargs)
            return - loss_val

        kernel_particle_loss_fn = lambda ps: scaled_loss(rng_key, self.constrain_fn(classic_uparams),
                                                         self.constrain_fn(unravel_pytree(ps)))
        loss, particle_ljp_grads = jax.vmap(jax.value_and_grad(kernel_particle_loss_fn))(stein_particles)
        classic_param_grads = jax.vmap(lambda ps: jax.grad(lambda cps:
                                                           scaled_loss(rng_key, self.constrain_fn(cps),
                                                                       self.constrain_fn(unravel_pytree(ps))))(
            classic_uparams))(stein_particles)
        classic_param_grads = tree_map(jax.partial(np.mean, axis=0), classic_param_grads)

        # 3. Calculate kernel on monolithic particle
        kernel = self.kernel_fn.compute(stein_particles, particle_info, kernel_particle_loss_fn)

        # 4. Calculate the attractive force and repulsive force on the monolithic particles
        attractive_force = jax.vmap(lambda y: np.sum(
            jax.vmap(lambda x, x_ljp_grad: self._apply_kernel(kernel, x, y, x_ljp_grad))(stein_particles,
                                                                                         particle_ljp_grads), axis=0))(
            stein_particles)
        repulsive_force = jax.vmap(lambda y: np.sum(
            jax.vmap(lambda x: self.repulsion_temperature * self._kernel_grad(kernel, x, y))(stein_particles), axis=0))(
            stein_particles)
        particle_grads = (attractive_force + repulsive_force) / self.num_particles

        # 5. Decompose the monolithic particle forces back to concrete parameter values
        stein_param_grads = unravel_pytree_batched(particle_grads)

        # 6. Return loss and gradients (based on parameter forces)
        res_grads = tree_map(lambda x: -x, {**classic_param_grads, **stein_param_grads})
        return -np.mean(loss), res_grads

    def _score_sp_mcmc(self, rng_key, subset_idxs, stein_uparams, sp_mcmc_subset_uparams, classic_uparams,
                       *args, **kwargs):
        if self.sp_mode == 'local':
            _, ksd = self._svgd_loss_and_grads(rng_key, {**sp_mcmc_subset_uparams, **classic_uparams}, *args, **kwargs)
        else:
            stein_uparams = {p: ops.index_update(v, subset_idxs, sp_mcmc_subset_uparams[p]) for p, v in
                             stein_uparams.items()}
            _, ksd = self._svgd_loss_and_grads(rng_key, {**stein_uparams, **classic_uparams}, *args, **kwargs)
        ksd_res = np.sum(np.concatenate([np.ravel(v) for v in ksd.values()]))
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
                                                       self.model, self.guide, *args, **kwargs))
        mcmc = MCMC(sampler, self.num_mcmc_warmup, self.num_mcmc_updates, num_chains=self.num_mcmc_particles,
                    progress_bar=False, chain_method='vectorized',
                    **self.mcmc_kwargs)
        stein_params = self.constrain_fn(stein_uparams)
        stein_subset_params = {p: v[0:self.num_mcmc_particles] for p, v in stein_params.items()}
        mcmc.warmup(warmup_key, *args, init_params=stein_subset_params, **kwargs)

        # 2. Choose MCMC particles
        mcmc_key, choice_key = jax.random.split(mcmc_key)
        if self.num_mcmc_particles == self.num_particles:
            idxs = np.arange(self.num_particles)
        else:
            if self.sp_mcmc_crit == 'rand':
                idxs = jax.random.shuffle(choice_key, np.arange(self.num_particles))[:self.num_mcmc_particles]
            elif self.sp_mcmc_crit == 'infl':
                _, grads = self._svgd_loss_and_grads(choice_key, unconstr_params, *args, **kwargs)
                ksd = np.linalg.norm(
                    np.concatenate([np.reshape(grads[p], (self.num_particles, -1)) for p in stein_uparams.keys()],
                                   axis=-1),
                    ord=2, axis=-1)
                idxs = np.argsort(ksd)[:self.num_mcmc_particles]
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
                                          classic_uparams, *args, **kwargs))(np.arange(self.num_mcmc_particles))
        mcmc_idx = np.argmax(scores)
        stein_uparams = {p: ops.index_update(v, idxs, sss_uparams[p][:, mcmc_idx]) for p, v in stein_uparams.items()}
        return {**stein_uparams, **classic_uparams}

    def init(self, rng_key, *args, **kwargs):
        """
        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: initial :data:`SVGDState`
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
        guide_param_names = set()
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                transforms[site['name']] = transform.inv
                if site['name'] in guide_init_params:
                    pval, _ = guide_init_params[site['name']]
                    if self.classic_guide_params_fn(site['name']):
                        pval = tree_map(lambda x: x[0], pval)
                else:
                    pval = site['value']
                params[site['name']] = transform.inv(pval)
                if site['name'] in guide_trace:
                    guide_param_names.add(site['name'])

        self._set_model_guide_attrs(guide_param_names, transforms, inv_transforms)

        return SVGDState(self.optim.init(params), rng_key)

    def _set_model_guide_attrs(self, guide_param_names, transforms, inv_transforms):
        self.guide_param_names = guide_param_names
        self.transforms = transforms
        self.inv_transforms = inv_transforms
        self.constrain_fn = jax.partial(transform_fn, inv_transforms)
        self.uconstrain_fn = jax.partial(transform_fn, transforms)

    def get_params(self, state):
        """
        Gets values at `param` sites of the `model` and `guide`.
        :param svi_state: current state of the optimizer.
        """
        params = self.constrain_fn(self.optim.get_params(state.optim_state))
        return params

    def store_checkout(self, state):
        """
        Create checkpoint with current state of optimizer.

        :param state: current state of the optimizer.
        """
        ch_dir = self.checkpoint_dir_path
        ch_dir.mkdir(exist_ok=True)

        ts = datetime.utcnow().strftime(SVGD.STRFTIME)
        (ch_dir / ts).mkdir()

        print(self.get_params(state))

        with bz2.open(ch_dir / ts / SVGD.STATE_FILE, 'w') as f:
            c_pickle.dump(self.get_params(state), f)
        with bz2.open(ch_dir / ts / SVGD.TRANSFORMS_FILE, 'w') as f:
            c_pickle.dump(self.transforms, f)
        with bz2.open(ch_dir / ts / SVGD.INV_TRANSFORMS_FILE, 'w') as f:
            c_pickle.dump(self.inv_transforms, f)
        with bz2.open(ch_dir / ts / SVGD.GUIDE_PARAM_NAMES_FILE, 'w') as f:
            c_pickle.dump(self.guide_param_names, f)

        print(f"Checkpoint at {ts} created!")

    def load_latest_checkout(self, rng_key):
        """
        Restore current state of optimization for latest checkpoint.

        :param rng_key: current state of the optimizer.
        """
        checkpoints = list(self.checkpoint_dir_path.iterdir())
        assert checkpoints, 'No checkpoints available!'

        checkpoints.sort(key=lambda ts: time.mktime(time.strptime(ts.name, SVGD.STRFTIME)), reverse=True)
        latest = checkpoints[0]

        with bz2.BZ2File(latest / SVGD.STATE_FILE) as f:
            raw_state = c_pickle.load(f)
            state = SVGDState(self.optim.init(raw_state), rng_key)

        with bz2.BZ2File(latest / SVGD.TRANSFORMS_FILE) as f:
            transforms = c_pickle.load(f)

        with bz2.BZ2File(latest / SVGD.INV_TRANSFORMS_FILE) as f:
            inv_transforms = c_pickle.load(f)

        with bz2.BZ2File(latest / SVGD.GUIDE_PARAM_NAMES_FILE) as f:
            guide_param_names = c_pickle.load(f)

        print(f"Loaded checkpoint from {latest.name}!")

        self._set_model_guide_attrs(guide_param_names, transforms, inv_transforms)

        return state

    def update(self, state, *args, **kwargs):
        """
        Take a single step of SVGD (possibly on a batch / minibatch of data),
        using the optimizer.
        :param state: current state of SVGD.
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
        return SVGDState(optim_state, rng_key), loss_val

    def run(self, rng_key, num_steps, *args, return_last=True, progbar=True, **kwargs):
        def bodyfn(i, info):
            svgd_state, losses = info
            svgd_state, loss = self.update(svgd_state, *args, **kwargs)
            losses = ops.index_update(losses, i, loss)
            return svgd_state, losses

        svgd_state = self.init(rng_key, *args, **kwargs)
        losses = np.empty((num_steps,))
        if not progbar:
            svgd_state, losses = fori_loop(0, num_steps, bodyfn, (svgd_state, losses))
        else:
            with tqdm.trange(num_steps) as t:
                for i in t:
                    svgd_state, losses = jax.jit(bodyfn)(i, (svgd_state, losses))
                    t.set_description('SVGD {:.5}'.format(losses[i]), refresh=False)
                    t.update()
        loss_res = losses[-1] if return_last else losses
        return svgd_state, loss_res

    def evaluate(self, state, *args, **kwargs):
        """
        Take a single step of SVGD (possibly on a batch / minibatch of data).
        :param state: current state of SVGD.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate loss given the current parameter values (held within `state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given a state
        _, rng_key_eval = jax.random.split(state.rng_key)
        params = self.get_params(state)
        loss_val, _ = self._svgd_loss_and_grads(rng_key_eval, params,
                                                *args, **kwargs, **self.static_kwargs)
        return loss_val

    def predict(self, state, *args, num_samples=1, **kwargs):
        def predict_model(rng_key, params):
            guide_trace = handlers.trace(handlers.substitute(handlers.seed(self.guide, rng_key), params)
                                         ).get_trace(*args, **kwargs)
            model_trace = handlers.trace(handlers.replay(
                handlers.substitute(handlers.seed(self.model, rng_key), params), guide_trace)
            ).get_trace(*args, **kwargs)
            return {name: site['value'] for name, site in model_trace.items() if ('is_observed' not in site) or not site['is_observed']}

        _, rng_key_predict = jax.random.split(state.rng_key)
        params = self.get_params(state)
        classic_params = {p: v for p, v in params.items() if
                          p not in self.guide_param_names or self.classic_guide_params_fn(p)}
        stein_params = {p: v for p, v in params.items() if p not in classic_params}
        if num_samples == 1:
            return jax.vmap(lambda sp: predict_model(rng_key_predict, {**sp, **classic_params}))(stein_params)
        else:
            return jax.vmap(lambda rk: jax.vmap(lambda sp: predict_model(rk, {**sp, **classic_params}))(stein_params))(
                jax.random.split(rng_key_predict))
