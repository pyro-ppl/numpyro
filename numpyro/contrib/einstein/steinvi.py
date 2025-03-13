# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import deepcopy
import functools
from functools import partial
from itertools import chain
import operator

from jax import grad, numpy as jnp, random, tree, vmap
from jax.flatten_util import ravel_pytree
from jax.lax import scan

from numpyro import handlers
from numpyro.contrib.einstein.stein_loss import SteinLoss
from numpyro.contrib.einstein.stein_util import (
    batch_ravel_pytree,
    get_parameter_transform,
)
from numpyro.distributions import Distribution
from numpyro.infer.autoguide import AutoDelta, AutoGuide
from numpyro.infer.util import transform_fn
from numpyro.util import fori_collect

SteinVIState = namedtuple(
    "SteinVIState",
    ["optim_state", "rng_key", "loss_temperature", "repulsion_temperature"],
)
SteinVIRunResult = namedtuple("SteinRunResult", ["params", "state", "losses"])


def _numel(shape):
    return functools.reduce(operator.mul, shape, 1)


class SteinVI:
    """Variational inference with Stein mixtures inference [1].

    **Example:**

    .. doctest::

        >>> from jax import random, numpy as jnp

        >>> from numpyro import sample, param, plate
        >>> from numpyro.distributions import Beta, Bernoulli
        >>> from numpyro.distributions.constraints import positive

        >>> from numpyro.optim import Adagrad
        >>> from numpyro.contrib.einstein import MixtureGuidePredictive, SteinVI, RBFKernel

        >>> def model(data):
        ...     f = sample("fairness", Beta(10, 10))
        ...     n = data.shape[0] if data is not None else 1
        ...     with plate("N", n):
        ...         sample("obs", Bernoulli(f), obs=data)

        >>> def guide(data):
        ...     # Initialize all particles in the same point.
        ...     alpha_q = param("alpha_q", 15., constraint=positive)
        ...     # Initialize particles by sampling an Exponential distribution.
        ...     beta_q = param("beta_q",
        ...                     lambda rng_key: random.exponential(rng_key),
        ...                     constraint=positive)
        ...     sample("fairness", Beta(alpha_q, beta_q))

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])

        >>> opt = Adagrad(step_size=0.05)
        >>> k = RBFKernel()
        >>> stein = SteinVI(model, guide, opt, k, num_stein_particles=2)

        >>> stein_result = stein.run(random.PRNGKey(0), 200, data)
        >>> params = stein_result.params

        >>> # Use guide to make predictions.
        >>> predictive = MixtureGuidePredictive(model, guide, params, num_samples=10, guide_sites=stein.guide_sites)
        >>> samples = predictive(random.PRNGKey(1), data=None)

    :param Callable model: Python callable with NumPyro primitives for the model.
    :param Callable guide: Python callable with NumPyro primitives for the guide.
    :param _NumPyroOptim optim: An instance of :class:`~numpyro.optim._NumpyroOptim`.
        Adagrad should be preferred over Adam [1].
    :param SteinKernel kernel_fn: Function that computes the reproducing kernel to use with Stein mixture
        inference. We currently recommend :class:`~numpyro.contrib.einstein.RBFKernel`.
        This may change as criteria for kernel selection are not well understood yet.
    :param num_stein_particles: Number of particles (i.e., mixture components) in the mixture approximation.
        Default is `10`.
    :param num_elbo_particles: Number of Monte Carlo draws used to approximate the attractive force gradient.
        More particles give better gradient approximations. Default is `10`.
    :param Float loss_temperature: Scaling factor of the attractive force. Default is `1`.
    :param Float repulsion_temperature: Scaling factor of the repulsive force [2].
        We recommend not scaling the repulsion. Default is `1`.
    :param Callable non_mixture_guide_param_fn: Predicate on names of parameters in the guide which should be optimized
        using one particle. This could be parameters for large normal networks or other transformation.
        Default excludes all parameters from this option.
    :param static_kwargs: Static keyword arguments for the model and guide. These arguments cannot change
        during inference.

    **References:**

    1. Rønning, Ola, et al. "ELBOing Stein: Variational Bayes with Stein Mixture Inference."
        arXiv preprint arXiv:2410.22948 (2024).
    2. Liu, Chang, et al. "Understanding and Accelerating Particle-Based Variational Inference."
        International Conference on Machine Learning. PMLR, 2019.
    3. Wang, Dilin, and Qiang Liu. "Nonlinear Stein Variational Gradient Descent for Learning Diversified Mixture Models."
        International Conference on Machine Learning. PMLR, 2019.
    """  # noqa: E501

    def __init__(
        self,
        model,
        guide,
        optim,
        kernel_fn,
        num_stein_particles=10,
        num_elbo_particles=10,
        loss_temperature=1.0,
        repulsion_temperature=1.0,
        non_mixture_guide_params_fn=lambda name: False,
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
                    assert guide.init_loc_fn.keywords.get("radius", None) != 0.0, (
                        init_loc_error_message
                    )
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

    def _svgd_loss_and_grads(
        self,
        rng_key,
        unconstr_params,
        loss_temperature,
        repulsion_temperature,
        *args,
        **kwargs,
    ):
        # Separate model and guide parameters, since only guide parameters are updated using Stein
        # Split parameters into model and guide components - only unflagged guide parameters are
        # optimized via Stein forces.
        nonmix_uparams = {  # Includes any marked guide parameters and all model parameters
            p: v
            for p, v in unconstr_params.items()
            if p not in self.guide_sites or self.non_mixture_params_fn(p)
        }

        stein_uparams = {
            p: v for p, v in unconstr_params.items() if p not in nonmix_uparams
        }

        # Collect guide parameters into a monolithic particle.
        stein_particles, unravel_pytree, unravel_pytree_batched = batch_ravel_pytree(
            stein_uparams, nbatch_dims=1
        )

        # Kernel behavior varies based on particle site locations. The particle_info dictionary
        # maps site names to their corresponding dimensional ranges as (start, end) tuples.
        particle_info, _ = self._calc_particle_info(
            stein_uparams, stein_particles.shape[0]
        )

        def particle_transform_fn(particle):
            params = unravel_pytree(particle)
            ctparams = self.constrain_fn(self.particle_transform_fn(params))
            ctparticle, _ = ravel_pytree(ctparams)
            return ctparticle

        model = handlers.scale(self._inference_model, loss_temperature)

        def stein_loss_fn(key, particle, particle_idx):
            return self.stein_loss.particle_loss(
                rng_key=key,
                model=model,
                guide=self.guide,
                # Stein particles evolve in unconstrained space, but gradient computations must account
                # for the transformation to constrained space
                selected_particle=self.constrain_fn(unravel_pytree(particle)),
                unravel_pytree=unravel_pytree,
                flat_particles=vmap(particle_transform_fn)(stein_particles),
                select_index=particle_idx,
                model_args=args,
                model_kwargs=kwargs,
                param_map=self.constrain_fn(nonmix_uparams),
            )

        kernel = self.kernel_fn.compute(
            rng_key, stein_particles, particle_info, stein_loss_fn
        )
        attractive_key, classic_key = random.split(rng_key)

        def compute_attr_force(rng_key, particles):
            # Second term of eq. 9 from https://arxiv.org/pdf/2410.22948.
            def body(attr_force, state, y):
                key, x, i = state
                x_grad = grad(stein_loss_fn, argnums=1)(key, x, i)
                attr_force = attr_force + self._apply_kernel(kernel, x, y, x_grad)
                return attr_force, None

            particle_keys = random.split(rng_key, self.stein_loss.stein_num_particles)
            init = jnp.zeros_like(particles[0])
            idxs = jnp.arange(self.num_stein_particles)

            attr_force, _ = vmap(
                lambda y, key: scan(
                    partial(body, y=y),
                    init,
                    (random.split(key, self.num_stein_particles), particles, idxs),
                )
            )(particles, particle_keys)

            return attr_force

        attractive_force = compute_attr_force(attractive_key, stein_particles)

        # Third term of eq. 9 from https://arxiv.org/pdf/2410.22948.
        repulsive_force = vmap(
            lambda y: jnp.mean(
                vmap(lambda x: repulsion_temperature * self._kernel_grad(kernel, x, y))(
                    stein_particles
                ),
                axis=0,
            )
        )(stein_particles)

        particle_grads = attractive_force + repulsive_force

        # Compute non-mixture parameter gradients.
        nonmix_uparam_grads = grad(
            lambda cps: -self.stein_loss.loss(
                classic_key,
                self.constrain_fn(cps),
                model,
                self.guide,
                unravel_pytree_batched(vmap(particle_transform_fn)(stein_particles)),
                *args,
                **kwargs,
            )
        )(nonmix_uparams)

        # Decompose the monolithic particle forces back to concrete parameter values.
        stein_uparam_grads = unravel_pytree_batched(particle_grads)

        # Return loss and gradients (based on parameter forces).
        res_grads = tree.map(
            lambda x: -x, {**nonmix_uparam_grads, **stein_uparam_grads}
        )
        return jnp.linalg.norm(particle_grads), res_grads

    def init(self, rng_key, *args, **kwargs):
        """Register random variable transformations, constraints and determine initialize positions of the particles.

        :param jax.random.PRNGKey rng_key: Random number generator seed.
        :param args: Positional arguments to the model and guide.
        :param kwargs: Keyword arguments to the model and guide.
        :return: Initial :data:`SteinVIState`.
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
        for site in model_trace.values():
            if (
                "fn" in site
                and site["type"] == "sample"
                and not site["is_observed"]
                and isinstance(site["fn"], Distribution)
                and site["fn"].is_discrete
            ):
                if site["fn"].has_enumerate_support:
                    raise Exception(
                        "Cannot enumerate model with discrete variables without enumerate support"
                    )
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in chain(model_trace.values(), guide_trace.values()):
            if site["type"] == "param":
                transform = get_parameter_transform(site)
                inv_transforms[site["name"]] = transform
                transforms[site["name"]] = transform.inv
                particle_transforms[site["name"]] = transform
                if site["name"] in guide_init_params:
                    pval = guide_init_params[site["name"]]
                    if self.non_mixture_params_fn(site["name"]):
                        pval = tree.map(lambda x: x[0], pval)
                else:
                    pval = site["value"]
                params[site["name"]] = transform.inv(pval)
                if site["name"] in guide_trace:
                    guide_param_names.add(site["name"])

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
        return SteinVIState(
            self.optim.init(params),
            rng_key,
            self.loss_temperature,
            self.repulsion_temperature,
        )

    def get_params(self, state: SteinVIState):
        """Gets values at `param` sites of the `model` and `guide`.

        :param SteinVIState state: Current state of optimization.
        :return: Constraint parameters (i.e., particles).
        """
        params = self.constrain_fn(self.optim.get_params(state.optim_state))
        return params

    def update(self, state: SteinVIState, *args, **kwargs) -> SteinVIState:
        """Take a single step of SteinVI using the optimizer. We recommend using
        the run method instead of update.

        :param SteinVIState state: Current state of inference.
        :param args: Position arguments to the model and guide.
        :param kwargs: Keyword arguments to the model and guide.
        :return: next :data:`SteinVIState`
        """
        rng_key, rng_key_mcmc, rng_key_step = random.split(state.rng_key, num=3)
        params = self.optim.get_params(state.optim_state)
        optim_state = state.optim_state
        loss_val, grads = self._svgd_loss_and_grads(
            rng_key_step,
            params,
            state.loss_temperature,
            state.repulsion_temperature,
            *args,
            **kwargs,
            **self.static_kwargs,
        )
        optim_state = self.optim.update(grads, optim_state, value=loss_val)
        return SteinVIState(
            optim_state, rng_key, state.loss_temperature, state.repulsion_temperature
        ), loss_val

    def setup_run(self, rng_key, num_steps, args, init_state, kwargs):
        if init_state is None:
            state = self.init(rng_key, *args, **kwargs)
        else:
            assert isinstance(init_state, ASVGDState), (
                "The init_state much be an instance of ASVGDState"
            )
            state = init_state

        loss = self.evaluate(state, *args, **kwargs)

        info_init = (state, loss)

        def step(info):
            state, loss = info
            return self.update(state, *args, **kwargs)  # uses closure!

        def collect(info):
            _, loss = info
            return loss

        def extract(info):
            state, _ = info
            return state

        def diagnostic(info):
            _, loss = info
            return f"Stein force {loss:.2f}."

        return step, diagnostic, collect, extract, info_init

    def run(
        self,
        rng_key,
        num_steps,
        *args,
        progress_bar=True,
        init_state=None,
        **kwargs,
    ):
        """Run SteinVI inference.

        :param jax.random.PRNGKey rng_key: Random number generator seed.
        :param int num_steps: Number of steps to optimize.
        :param *args: Positional arguments to the model and guide.
        :param bool progress_bar: Use a progress bar. Default is `True`.
            Inference is faster with `False`.
        :param SteinVIState init_state: Initial state of inference.
            Default is ``None``, which will initialize using init before running inference.
        :param **kwargs: Keyword arguments to the model and guide.
        """
        step, diagnostic, collect, extract, init_info = self.setup_run(
            rng_key, num_steps, args, init_state, kwargs
        )

        auxiliaries, last_res = fori_collect(
            0,
            num_steps,
            step,
            init_info,
            progbar=progress_bar,
            transform=collect,
            return_last_val=True,
            diagnostics_fn=diagnostic if progress_bar else None,
        )

        state = extract(last_res)
        return SteinVIRunResult(self.get_params(state), state, auxiliaries)

    def evaluate(self, state: SteinVIState, *args, **kwargs):
        """Take a single step of Stein (possibly on a batch / minibatch of data).

        :param SteinVIState state: Current state of inference.
        :param args: Positional arguments to the model and guide.
        :param kwargs: Keyword arguments to the model and guide.
        :return: Normed Stein force given by :data:`SteinVIState`.
        """
        # we split to have the same seed as `update_fn` given a state
        _, _, rng_key_eval = random.split(state.rng_key, num=3)
        params = self.optim.get_params(state.optim_state)
        normed_stein_force, _ = self._svgd_loss_and_grads(
            rng_key_eval,
            params,
            state.loss_temperature,
            state.repulsion_temperature,
            *args,
            **kwargs,
            **self.static_kwargs,
        )
        return normed_stein_force


class SVGD(SteinVI):
    """Stein variational gradient descent [1].

    **Example:**

    .. doctest::

        >>> from jax import random, numpy as jnp

        >>> from numpyro import sample, param, plate
        >>> from numpyro.distributions import Beta, Bernoulli
        >>> from numpyro.distributions.constraints import positive

        >>> from numpyro.optim import Adagrad
        >>> from numpyro.contrib.einstein import SVGD, RBFKernel
        >>> from numpyro.infer import Predictive

        >>> def model(data):
        ...     f = sample("fairness", Beta(10, 10))
        ...     n = data.shape[0] if data is not None else 1
        ...     with plate("N", n):
        ...         sample("obs", Bernoulli(f), obs=data)

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])

        >>> opt = Adagrad(step_size=0.05)
        >>> k = RBFKernel()
        >>> svgd = SVGD(model, opt, k, num_stein_particles=2)

        >>> svgd_result = svgd.run(random.PRNGKey(0), 200, data)

        >>> params = svgd_result.params
        >>> predictive = Predictive(model, guide=svgd.guide, params=params, num_samples=10, batch_ndims=1)
        >>> samples = predictive(random.PRNGKey(1), data=None)

    :param Callable model: Python callable with NumPyro primitives for the model.
    :param Callable guide: Python callable with NumPyro primitives for the guide.
    :param _NumPyroOptim optim: An instance of :class:`~numpyro.optim._NumpyroOptim`.
        Adagrad should be preferred over Adam [1].
    :param SteinKernel kernel_fn: Function that computes the reproducing kernel to use with SVGD.
        We currently recommend :class:`~numpyro.contrib.einstein.RBFKernel`. This may change as criteria for
        kernel selection are not well understood yet.
    :param num_stein_particles: Number of particles (i.e., mixture components) in the mixture approximation.
        Default is 10.
    :param Dict guide_kwargs: Keyword arguments for :class:`~numpyro.infer.autoguide.AutoDelta`.
        Default behaviour is the same as the default for :class:`~numpyro.infer.autoguide.AutoDelta`.

        Usage::

            opt = Adagrad(step_size=0.05)
            k = RBFKernel()
            svgd = SVGD(model, opt, k, guide_kwargs={'init_loc_fn': partial(init_to_uniform, radius=0.1)})

    :param Dict static_kwargs: Static keyword arguments for the model and guide. These arguments cannot
        change during inference.

    **References:**

    1. Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm."
        Advances in neural information processing systems 29 (2016).
    """

    def __init__(
        self,
        model,
        optim,
        kernel_fn,
        num_stein_particles=10,
        guide_kwargs={},
        **static_kwargs,
    ):
        super().__init__(
            model=model,
            guide=AutoDelta(model, **guide_kwargs),
            optim=optim,
            kernel_fn=kernel_fn,
            num_stein_particles=num_stein_particles,
            # With a Delta guide we only need one draw
            # per particle to get its contribution to the expectation.
            num_elbo_particles=1,
            loss_temperature=1.0 / float(num_stein_particles),
            # For SVGD repulsion temperature != 1 changes the
            # target posterior so we keep it fixed at 1.
            repulsion_temperature=1.0,
            non_mixture_guide_params_fn=lambda name: False,
            **static_kwargs,
        )


ASVGDState = namedtuple(
    "ASVGDState",
    ["step_count", "num_steps", "num_cycles", "transition_speed", "steinvi_state"],
)


class ASVGD(SVGD):
    """Annealing Stein variational gradient descent [1].

    **Example:**

    .. doctest::

        >>> from jax import random, numpy as jnp

        >>> from numpyro import sample, param, plate
        >>> from numpyro.distributions import Beta, Bernoulli
        >>> from numpyro.distributions.constraints import positive

        >>> from numpyro.optim import Adagrad
        >>> from numpyro.contrib.einstein import ASVGD, RBFKernel
        >>> from numpyro.infer import Predictive

        >>> def model(data):
        ...     f = sample("fairness", Beta(10, 10))
        ...     n = data.shape[0] if data is not None else 1
        ...     with plate("N", n):
        ...         sample("obs", Bernoulli(f), obs=data)

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])

        >>> opt = Adagrad(step_size=0.05)
        >>> k = RBFKernel()
        >>> asvgd = ASVGD(model, opt, k, num_stein_particles=2)

        >>> asvgd_result = asvgd.run(random.PRNGKey(0), 200, data)

        >>> params = asvgd_result.params
        >>> predictive = Predictive(model, guide=asvgd.guide, params=params, num_samples=10, batch_ndims=1)
        >>> samples = predictive(random.PRNGKey(1), data=None)

    :param Callable model: Python callable with NumPyro primitives for the model.
    :param Callable guide: Python callable with NumPyro primitives for the guide.
    :param _NumPyroOptim optim: An instance of :class:`~numpyro.optim._NumpyroOptim`.
        Adagrad should be preferred over Adam [1].
    :param SteinKernel kernel_fn: Function that computes the reproducing kernel to use with ASVGD.
        We currently recommend :class:`~numpyro.contrib.einstein.RBFKernel`.
        This may change as criteria for kernel selection are not well understood yet.
    :param num_stein_particles: Number of particles (i.e., mixture components) in the mixture approximation.
        Default is `10`.
    :param num_cycles: The total number of cycles during inference. This corresponds to :math:`C` in eq. 4 of [1].
        Default is `10`.
    :param trans_speed: Speed of transition between two phases during inference. This corresponds to :math:`p` in eq. 4
        of [1]. Default is `10`.
    :param Dict guide_kwargs: Keyword arguments for :class:`~numpyro.infer.autoguide.AutoDelta`.
        Default behaviour is the same as the default for :class:`~numpyro.infer.autoguide.AutoDelta`.

        Usage::

            opt = Adagrad(step_size=0.05)
            k = RBFKernel()
            asvgd = ASVGD(model, opt, k, guide_kwargs={'init_loc_fn': partial(init_to_uniform, radius=0.1)})

    :param Dict static_kwargs: Static keyword arguments for the model and guide. These arguments cannot
        change during inference.

    **References:**

    1. D'Angelo, Francesco, and Vincent Fortuin. "Annealed Stein Variational Gradient Descent."
        Third Symposium on Advances in Approximate Bayesian Inference, 2021.
    """

    def __init__(
        self,
        model,
        optim,
        kernel_fn,
        num_stein_particles=10,
        num_cycles=10,
        transition_speed=10,
        guide_kwargs={},
        **static_kwargs,
    ):
        assert num_cycles > 0, f"The number of cycles must be >0. Got {num_cycles}."
        assert transition_speed > 0, (
            f"The transition speed must be >0. Got {transition_speed}."
        )

        self.num_cycles = num_cycles
        self.transition_speed = transition_speed

        super().__init__(
            model,
            optim,
            kernel_fn,
            num_stein_particles,
            guide_kwargs,
            **static_kwargs,
        )

    @staticmethod
    def _cyclical_annealing(
        step_count: int, num_steps: int, num_cycles: int, trans_speed: int
    ):
        """Cyclical annealing schedule as in eq. 4 of [1].

        **References** (MLA)

        1. D'Angelo, Francesco, and Vincent Fortuin. "Annealed Stein Variational Gradient Descent."
            Third Symposium on Advances in Approximate Bayesian Inference, 2021.

        :param step_count: The current number of steps taken.
        :param num_steps: The total number of steps. Corresponds to $T$ in eq. 4 of [1].
        :param num_cycles: The total number of cycles. Corresponds to $C$ in eq. 4 of [1].
        :param trans_speed: Speed of transition between two phases. Corresponds to $p$ in eq. 4 of [1].
        """
        norm = (num_steps + 1) / num_cycles
        cycle_len = num_steps // num_cycles

        # Safegaurd against num_cycles=1, which would cause division by zero
        last_start = jnp.maximum((num_cycles - 1) * cycle_len, 1.0)
        last_cycle = step_count // last_start

        return (1 - last_cycle) * (
            ((step_count % cycle_len) + 1) / norm
        ) ** trans_speed + last_cycle

    def init(self, rng_key, num_steps, *args, **kwargs):
        """Register random variable transformations, constraints and determine initialize positions of the particles.

        :param jax.random.PRNGKey rng_key: Random number generator seed.
        :param args: Positional arguments to the model and guide.
        :param num_steps: Total number of steps in the optimization.
        :param kwargs: Keyword arguments to the model and guide.
        :return: Initial :data:`ASVGDState`.
        """
        # Sets initial loss temperature to 1, the temperature is adjusted by calls to `self.update``.
        steinvi_state = super().init(rng_key, *args, **kwargs)
        return ASVGDState(
            step_count=0.0,
            num_steps=float(num_steps),
            num_cycles=float(self.num_cycles),
            transition_speed=float(self.transition_speed),
            steinvi_state=steinvi_state,
        )

    def get_params(self, state: ASVGDState):
        return super().get_params(state.steinvi_state)

    def update(self, state: ASVGDState, *args, **kwargs) -> ASVGDState:
        step_count, num_steps, num_cycles, transition_speed, steinvi_state = state

        # Compute the loss temperature
        loss_temperature = ASVGD._cyclical_annealing(
            step_count, num_steps, num_cycles, transition_speed
        )

        steinvi_state = SteinVIState(
            rng_key=steinvi_state.rng_key,
            optim_state=steinvi_state.optim_state,
            loss_temperature=loss_temperature,
            repulsion_temperature=steinvi_state.repulsion_temperature,
        )
        new_steinvi_state, loss_val = super().update(steinvi_state, *args, **kwargs)

        new_asvgd_state = ASVGDState(
            step_count + 1,
            state.num_steps,
            state.num_cycles,
            state.transition_speed,
            steinvi_state=new_steinvi_state,
        )
        return new_asvgd_state, loss_val

    def evaluate(self, state: ASVGDState, *args, **kwargs):
        """Take a single step of Stein (possibly on a batch / minibatch of data).

        :param ASVGDState state: Current state of inference.
        :param args: Positional arguments to the model and guide.
        :param kwargs: Keyword arguments to the model and guide.
        :return: Normed Stein force.
        """

        return super().evaluate(state.steinvi_state, *args, **kwargs)

    def setup_run(self, rng_key, num_steps, args, init_state, kwargs):
        if init_state is None:
            state = self.init(rng_key, num_steps, *args, **kwargs)
        else:
            assert isinstance(init_state, ASVGDState), (
                "The init_state much be an instance of ASVGDState"
            )
            state = init_state

        loss = self.evaluate(state, *args, **kwargs)

        info_init = (state, loss)

        def step(info):
            state, loss = info
            return self.update(state, *args, **kwargs)  # uses closure!

        def collect(info):
            _, loss = info
            return loss

        def extract(info):
            state, _ = info
            return state

        def diagnostic(info):
            _, loss = info
            return f"Stein force {loss:.2f}."

        return step, diagnostic, collect, extract, info_init

    def run(
        self,
        rng_key,
        num_steps,
        *args,
        progress_bar=True,
        init_state=None,
        **kwargs,
    ):
        """Run ASVGD inference.

        :param jax.random.PRNGKey rng_key: Random number generator seed.
        :param int num_steps: Number of steps to optimize.
        :param *args: Positional arguments to the model and guide.
        :param bool progress_bar: Use a progress bar. Default is `True`.
            Inference is faster with `False`.
        :param SteinVIState init_state: Initial state of inference.
            Default is ``None``, which will initialize using init before running inference.
        :param **kwargs: Keyword arguments to the model and guide.
        """
        step, diagnostic, collect, extract, init_info = self.setup_run(
            rng_key, num_steps, args, init_state, kwargs
        )

        auxiliaries, last_res = fori_collect(
            0,
            num_steps,
            step,
            init_info,
            progbar=progress_bar,
            transform=collect,
            return_last_val=True,
            diagnostics_fn=diagnostic if progress_bar else None,
        )

        state = extract(last_res)
        return SteinVIRunResult(self.get_params(state), state, auxiliaries)
