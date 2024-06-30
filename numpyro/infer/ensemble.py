# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import namedtuple
import warnings

import jax
from jax import random, vmap
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde

import numpyro.distributions as dist
from numpyro.infer.ensemble_util import batch_ravel_pytree, get_nondiagonal_indices
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity, is_prng_key

EnsembleSamplerState = namedtuple(
    "EnsembleSamplerState", ["z", "inner_state", "rng_key"]
)
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **inner_state** - A namedtuple containing information needed to update half the ensemble.
 - **rng_key** - random number generator seed used for generating proposals, etc.
"""

AIESState = namedtuple("AIESState", ["i", "accept_prob", "mean_accept_prob", "rng_key"])
"""
A :func:`~collections.namedtuple` consisting of the following fields.

 - **i** - iteration.
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup adaptation or sampling (for diagnostics).
 - **rng_key** - random number generator seed used for generating proposals, etc.
"""

ESSState = namedtuple(
    "ESSState", ["i", "n_expansions", "n_contractions", "mu", "rng_key"]
)
"""
A :func:`~collections.namedtuple` used as an inner state for Ensemble Sampler.
This consists of the following fields:

 - **i** - iteration.
 - **n_expansions** - number of expansions in the current batch. Used for tuning mu.
 - **n_contractions** - number of contractions in the current batch. Used for tuning mu.
 - **mu** - Scale factor. This is tuned if tune_mu=True.
 - **rng_key** - random number generator seed used for generating proposals, etc.
"""


class EnsembleSampler(MCMCKernel, ABC):
    """
    Abstract class for ensemble samplers. Each MCMC sample is divided into two sub-iterations
    in which half of the ensemble is updated.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param bool randomize_split: whether or not to permute the chain order at each iteration.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def __init__(
        self, model=None, potential_fn=None, *, randomize_split, init_strategy
    ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")

        self._model = model
        self._potential_fn = potential_fn
        self._batch_log_density = None
        # unravel an (n_chains, n_params) Array into a pytree and
        # evaluate the log density at each chain

        # --- other hyperparams go here
        self._num_chains = None  # must be an even number >= 2
        self._randomize_split = randomize_split
        # ---

        self._init_strategy = init_strategy
        self._postprocess_fn = None

    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return "z"

    @property
    def is_ensemble_kernel(self):
        return True

    @abstractmethod
    def init_inner_state(self, rng_key):
        """return inner_state"""
        raise NotImplementedError

    @abstractmethod
    def update_active_chains(self, active, inactive, inner_state):
        """return (updated active set of chains, updated inner state)"""
        raise NotImplementedError

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            (
                new_params_info,
                potential_fn_gen,
                self._postprocess_fn,
                _,
            ) = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                validate_grad=False,
            )
            new_init_params = new_params_info[0]
            self._potential_fn = potential_fn_gen(*model_args, **model_kwargs)

            if init_params is None:
                init_params = new_init_params

        flat_params, unravel_fn = batch_ravel_pytree(init_params)
        self._batch_log_density = lambda z: -vmap(self._potential_fn)(unravel_fn(z))

        if self._num_chains < 2 * flat_params.shape[1]:
            warnings.warn(
                "Setting n_chains to at least 2*n_params is strongly recommended.\n"
                f"n_chains: {self._num_chains}, n_params: {flat_params.shape[1]}"
            )

        return init_params

    def init(
        self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}
    ):
        assert not is_prng_key(rng_key), (
            "EnsembleSampler only supports chain_method='vectorized' with num_chains > 1.\n"
            "If you want to run chains in parallel, please raise a github issue."
        )
        assert rng_key.shape[0] % 2 == 0, "Number of chains must be even."
        self._num_chains = rng_key.shape[0]
        if init_params is not None:
            assert all(
                [
                    param.shape[0] == self._num_chains
                    for param in jax.tree.leaves(init_params)
                ]
            ), "The batch dimension of each param must match n_chains"

        rng_key, rng_key_inner_state, rng_key_init_model = random.split(rng_key[0], 3)
        rng_key_init_model = random.split(rng_key_init_model, self._num_chains)

        init_params = self._init_state(
            rng_key_init_model, model_args, model_kwargs, init_params
        )

        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with `potential_fn`."
            )

        self._num_warmup = num_warmup

        return EnsembleSamplerState(
            init_params, self.init_inner_state(rng_key_inner_state), rng_key
        )

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        z, inner_state, rng_key = state
        rng_key, _ = random.split(rng_key)
        z_flat, unravel_fn = batch_ravel_pytree(z)

        if self._randomize_split:
            z_flat = random.permutation(rng_key, z_flat, axis=0)

        split_ind = self._num_chains // 2

        def body_fn(i, z_flat_inner_state):
            z_flat, inner_state = z_flat_inner_state

            active, inactive = jax.lax.cond(
                i == 0,
                lambda x: (x[:split_ind], x[split_ind:]),
                lambda x: (x[split_ind:], x[split_ind:]),
                z_flat,
            )

            z_updates, inner_state = self.update_active_chains(
                active, inactive, inner_state
            )

            z_flat = jax.lax.cond(
                i == 0,
                lambda x: x.at[:split_ind].set(z_updates),
                lambda x: x.at[split_ind:].set(z_updates),
                z_flat,
            )
            return (z_flat, inner_state)

        z_flat, inner_state = jax.lax.fori_loop(0, 2, body_fn, (z_flat, inner_state))

        return EnsembleSamplerState(unravel_fn(z_flat), inner_state, rng_key)


class AIES(EnsembleSampler):
    """
    Affine-Invariant Ensemble Sampling: a gradient free method that informs Metropolis-Hastings
    proposals by sharing information between chains. Suitable for low to moderate dimensional models.
    Generally, `num_chains` should be at least twice the dimensionality of the model.

    .. note:: This kernel must be used with `num_chains` > 1 and `chain_method="vectorized`
        in :class:`MCMC`. The number of chains must be divisible by 2.

    **References:**

    1. *emcee: The MCMC Hammer* (https://iopscience.iop.org/article/10.1086/670067),
        Daniel Foreman-Mackey, David W. Hogg, Dustin Lang, and Jonathan Goodman.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param bool randomize_split: whether or not to permute the chain order at each iteration.
        Defaults to False.
    :param moves: a dictionary mapping moves to their respective probabilities of being selected.
        Valid keys are `AIES.DEMove()` and `AIES.StretchMove()`. Both tend to work well in practice.
        If the sum of probabilites exceeds 1, the probabilities will be normalized. Defaults to `{AIES.DEMove(): 1.0}`.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.

    **Example**

    .. doctest::

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, AIES

        >>> def model():
        ...    x = numpyro.sample("x", dist.Normal().expand([10]))
        ...    numpyro.sample("obs", dist.Normal(x, 1.0), obs=jnp.ones(10))
        >>>
        >>> kernel = AIES(model, moves={AIES.DEMove() : 0.5,
        ...                            AIES.StretchMove() : 0.5})
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=20, chain_method='vectorized')
        >>> mcmc.run(jax.random.PRNGKey(0))
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        randomize_split=False,
        moves=None,
        init_strategy=init_to_uniform,
    ):
        if not moves:
            self._moves = [AIES.DEMove()]
            self._weights = jnp.array([1.0])
        else:
            self._moves = list(moves.keys())
            self._weights = jnp.array([weight for weight in moves.values()]) / len(
                moves
            )

            assert all(
                [hasattr(move, "__call__") for move in self._moves]
            ), "Each move must be a callable (one of AIES.DEMove(), or AIES.StretchMove())."
            assert jnp.all(
                self._weights >= 0
            ), "Each specified move must have probability >= 0"

        super().__init__(
            model,
            potential_fn,
            randomize_split=randomize_split,
            init_strategy=init_strategy,
        )

    def get_diagnostics_str(self, state):
        return "acc. prob={:.2f}".format(state.inner_state.mean_accept_prob)

    def init_inner_state(self, rng_key):
        # XXX hack -- we don't know num_chains until we init the inner state
        self._moves = [
            move(self._num_chains) if move.__name__ == "make_de_move" else move
            for move in self._moves
        ]

        return AIESState(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), rng_key)

    def update_active_chains(self, active, inactive, inner_state):
        i, _, mean_accept_prob, rng_key = inner_state
        rng_key, move_key, proposal_key, accept_key = random.split(rng_key, 4)

        move_i = random.choice(move_key, len(self._moves), p=self._weights)
        proposal, factors = jax.lax.switch(
            move_i, self._moves, proposal_key, active, inactive
        )

        # --- evaluate the proposal ---
        log_accept_prob = (
            factors
            + self._batch_log_density(proposal)
            - self._batch_log_density(active)
        )

        accepted = dist.Uniform().sample(accept_key, (active.shape[0],)) < jnp.exp(
            log_accept_prob
        )
        updated_active_chains = jnp.where(accepted[:, jnp.newaxis], proposal, active)

        accept_prob = jnp.count_nonzero(accepted) / accepted.shape[0]
        itr = i + 0.5
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob = mean_accept_prob + (accept_prob - mean_accept_prob) / n

        return updated_active_chains, AIESState(
            itr, accept_prob, mean_accept_prob, rng_key
        )

    @staticmethod
    def DEMove(sigma=1.0e-5, g0=None):
        """
        A proposal using differential evolution.

        This `Differential evolution proposal
        <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
        implemented following `Nelson et al. (2013)
        <https://doi.org/10.1088/0067-0049/210/1/11>`_.

        :param sigma: (optional)
            The standard deviation of the Gaussian used to stretch the proposal vector.
            Defaults to `1.0.e-5`.
        :param g0 (optional):
            The mean stretch factor for the proposal vector. By default,
            it is `2.38 / sqrt(2*ndim)` as recommended by the two references.
        """

        def make_de_move(n_chains):
            PAIRS = get_nondiagonal_indices(n_chains // 2)

            def de_move(rng_key, active, inactive):
                pairs_key, gamma_key = random.split(rng_key)
                n_active_chains, n_params = inactive.shape

                # XXX: if we pass in n_params to parent scope we don't need to
                # recompute this each time
                g = 2.38 / jnp.sqrt(2.0 * n_params) if not g0 else g0

                selected_pairs = random.choice(
                    pairs_key, PAIRS, shape=(n_active_chains,)
                )

                # Compute diff vectors
                diffs = jnp.diff(inactive[selected_pairs], axis=1).squeeze(axis=1)

                # Sample a gamma value for each walker following Nelson et al. (2013)
                gamma = dist.Normal(g, g * sigma).sample(
                    gamma_key, sample_shape=(n_active_chains, 1)
                )

                # In this way, sigma is the standard deviation of the distribution of gamma,
                # instead of the standard deviation of the distribution of the proposal as proposed by Ter Braak (2006).
                # Otherwise, sigma should be tuned for each dimension, which confronts the idea of affine-invariance.
                proposal = active + gamma * diffs

                return proposal, jnp.zeros(n_active_chains)

            return de_move

        return make_de_move

    @staticmethod
    def StretchMove(a=2.0):
        """
        A `Goodman & Weare (2010)
        <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
        parallelization as described in `Foreman-Mackey et al. (2013)
        <https://arxiv.org/abs/1202.3665>`_.

        :param a: (optional)
            The stretch scale parameter. (default: ``2.0``)
        """

        def stretch_move(rng_key, active, inactive):
            n_active_chains, n_params = active.shape
            unif_key, idx_key = random.split(rng_key)

            zz = (
                (a - 1.0) * random.uniform(unif_key, shape=(n_active_chains,)) + 1
            ) ** 2.0 / a
            factors = (n_params - 1.0) * jnp.log(zz)
            r_idxs = random.randint(
                idx_key, shape=(n_active_chains,), minval=0, maxval=n_active_chains
            )

            proposal = (
                inactive[r_idxs] - (inactive[r_idxs] - active) * zz[:, jnp.newaxis]
            )

            return proposal, factors

        return stretch_move


class ESS(EnsembleSampler):
    """
    Ensemble Slice Sampling: a gradient free method that finds better slice sampling directions
    by sharing information between chains. Suitable for low to moderate dimensional models.
    Generally, `num_chains` should be at least twice the dimensionality of the model.

    .. note:: This kernel must be used with `num_chains` > 1 and `chain_method="vectorized`
        in :class:`MCMC`. The number of chains must be divisible by 2.

    **References:**

    1. *zeus: a PYTHON implementation of ensemble slice sampling for efficient Bayesian parameter inference* (https://academic.oup.com/mnras/article/508/3/3589/6381726),
        Minas Karamanis, Florian Beutler, and John A. Peacock.
    2. *Ensemble slice sampling* (https://link.springer.com/article/10.1007/s11222-021-10038-2),
        Minas Karamanis, Florian Beutler.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param bool randomize_split: whether or not to permute the chain order at each iteration.
        Defaults to True.
    :param moves: a dictionary mapping moves to their respective probabilities of being selected.
        If the sum of probabilites exceeds 1, the probabilities will be normalized. Valid keys include:
        `ESS.DifferentialMove()` -> default proposal, works well along a wide range of target distributions,
        `ESS.GaussianMove()` -> for approximately normally distributed targets,
        `ESS.KDEMove()` -> for multimodal posteriors - requires large `num_chains`, and they must be well initialized
        `ESS.RandomMove()` -> no chain interaction, useful for debugging.
        Defaults to `{ESS.DifferentialMove(): 1.0}`.

    :param int max_steps: number of maximum stepping-out steps per sample. Defaults to 10,000.
    :param int max_iter: number of maximum expansions/contractions per sample. Defaults to 10,000.
    :param float init_mu: initial scale factor. Defaults to 1.0.
    :param bool tune_mu: whether or not to tune the initial scale factor. Defaults to True.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.

    **Example**

    .. doctest::

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, ESS

        >>> def model():
        ...    x = numpyro.sample("x", dist.Normal().expand([10]))
        ...    numpyro.sample("obs", dist.Normal(x, 1.0), obs=jnp.ones(10))
        >>>
        >>> kernel = ESS(model, moves={ESS.DifferentialMove() : 0.8,
        ...                            ESS.RandomMove() : 0.2})
        >>> mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=20, chain_method='vectorized')
        >>> mcmc.run(jax.random.PRNGKey(0))
    """

    def __init__(
        self,
        model=None,
        potential_fn=None,
        randomize_split=True,
        moves=None,
        max_steps=10_000,
        max_iter=10_000,
        init_mu=1.0,
        tune_mu=True,
        init_strategy=init_to_uniform,
    ):
        if not moves:
            self._moves = [ESS.DifferentialMove()]
            self._weights = jnp.array([1.0])
        else:
            self._moves = list(moves.keys())
            self._weights = jnp.array([weight for weight in moves.values()]) / len(
                moves
            )

            assert all([hasattr(move, "__call__") for move in self._moves]), (
                "Each move must be a callable (one of `ESS.DifferentialMove()`, "
                "`ESS.GaussianMove()`, `ESS.KDEMove()`, `ESS.RandomMove()`)"
            )

            assert jnp.all(
                self._weights >= 0
            ), "Each specified move must have probability >= 0"
            assert init_mu > 0, "Scale factor should be strictly positive"

        self._max_steps = max_steps  # max number of stepping out steps
        self._max_iter = max_iter  # max number of expansions/contractions
        self._init_mu = init_mu
        self._tune_mu = tune_mu

        super().__init__(
            model,
            potential_fn,
            randomize_split=randomize_split,
            init_strategy=init_strategy,
        )

    def init_inner_state(self, rng_key):
        self.batch_log_density = lambda x: self._batch_log_density(x)[:, jnp.newaxis]

        # XXX hack -- we don't know num_chains until we init the inner state
        self._moves = [
            move(self._num_chains)
            if move.__name__ == "make_differential_move"
            else move
            for move in self._moves
        ]

        return ESSState(
            jnp.array(0.0), jnp.array(0), jnp.array(0), self._init_mu, rng_key
        )

    def update_active_chains(self, active, inactive, inner_state):
        i, n_expansions, n_contractions, mu, rng_key = inner_state
        (
            rng_key,
            move_key,
            dir_key,
            height_key,
            step_out_key,
            shrink_key,
        ) = random.split(rng_key, 6)

        n_active_chains, n_params = active.shape

        move_i = random.choice(move_key, len(self._moves), p=self._weights)
        directions = jax.lax.switch(move_i, self._moves, dir_key, inactive, mu)

        log_slice_height = self.batch_log_density(active) - dist.Exponential().sample(
            height_key, sample_shape=(n_active_chains, 1)
        )

        curr_n_expansions, L, R = self._step_out(
            step_out_key, log_slice_height, active, directions
        )
        proposal, curr_n_contractions = self._shrink(
            shrink_key, log_slice_height, L, R, active, directions
        )

        n_expansions += curr_n_expansions
        n_contractions += curr_n_contractions
        itr = i + 0.5

        if self._tune_mu:
            safe_n_expansions = jnp.max(jnp.array([1, n_expansions]))

            # only update tuning scale if a full iteration has passed
            mu, n_expansions, n_contractions = jax.lax.cond(
                jnp.all(itr % 1 == 0),
                lambda n_exp, n_con: (
                    2.0 * n_exp / (n_exp + n_con),
                    jnp.array(0),
                    jnp.array(0),
                ),
                lambda _, __: (mu, n_expansions, n_contractions),
                safe_n_expansions,
                n_contractions,
            )

        return proposal, ESSState(itr, n_expansions, n_contractions, mu, rng_key)

    @staticmethod
    def RandomMove():
        """
        The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Random Move" with parallelization.
        When this move is used the walkers move along random directions. There is no communication between the
        walkers and this Move corresponds to the vanilla Slice Sampling method. This Move should be used for
        debugging purposes only.
        """

        def random_move(rng_key, inactive, mu):
            directions = dist.Normal(loc=0, scale=1).sample(
                rng_key, sample_shape=inactive.shape
            )
            directions /= jnp.linalg.norm(directions, axis=0)

            return 2.0 * mu * directions

        return random_move

    @staticmethod
    def KDEMove(bw_method=None):
        """
        The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "KDE Move" with parallelization.
        When this Move is used the distribution of the walkers of the complementary ensemble is traced using
        a Gaussian Kernel Density Estimation methods. The walkers then move along random direction vectos
        sampled from this distribution.
        """

        def kde_move(rng_key, inactive, mu):
            n_active_chains, n_params = inactive.shape

            kde = gaussian_kde(inactive.T, bw_method=bw_method)

            vectors = kde.resample(rng_key, (2 * n_active_chains,)).T
            directions = vectors[:n_active_chains] - vectors[n_active_chains:]

            return 2.0 * mu * directions

        return kde_move

    @staticmethod
    def GaussianMove():
        """
        The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Gaussian Move" with parallelization.
        When this Move is used the walkers move along directions defined by random vectors sampled from the Gaussian
        approximation of the walkers of the complementary ensemble.
        """

        # In high dimensional regimes with sufficiently small n_active_chains,
        # it is more efficient to sample without computing the Cholesky
        # decomposition of the covariance matrix:

        # eps = dist.Normal(0, 1).sample(rng_key, (n_active_chains, n_active_chains))
        # return 2.0 * mu * (eps @ (inactive - jnp.mean(inactive, axis=0)) / jnp.sqrt(n_active_chains))

        def gaussian_move(rng_key, inactive, mu):
            n_active_chains, n_params = inactive.shape
            cov = jnp.cov(inactive, rowvar=False)

            return (
                2.0
                * mu
                * dist.MultivariateNormal(0, cov).sample(
                    rng_key, sample_shape=(n_active_chains,)
                )
            )

        return gaussian_move

    @staticmethod
    def DifferentialMove():
        """
        The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Differential Move" with parallelization.
        When this Move is used the walkers move along directions defined by random pairs of walkers sampled (with no
        replacement) from the complementary ensemble. This is the default choice and performs well along a wide range
        of target distributions.
        """

        def make_differential_move(n_chains):
            PAIRS = get_nondiagonal_indices(n_chains // 2)

            def differential_move(rng_key, inactive, mu):
                n_active_chains, n_params = inactive.shape

                selected_pairs = random.choice(rng_key, PAIRS, shape=(n_active_chains,))
                diffs = jnp.diff(inactive[selected_pairs], axis=1).squeeze(
                    axis=1
                )  # get the pairwise difference of each vector

                return 2.0 * mu * diffs

            return differential_move

        return make_differential_move

    def _step_out(self, rng_key, log_slice_height, active, directions):
        init_L_key, init_J_key = random.split(rng_key)
        n_active_chains, n_params = active.shape

        iteration = 0
        n_expansions = 0
        # set initial interval boundaries
        L = -dist.Uniform().sample(init_L_key, sample_shape=(n_active_chains, 1))
        R = L + 1.0

        # stepping out
        J = jnp.floor(
            dist.Uniform(low=0, high=self._max_steps).sample(
                init_J_key, sample_shape=(n_active_chains, 1)
            )
        )
        K = (self._max_steps - 1) - J

        # left stepping-out initialisation
        mask_J = jnp.full((n_active_chains, 1), True)
        # right stepping-out initialisation
        mask_K = jnp.full((n_active_chains, 1), True)

        init_values = (n_expansions, L, R, J, K, mask_J, mask_K, iteration)

        def cond_fn(args):
            n_expansions, L, R, J, K, mask_J, mask_K, iteration = args

            return (jnp.count_nonzero(mask_J) + jnp.count_nonzero(mask_K) > 0) & (
                iteration < self._max_iter
            )

        def body_fn(args):
            n_expansions, L, R, J, K, mask_J, mask_K, iteration = args

            log_prob_L = self.batch_log_density(directions * L + active)
            log_prob_R = self.batch_log_density(directions * R + active)

            can_expand_L = log_prob_L > log_slice_height
            L = jnp.where(can_expand_L, L - 1, L)
            J = jnp.where(can_expand_L, J - 1, J)
            mask_J = jnp.where(can_expand_L, mask_J, False)

            can_expand_R = log_prob_R > log_slice_height
            R = jnp.where(can_expand_R, R + 1, R)
            K = jnp.where(can_expand_R, K - 1, K)
            mask_K = jnp.where(can_expand_R, mask_K, False)

            iteration += 1
            n_expansions += jnp.count_nonzero(can_expand_L) + jnp.count_nonzero(
                can_expand_R
            )

            return (n_expansions, L, R, J, K, mask_J, mask_K, iteration)

        n_expansions, L, R, J, K, mask_J, mask_K, iteration = jax.lax.while_loop(
            cond_fn, body_fn, init_values
        )

        return n_expansions, L, R

    def _shrink(self, rng_key, log_slice_height, L, R, active, directions):
        n_active_chains, n_params = active.shape

        iteration = 0
        n_contractions = 0
        widths = jnp.zeros((n_active_chains, 1))
        proposed = jnp.zeros((n_active_chains, n_params))
        can_shrink = jnp.full((n_active_chains, 1), True)

        init_values = (
            rng_key,
            proposed,
            n_contractions,
            L,
            R,
            widths,
            can_shrink,
            iteration,
        )

        def cond_fn(args):
            (
                rng_key,
                proposed,
                n_contractions,
                L,
                R,
                widths,
                can_shrink,
                iteration,
            ) = args

            return (jnp.count_nonzero(can_shrink) > 0) & (iteration < self._max_iter)

        def body_fn(args):
            (
                rng_key,
                proposed,
                n_contractions,
                L,
                R,
                widths,
                can_shrink,
                iteration,
            ) = args

            rng_key, _ = random.split(rng_key)

            widths = jnp.where(
                can_shrink, dist.Uniform(low=L, high=R).sample(rng_key), widths
            )

            # compute new positions
            proposed = jnp.where(can_shrink, directions * widths + active, proposed)
            proposed_log_prob = self.batch_log_density(proposed)

            # shrink slices
            can_shrink = proposed_log_prob < log_slice_height

            L_cond = can_shrink & (widths < 0.0)
            L = jnp.where(L_cond, widths, L)

            R_cond = can_shrink & (widths > 0.0)
            R = jnp.where(R_cond, widths, R)

            iteration += 1
            n_contractions += jnp.count_nonzero(L_cond) + jnp.count_nonzero(R_cond)

            return (
                rng_key,
                proposed,
                n_contractions,
                L,
                R,
                widths,
                can_shrink,
                iteration,
            )

        (
            rng_key,
            proposed,
            n_contractions,
            L,
            R,
            widths,
            can_shrink,
            iteration,
        ) = jax.lax.while_loop(cond_fn, body_fn, init_values)

        return proposed, n_contractions
