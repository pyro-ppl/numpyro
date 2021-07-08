# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import namedtuple, partial
import warnings

import tqdm

import jax
from jax import jit, lax, random
import jax.numpy as jnp
from jax.tree_util import tree_map

from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import replay, seed, trace
from numpyro.infer.util import helpful_support_errors, transform_fn
from numpyro.optim import _NumPyroOptim

SVIState = namedtuple("SVIState", ["optim_state", "mutable_state", "rng_key"])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **optim_state** - current optimizer's state.
 - **mutable_state** - extra state to store values of `"mutable"` sites
 - **rng_key** - random number generator seed used for the iteration.
"""


SVIRunResult = namedtuple("SVIRunResult", ["params", "state", "losses"])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **params** - the optimized parameters.
 - **state** - the last :class:`SVIState`
 - **losses** - the losses collected at every step.
"""


def _make_loss_fn(
    elbo,
    rng_key,
    constrain_fn,
    model,
    guide,
    args,
    kwargs,
    static_kwargs,
    mutable_state=None,
):
    def loss_fn(params):
        params = constrain_fn(params)
        if mutable_state is not None:
            params.update(mutable_state)
            result = elbo.loss_with_mutable_state(
                rng_key, params, model, guide, *args, **kwargs, **static_kwargs
            )
            return result["loss"], result["mutable_state"]
        else:
            return (
                elbo.loss(
                    rng_key, params, model, guide, *args, **kwargs, **static_kwargs
                ),
                None,
            )

    return loss_fn


class SVI(object):
    """
    Stochastic Variational Inference given an ELBO loss objective.

    **References**

    1. *SVI Part I: An Introduction to Stochastic Variational Inference in Pyro*,
       (http://pyro.ai/examples/svi_part_i.html)

    **Example:**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.distributions import constraints
        >>> from numpyro.infer import SVI, Trace_ELBO

        >>> def model(data):
        ...     f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
        ...     with numpyro.plate("N", data.shape[0]):
        ...         numpyro.sample("obs", dist.Bernoulli(f), obs=data)

        >>> def guide(data):
        ...     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
        ...     beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
        ...                            constraint=constraints.positive)
        ...     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
        >>> optimizer = numpyro.optim.Adam(step_size=0.0005)
        >>> svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        >>> svi_result = svi.run(random.PRNGKey(0), 2000, data)
        >>> params = svi_result.params
        >>> inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: An instance of :class:`~numpyro.optim._NumpyroOptim`, a
        ``jax.experimental.optimizers.Optimizer`` or an Optax
        ``GradientTransformation``. If you pass an Optax optimizer it will
        automatically be wrapped using :func:`numpyro.contrib.optim.optax_to_numpyro`.

            >>> from optax import adam, chain, clip
            >>> svi = SVI(model, guide, chain(clip(10.0), adam(1e-3)), loss=Trace_ELBO())

    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

    def __init__(self, model, guide, optim, loss, **static_kwargs):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.static_kwargs = static_kwargs
        self.constrain_fn = None

        if isinstance(optim, _NumPyroOptim):
            self.optim = optim
        elif isinstance(optim, jax.experimental.optimizers.Optimizer):
            self.optim = _NumPyroOptim(lambda *args: args, *optim)
        else:
            try:
                import optax

                from numpyro.contrib.optim import optax_to_numpyro
            except ImportError:
                raise ImportError(
                    "It looks like you tried to use an optimizer that isn't an "
                    "instance of numpyro.optim._NumPyroOptim or "
                    "jax.experimental.optimizers.Optimizer. There is experimental "
                    "support for Optax optimizers, but you need to install Optax. "
                    "It can be installed with `pip install optax`."
                )

            if not isinstance(optim, optax.GradientTransformation):
                raise TypeError(
                    "Expected either an instance of numpyro.optim._NumPyroOptim, "
                    "jax.experimental.optimizers.Optimizer or "
                    "optax.GradientTransformation. Got {}".format(type(optim))
                )

            self.optim = optax_to_numpyro(optim)

    def init(self, rng_key, *args, **kwargs):
        """
        Gets the initial SVI state.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: the initial :data:`SVIState`
        """
        rng_key, model_seed, guide_seed = random.split(rng_key, 3)
        model_init = seed(self.model, model_seed)
        guide_init = seed(self.guide, guide_seed)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = trace(replay(model_init, guide_trace)).get_trace(
            *args, **kwargs, **self.static_kwargs
        )
        params = {}
        inv_transforms = {}
        mutable_state = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site["type"] == "param":
                constraint = site["kwargs"].pop("constraint", constraints.real)
                with helpful_support_errors(site):
                    transform = biject_to(constraint)
                inv_transforms[site["name"]] = transform
                params[site["name"]] = transform.inv(site["value"])
            elif site["type"] == "mutable":
                mutable_state[site["name"]] = site["value"]
            elif (
                site["type"] == "sample"
                and (not site["is_observed"])
                and site["fn"].support.is_discrete
            ):
                warnings.warn(
                    "Currently, SVI does not support models with discrete latent variables"
                )

        if not mutable_state:
            mutable_state = None
        self.constrain_fn = partial(transform_fn, inv_transforms)
        # we convert weak types like float to float32/float64
        # to avoid recompiling body_fn in svi.run
        params, mutable_state = tree_map(
            lambda x: lax.convert_element_type(x, jnp.result_type(x)),
            (params, mutable_state),
        )
        return SVIState(self.optim.init(params), mutable_state, rng_key)

    def get_params(self, svi_state):
        """
        Gets values at `param` sites of the `model` and `guide`.

        :param svi_state: current state of SVI.
        :return: the corresponding parameters
        """
        params = self.constrain_fn(self.optim.get_params(svi_state.optim_state))
        return params

    def update(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(svi_state, loss)`.
        """
        rng_key, rng_key_step = random.split(svi_state.rng_key)
        loss_fn = _make_loss_fn(
            self.loss,
            rng_key_step,
            self.constrain_fn,
            self.model,
            self.guide,
            args,
            kwargs,
            self.static_kwargs,
            mutable_state=svi_state.mutable_state,
        )
        (loss_val, mutable_state), optim_state = self.optim.eval_and_update(
            loss_fn, svi_state.optim_state
        )
        return SVIState(optim_state, mutable_state, rng_key), loss_val

    def stable_update(self, svi_state, *args, **kwargs):
        """
        Similar to :meth:`update` but returns the current state if the
        the loss or the new state contains invalid values.

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(svi_state, loss)`.
        """
        rng_key, rng_key_step = random.split(svi_state.rng_key)
        loss_fn = _make_loss_fn(
            self.loss,
            rng_key_step,
            self.constrain_fn,
            self.model,
            self.guide,
            args,
            kwargs,
            self.static_kwargs,
            mutable_state=svi_state.mutable_state,
        )
        (loss_val, mutable_state), optim_state = self.optim.eval_and_stable_update(
            loss_fn, svi_state.optim_state
        )
        return SVIState(optim_state, mutable_state, rng_key), loss_val

    def run(
        self,
        rng_key,
        num_steps,
        *args,
        progress_bar=True,
        stable_update=False,
        **kwargs
    ):
        """
        (EXPERIMENTAL INTERFACE) Run SVI with `num_steps` iterations, then return
        the optimized parameters and the stacked losses at every step. If `num_steps`
        is large, setting `progress_bar=False` can make the run faster.

        .. note:: For a complex training process (e.g. the one requires early stopping,
            epoch training, varying args/kwargs,...), we recommend to use the more
            flexible methods :meth:`init`, :meth:`update`, :meth:`evaluate` to
            customize your training procedure.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param int num_steps: the number of optimization steps.
        :param args: arguments to the model / guide
        :param bool progress_bar: Whether to enable progress bar updates. Defaults to
            ``True``.
        :param bool stable_update: whether to use :meth:`stable_update` to update
            the state. Defaults to False.
        :param kwargs: keyword arguments to the model / guide
        :return: a namedtuple with fields `params` and `losses` where `params`
            holds the optimized values at :class:`numpyro.param` sites,
            and `losses` is the collected loss during the process.
        :rtype: SVIRunResult
        """

        def body_fn(svi_state, _):
            if stable_update:
                svi_state, loss = self.stable_update(svi_state, *args, **kwargs)
            else:
                svi_state, loss = self.update(svi_state, *args, **kwargs)
            return svi_state, loss

        svi_state = self.init(rng_key, *args, **kwargs)
        if progress_bar:
            losses = []
            with tqdm.trange(1, num_steps + 1) as t:
                batch = max(num_steps // 20, 1)
                for i in t:
                    svi_state, loss = jit(body_fn)(svi_state, None)
                    losses.append(loss)
                    if i % batch == 0:
                        if stable_update:
                            valid_losses = [x for x in losses[i - batch :] if x == x]
                            num_valid = len(valid_losses)
                            if num_valid == 0:
                                avg_loss = float("nan")
                            else:
                                avg_loss = sum(valid_losses) / num_valid
                        else:
                            avg_loss = sum(losses[i - batch :]) / batch
                        t.set_postfix_str(
                            "init loss: {:.4f}, avg. loss [{}-{}]: {:.4f}".format(
                                losses[0], i - batch + 1, i, avg_loss
                            ),
                            refresh=False,
                        )
            losses = jnp.stack(losses)
        else:
            svi_state, losses = lax.scan(body_fn, svi_state, None, length=num_steps)

        # XXX: we also return the last svi_state for further inspection of both
        # optimizer's state and mutable state.
        return SVIRunResult(self.get_params(svi_state), svi_state, losses)

    def evaluate(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_key_eval = random.split(svi_state.rng_key)
        params = self.get_params(svi_state)
        return self.loss.loss(
            rng_key_eval,
            params,
            self.model,
            self.guide,
            *args,
            **kwargs,
            **self.static_kwargs
        )
