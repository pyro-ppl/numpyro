# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch

from jax import random
import jax.numpy as jnp

try:
    from jaxns import (
        DefaultNestedSampler,
        Model,
        Prior,
        TerminationCondition,
        plot_cornerplot,
        plot_diagnostics,
        resample,
        summary,
    )
    from jaxns.utils import NestedSamplerResults

except ImportError as e:
    raise ImportError(
        "To use this module, please install `jaxns` package. It can be"
        " installed with `pip install jaxns` with python>=3.8"
    ) from e

import tensorflow_probability.substrates.jax as tfp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import Reparam
from numpyro.infer.util import _guess_max_plate_nesting, _validate_model, log_density

__all__ = ["NestedSampler"]

tfpd = tfp.distributions


@singledispatch
def uniform_reparam_transform(d):
    """
    A helper for :class:`UniformReparam` to get the transform that transforms
    a uniform distribution over a unit hypercube to the target distribution `d`.
    """
    if isinstance(d, dist.TransformedDistribution):
        outer_transform = dist.transforms.ComposeTransform(d.transforms)
        return lambda q: outer_transform(uniform_reparam_transform(d.base_dist)(q))

    if isinstance(
        d, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
    ):
        return lambda q: uniform_reparam_transform(d.base_dist)(q)

    return d.icdf


@uniform_reparam_transform.register(dist.MultivariateNormal)
def _(d):
    outer_transform = dist.transforms.LowerCholeskyAffine(d.loc, d.scale_tril)
    return lambda q: outer_transform(dist.Normal(0, 1).icdf(q))


@uniform_reparam_transform.register(dist.BernoulliLogits)
@uniform_reparam_transform.register(dist.BernoulliProbs)
def _(d):
    def transform(q):
        x = q < d.probs
        return x.astype(jnp.result_type(x, int))

    return transform


@uniform_reparam_transform.register(dist.CategoricalLogits)
@uniform_reparam_transform.register(dist.CategoricalProbs)
def _(d):
    return lambda q: jnp.sum(jnp.cumsum(d.probs, axis=-1) < q[..., None], axis=-1)


@uniform_reparam_transform.register(dist.Dirichlet)
def _(d):
    gamma_dist = dist.Gamma(d.concentration)

    def transform_fn(q):
        # NB: icdf is not available yet for Gamma distribution
        # so this will raise an NotImplementedError for now.
        # We will need scipy.special.gammaincinv, which is not available yet in JAX
        # see issue: https://github.com/google/jax/issues/5350
        # TODO: consider wrap jaxns GammaPrior transform implementation
        gammas = uniform_reparam_transform(gamma_dist)(q)
        return gammas / gammas.sum(-1, keepdims=True)

    return transform_fn


class UniformReparam(Reparam):
    """
    Reparameterize a distribution to a Uniform over the unit hypercube.

    Most univariate distribution uses Inverse CDF for the reparameterization.
    """

    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        shape = fn.shape()
        fn, expand_shape, event_dim = self._unwrap(fn)
        transform = uniform_reparam_transform(fn)
        tiny = jnp.finfo(jnp.result_type(float)).tiny

        x = numpyro.sample(
            "{}_base".format(name),
            dist.Uniform(tiny, 1).expand(shape).to_event(event_dim).mask(False),
        )
        # Simulate a numpyro.deterministic() site.
        return None, transform(x)


class NestedSampler:
    """
    (EXPERIMENTAL) A wrapper for `jaxns <https://github.com/Joshuaalbert/jaxns>`_ ,
    a nested sampling package based on JAX.

    See reference [1] for details on the meaning of each parameter.
    Please consider citing this reference if you use the nested sampler in your research.

    .. note:: To enumerate over a discrete latent variable, you can add the keyword
        `infer={"enumerate": "parallel"}` to the corresponding `sample` statement.

    .. note:: To improve the performance, please consider enabling x64 mode at the beginning
        of your NumPyro program ``numpyro.enable_x64()``.

    **References**

    1. *JAXNS: a high-performance nested sampling package based on JAX*,
       Joshua G. Albert (https://arxiv.org/abs/2012.15286)

    :param callable model: a call with NumPyro primitives
    :param dict constructor_kwargs: additional keyword arguments to construct an upstream
        :class:`jaxns.NestedSampler` instance.
    :param dict termination_kwargs: keyword arguments to terminate the sampler. Please
        refer to the upstream :meth:`jaxns.NestedSampler.__call__` method.

    **Example**

    .. doctest::

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.contrib.nested_sampling import NestedSampler

        >>> true_coefs = jnp.array([1., 2., 3.])
        >>> data = random.normal(random.PRNGKey(0), (2000, 3))
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(1))
        >>>
        >>> def model(data, labels):
        ...     coefs = numpyro.sample('coefs', dist.Normal(0, 1).expand([3]))
        ...     intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
        ...     return numpyro.sample('y', dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)),
        ...                           obs=labels)
        >>>
        >>> ns = NestedSampler(model)
        >>> ns.run(random.PRNGKey(2), data, labels)
        >>> samples = ns.get_samples(random.PRNGKey(3), num_samples=1000)
        >>> assert jnp.mean(jnp.abs(samples['intercept'])) < 0.05
        >>> print(jnp.mean(samples['coefs'], axis=0))  # doctest: +SKIP
        [0.93661342 1.95034876 2.86123884]
    """

    def __init__(
        self,
        model,
        *,
        constructor_kwargs=None,
        termination_kwargs=None,
    ):
        self.model = model
        self.constructor_kwargs = (
            constructor_kwargs if constructor_kwargs is not None else {}
        )
        self.termination_kwargs = (
            termination_kwargs if termination_kwargs is not None else {}
        )
        self._samples = None
        self._log_weights = None
        self._results: NestedSamplerResults | None = None

    def run(self, rng_key, *args, **kwargs):
        """
        Run the nested samplers and collect weighted samples.

        :param random.PRNGKey rng_key: Random number generator key to be used for the sampling.
        :param args: The arguments needed by the `model`.
        :param kwargs: The keyword arguments needed by the `model`.
        """
        rng_sampling, rng_predictive = random.split(rng_key)
        # reparam the model so that latent sites have Uniform(0, 1) priors
        prototype_trace = trace(seed(self.model, rng_key)).get_trace(*args, **kwargs)
        param_names = [
            site["name"]
            for site in prototype_trace.values()
            if site["type"] == "sample"
            and not site["is_observed"]
            and site["infer"].get("enumerate", "") != "parallel"
        ]
        deterministics = [
            site["name"]
            for site in prototype_trace.values()
            if site["type"] == "deterministic"
        ]
        reparam_model = reparam(
            self.model, config={k: UniformReparam() for k in param_names}
        )

        # enable enumerate if needed
        has_enum = any(
            site["type"] == "sample"
            and site["infer"].get("enumerate", "") == "parallel"
            for site in prototype_trace.values()
        )
        if has_enum:
            from numpyro.contrib.funsor import enum, log_density as log_density_

            max_plate_nesting = _guess_max_plate_nesting(prototype_trace)
            _validate_model(prototype_trace)
            reparam_model = enum(reparam_model, -max_plate_nesting - 1)
        else:
            log_density_ = log_density

        # Jaxns requires loglikelihood function to have explicit signatures.
        local_dict = {}
        loglik_fn_def = """def loglik_fn({}):\n
        \tparams = dict({})\n
        \treturn log_density_(reparam_model, args, kwargs, params)[0]
        """.format(
            ", ".join([f"{name}_base" for name in param_names]),
            ", ".join([f"{name}_base={name}_base" for name in param_names]),
        )
        exec(loglik_fn_def, locals(), local_dict)
        loglik_fn = local_dict["loglik_fn"]

        # use NestedSampler with identity prior chain
        def prior_model():
            params = []
            for name in param_names:
                shape = prototype_trace[name]["fn"].shape()
                param = yield Prior(
                    tfpd.Uniform(low=jnp.zeros(shape), high=jnp.ones(shape)),
                    name=name + "_base",
                )
                params.append(param)
            return tuple(params)

        model = Model(prior_model=prior_model, log_likelihood=loglik_fn)

        default_constructor_kwargs = dict(
            num_live_points=model.U_ndims * 25,
            num_parallel_workers=1,
            max_samples=1e4,
        )
        default_termination_kwargs = dict(dlogZ=1e-4)
        # Fill-in missing values with defaults. This allows user to inspect what was actually used by inspecting
        # these dictionaries
        list(
            map(
                lambda item: self.constructor_kwargs.setdefault(*item),
                default_constructor_kwargs.items(),
            )
        )
        list(
            map(
                lambda item: self.termination_kwargs.setdefault(*item),
                default_termination_kwargs.items(),
            )
        )

        default_ns = DefaultNestedSampler(
            model=model,
            **self.constructor_kwargs,
        )

        termination_reason, state = default_ns(
            rng_sampling, term_cond=TerminationCondition(**self.termination_kwargs)
        )
        results = default_ns.to_results(
            termination_reason=termination_reason, state=state
        )

        # transform base samples back to original domains
        # Here we only transform the first valid num_samples samples
        # NB: the number of weighted samples obtained from jaxns is results.num_samples
        # and only the first num_samples values of results.samples are valid.
        num_samples = results.total_num_samples
        samples = results.samples
        predictive = Predictive(
            reparam_model, samples, return_sites=param_names + deterministics
        )
        samples = predictive(rng_predictive, *args, **kwargs)
        # replace base samples in jaxns results by transformed samples
        self._results = results._replace(samples=samples)

    def get_samples(self, rng_key, num_samples):
        """
        Draws samples from the weighted samples collected from the run.

        :param random.PRNGKey rng_key: Random number generator key to be used to draw samples.
        :param int num_samples: The number of samples.
        :return: a dict of posterior samples
        """
        if self._results is None:
            raise RuntimeError(
                "NestedSampler.run(...) method should be called first to obtain results."
            )
        weighted_samples, sample_weights = self.get_weighted_samples()
        return resample(
            rng_key, weighted_samples, sample_weights, S=num_samples, replace=True
        )

    def get_weighted_samples(self):
        """
        Gets weighted samples and their corresponding log weights.
        """
        if self._results is None:
            raise RuntimeError(
                "NestedSampler.run(...) method should be called first to obtain results."
            )

        return self._results.samples, self._results.log_dp_mean

    def print_summary(self):
        """
        Print summary of the result. This is a wrapper of :func:`jaxns.utils.summary`.
        """
        if self._results is None:
            raise RuntimeError(
                "NestedSampler.run(...) method should be called first to obtain results."
            )
        summary(self._results)

    def diagnostics(self):
        """
        Plot diagnostics of the result. This is a wrapper of :func:`jaxns.plotting.plot_diagnostics`
        and :func:`jaxns.plotting.plot_cornerplot`.
        """
        if self._results is None:
            raise RuntimeError(
                "NestedSampler.run(...) method should be called first to obtain results."
            )
        plot_diagnostics(self._results)
        plot_cornerplot(self._results)
