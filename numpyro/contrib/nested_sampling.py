# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch
import warnings

from jax import config, nn, random, tree_util
import jax.numpy as jnp

try:
    # jaxns changes the default precision to double precision
    # so here we undo that action
    use_x64 = config.jax_enable_x64

    from jaxns.nested_sampling import NestedSampler as OrigNestedSampler
    from jaxns.plotting import plot_cornerplot, plot_diagnostics
    from jaxns.prior_transforms.common import ContinuousPrior
    from jaxns.prior_transforms.prior_chain import PriorChain, UniformBase
    from jaxns.utils import summary

    config.update("jax_enable_x64", use_x64)
except ImportError as e:
    raise ImportError(
        "To use this module, please install `jaxns` package. It can be"
        " installed with `pip install jaxns`"
    ) from e

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import Reparam
from numpyro.infer.util import _guess_max_plate_nesting, _validate_model, log_density

__all__ = ["NestedSampler"]


class UniformPrior(ContinuousPrior):
    def __init__(self, name, shape):
        prior_base = UniformBase(shape, jnp.result_type(float))
        super().__init__(name, shape, parents=[], tracked=True, prior_base=prior_base)

    def transform_U(self, U, **kwargs):
        return U


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
    :param int num_live_points: the number of live points. As a rule-of-thumb, we should
        allocate around 50 live points per possible mode.
    :param int max_samples: the maximum number of iterations and samples
    :param str sampler_name: either "slice" (default value) or "multi_ellipsoid"
    :param int depth: an integer which determines the maximum number of ellipsoids to
        construct via hierarchical splitting (typical range: 3 - 9, default to 5)
    :param int num_slices: the number of slice sampling proposals at each sampling step
        (typical range: 1 - 5, default to 5)
    :param float termination_frac: termination condition (typical range: 0.001 - 0.01)
        (default to 0.01).

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
        num_live_points=1000,
        max_samples=100000,
        sampler_name="slice",
        depth=5,
        num_slices=5,
        termination_frac=0.01
    ):
        self.model = model
        self.num_live_points = num_live_points
        self.max_samples = max_samples
        self.termination_frac = termination_frac
        self.sampler_name = sampler_name
        self.depth = depth
        self.num_slices = num_slices
        self._samples = None
        self._log_weights = None
        self._results = None

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

        def loglik_fn(**params):
            return log_density_(reparam_model, args, kwargs, params)[0]

        # use NestedSampler with identity prior chain
        prior_chain = PriorChain()
        for name in param_names:
            prior = UniformPrior(name + "_base", prototype_trace[name]["fn"].shape())
            prior_chain.push(prior)
        # XXX: the `marginalised` keyword in jaxns can be used to get expectation of some
        # quantity over posterior samples; it can be helpful to expose it in this wrapper
        ns = OrigNestedSampler(
            loglik_fn,
            prior_chain,
            sampler_name=self.sampler_name,
            sampler_kwargs={"depth": self.depth, "num_slices": self.num_slices},
            max_samples=self.max_samples,
            num_live_points=self.num_live_points,
            collect_samples=True,
        )
        # some places of jaxns uses float64 and raises some warnings if the default dtype is
        # float32, so we suppress them here to avoid confusion
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*will be truncated to dtype float32.*"
            )
            results = ns(rng_sampling, termination_frac=self.termination_frac)
        # transform base samples back to original domains
        # Here we only transform the first valid num_samples samples
        # NB: the number of weighted samples obtained from jaxns is results.num_samples
        # and only the first num_samples values of results.samples are valid.
        num_samples = results.num_samples
        samples = tree_util.tree_map(lambda x: x[:num_samples], results.samples)
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

        samples, log_weights = self.get_weighted_samples()
        p = nn.softmax(log_weights)
        idx = random.choice(rng_key, log_weights.shape[0], (num_samples,), p=p)
        return {k: v[idx] for k, v in samples.items()}

    def get_weighted_samples(self):
        """
        Gets weighted samples and their corresponding log weights.
        """
        if self._results is None:
            raise RuntimeError(
                "NestedSampler.run(...) method should be called first to obtain results."
            )

        num_samples = self._results.num_samples
        return self._results.samples, self._results.log_p[:num_samples]

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
