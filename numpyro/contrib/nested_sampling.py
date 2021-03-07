from functools import singledispatch

from jax import device_get, nn, random, tree_multimap
import jax.numpy as jnp
from jaxns.nested_sampling import NestedSampler as OrigNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.prior_transforms import ContinuousPrior, PriorChain

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer.reparam import Reparam
from numpyro.infer import Predictive
from numpyro.infer.util import _guess_max_plate_nesting, _validate_model, log_density


__all__ = ["NestedSampler"]


class UniformPrior(ContinuousPrior):
    def __init__(self, name, shape):
        super().__init__(name, shape, parents=[], tracked=True)

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

    if isinstance(d, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
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
        # NB: quantile is not implemented in tfd.Gamma
        # so this will raise an NotImplementedError for now.
        # We will need scipy.special.gammaincinv, which is not available yet in JAX
        # see issue: https://github.com/google/jax/issues/5350
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
        event_shape = fn.event_shape
        fn, expand_shape, event_dim = self._unwrap(fn)
        transform = uniform_reparam_transform(fn)

        x = numpyro.sample("{}_base".format(name),
                           dist.Uniform(0, 1).expand(expand_shape + event_shape).to_event(event_dim).mask(False))
        # Simulate a numpyro.deterministic() site.
        return None, transform(x)


class NestedSampler:
    """
    A wrapper for :class:`jaxns.nested_sampling.NestedSampler`, a nested sampling
    package based on JAX.

    See reference [1] for details on the meaning of each parameter.
    Please consider citing this reference if you use the nested sampler in your research.

    **References**

    1. *JAXNS: a high-performance nested sampling package based on JAX*,
       Joshua G. Albert (https://arxiv.org/abs/2012.15286)

    :param callable model: a call with NumPyro primitives
    :param int num_live_points: the number of live points. As a rule-of-thumb, we should
        allocate around 50 live points per possible mode.
    :param int max_samples: the maximum number of iterations and samples
    :param str sampler_name: either "slice" (default value) or "multi_ellipsoid"
    :param int depth: an integer which determines the maximum number of ellipsoids to
        construct via hierarchical splitting (typical range: 3 - 9)
    :param int num_slices: the number of slice sampling proposals at each sampling step
        (typical range: 1 - 5)
    :param float termination_frac: termination condition (typical range: 0.001 - 0.01)

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
        [0.91239292 1.91636061 2.81830897]
    """
    def __init__(self, model, *, num_live_points=1000, max_samples=100000,
                 sampler_name="slice", depth=3, num_slices=5, termination_frac=0.01):
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
        param_names = [site["name"] for site in prototype_trace.values()
                       if site["type"] == "sample" and not site["is_observed"]
                       and site["infer"].get("enumerate", "") != "parallel"]
        deterministics = [site["name"] for site in prototype_trace.values()
                          if site["type"] == "deterministic"]
        reparam_model = reparam(self.model, config={k: UniformReparam() for k in param_names})

        # enable enumerate if needed
        has_enum = any(site["type"] == "sample" and site["infer"].get("enumerate", "") == "parallel"
                       for site in prototype_trace.values())
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
        ns = OrigNestedSampler(loglik_fn, prior_chain, sampler_name=self.sampler_name,
                               sampler_kwargs={"depth": self.depth, "num_slices": self.num_slices},
                               max_samples=self.max_samples,
                               num_live_points=self.num_live_points,
                               collect_samples=True)
        results = ns(rng_sampling, termination_frac=self.termination_frac)

        num_samples = int(device_get(results.num_samples))
        log_weights = results.log_p[:num_samples]
        base_samples = {k: v[:num_samples] for k, v in results.samples.items()}
        # filter out samples with -inf weights and invalid values
        valid_masks = [jnp.isfinite(log_weights)] + \
            [((v > 0) & (v < 1)).reshape((num_samples, -1)).all(-1)
             for v in base_samples.values()]
        mask = jnp.stack(valid_masks, -1).all(-1)
        self._log_weights = log_weights[mask]

        # transform base samples back to original domains
        predictive = Predictive(reparam_model, base_samples,
                                return_sites=param_names + deterministics)
        samples = predictive(rng_predictive, *args, **kwargs)
        self._samples = {k: v[mask] for k, v in samples.items()}

        # replace base samples in jaxns results by transformed samples
        self._results = results._replace(samples=samples)

        print("Number of weighted samples:", mask.sum())
        print("Effective sample size:", round(float(device_get(results.ESS)), 1))

    def get_samples(self, rng_key, num_samples):
        """
        Draws samples from the weighted samples collected from the run.

        :param random.PRNGKey rng_key: Random number generator key to be used to draw samples.
        :param int num_samples: The number of samples.
        :return: a dict of posterior samples
        """
        p = nn.softmax(self._log_weights)
        idx = random.choice(rng_key, self._log_weights.shape[0], (num_samples,), p=p)
        return tree_multimap(lambda x: x[idx], self._samples)

    def diagnostics(self):
        """
        Plot diagnostics of the run.
        """
        plot_diagnostics(self._results)
        plot_cornerplot(self._results)
