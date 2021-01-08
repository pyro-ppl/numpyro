from functools import singledispatch

import numpy as np

from jax import device_get, nn, random, tree_multimap
import jax.numpy as jnp
from jaxns.nested_sampling import NestedSampler as OrigNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.prior_transforms import PriorChain, PriorTransform

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer.reparam import Reparam
from numpyro.infer import Predictive, log_likelihood


__all__ = ["NestedSampler"]


class ShapeTransform(PriorTransform):
    """
    Reshape a uniform vector to a target shape.
    """
    def __init__(self, name, to_shape):
        self._to_shape = to_shape
        super().__init__(name, np.prod(to_shape) if to_shape else 1, [], tracked=True)

    @property
    def to_shape(self):
        return self._to_shape

    def forward(self, U, **kwargs):
        return jnp.reshape(U, self.to_shape)


@singledispatch
def uniform_reparam_transform(d):
    """
    A helper for :class:`UniformReparam` to get the transform that transforms
    a uniform distribution over a unit hypercube to the target distribution `d`.
    """
    # use either `icdf` or `quantile` method if available
    if hasattr(d, "icdf") and callable(d.icdf):
        return d.icdf
    elif hasattr(d, "quantile") and callable(d.quantile):
        return d.quantile

    if isinstance(d, dist.TransformedDistribution):
        outer_transform = dist.transforms.ComposeTransform(d.transforms)
        return lambda q: outer_transform(uniform_reparam_transform(d.base_dist)(q))

    if isinstance(d, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
        return lambda q: uniform_reparam_transform(d.base_dist)(q)

    try:  # defer to TFP implementation
        import numpyro.contrib.tfp.distributions as tfd

        tfd_name = getattr(tfd, type(d).__name__)
        tfd_dist = tfd_name(**{k: getattr(d, k) for k in d.arg_constraints})
        return tfd_dist.quantile
    except ImportError:
        raise ImportError("Many NumPyro distributions do not have `.icdf(...)` method"
                          " to derive this uniform reparam transform. You might need to"
                          " install tensorflow_probability with: `pip install tfp-nightly`")
    except AttributeError:
        raise NotImplementedError


@uniform_reparam_transform.register(dist.MultivariateNormal)
def _(d):
    outer_transform = dist.transforms.LowerCholeskyAffine(d.loc, d.scale_tril)
    return lambda q: outer_transform(dist.Normal(0, 1).icdf(q))


@uniform_reparam_transform.register(dist.BernoulliLogits)
@uniform_reparam_transform.register(dist.BernoulliProbs)
def _(d):
    return lambda q: q < d.probs


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
        fn, event_dim = self._unwrap(fn)
        fn, _ = self._unexpand(fn)
        transform = uniform_reparam_transform(fn)

        x = numpyro.sample("{}_base".format(name),
                           dist.Uniform(0, 1).expand(shape).to_event(event_dim))
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
    :param depth: an integer which determines the maximum number of ellipsoids to
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
    def __init__(self, model, *, num_live_points=1000, max_samples=1e5,
                 depth=3, num_slices=5, termination_frac=0.001):
        self.model = model
        self.num_live_points = num_live_points
        self.max_samples = max_samples
        self.termination_frac = termination_frac
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
                       if site["type"] == "sample" and not site["is_observed"]]
        deterministics = [site["name"] for site in prototype_trace.values()
                          if site["type"] == "deterministic"]
        reparam_model = reparam(self.model, config={k: UniformReparam() for k in param_names})

        # define the likelihood of the model
        def loglik_fn(**params):
            loglik_dict = log_likelihood(reparam_model, params, *args, batch_ndims=0, **kwargs)
            return sum(x.sum() for x in loglik_dict.values())

        # use NestedSampler with identity prior chain
        prior_chain = PriorChain()
        for name in param_names:
            shape_transform = ShapeTransform(name + "_base", prototype_trace[name]["fn"].shape())
            prior_chain.push(shape_transform)
        ns = OrigNestedSampler(loglik_fn, prior_chain, sampler_name='slice')
        results = ns(rng_sampling, self.num_live_points, collect_samples=True,
                     max_samples=self.max_samples, termination_frac=self.termination_frac,
                     sampler_kwargs={"depth": self.depth, "num_slices": self.num_slices})
        num_samples = int(device_get(results.num_samples))

        # transform base samples back to original domains
        base_samples = {k: v[:num_samples] for k, v in results.samples.items()}
        predictive = Predictive(reparam_model, base_samples,
                                return_sites=param_names + deterministics)
        self._samples = predictive(rng_predictive, *args, **kwargs)
        self._log_weights = results.log_p[:num_samples]
        self._results = results._replace(samples={k: v for k, v in self._samples.items()
                                                  if k in param_names})

        print("Number of weighted samples:", num_samples)
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

    def diagnostics(self, cornerplot=True):
        """
        Plot diagnostics of the run.

        :param bool concerplot: whether to plot a cornerplot of the posterior samples
        """
        plot_diagnostics(self._results)
        if cornerplot:
            plot_cornerplot(self._results)
