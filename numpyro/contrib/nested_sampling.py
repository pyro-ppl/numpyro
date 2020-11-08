from functools import singledispatch

import numpy as np

from jax import device_get, random
import jax.numpy as jnp
from jaxns.nested_sampling import NestedSampler as OrigNestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.prior_transforms import PriorChain, PriorTransform

import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.handlers import reparam, seed, trace
from numpyro.infer.reparam import Reparam
from numpyro.infer import Predictive, log_likelihood


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

    try:  # defer to TFP implementation
        import numpyro.contrib.tfp.distributions as tfd

        tfd_name = getattr(tfd, type(d).__name__)
        tfd_dist = tfd_name(**{k: getattr(d, k) for k in d.arg_constraints})
        return tfd_dist.quantile
    except AttributeError:
        raise NotImplementedError


@uniform_reparam_transform.register(dist.MultivariateNormal)
def _(d):
    outer_transform = dist.transforms.LowerCholeskyAffine(d.loc, d.scale_tril)
    return lambda q: outer_transform(dist.Normal(0, 1).icdf(q))


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
    A wrapper for :class:`jaxns.nested_sampling.NestedSampler`.
    """
    def __init__(self, model, num_live_points, *, sampler_name='slice', max_samples=1e5, termination_frac=0.01):
        self.model = model
        self.num_live_points = num_live_points
        self.sampler_name = sampler_name
        self.max_samples = max_samples
        self.termination_frac = termination_frac
        self._num_samples = None
        self._results = None

    @property
    def num_samples(self):
        return self._num_samples

    def run(self, rng_key, *args, **kwargs):
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

        # use NestedSampler with prior chain
        prior_chain = PriorChain()
        for name in param_names:
            shape_transform = ShapeTransform(name + "_base", prototype_trace[name]["fn"].shape())
            prior_chain.push(shape_transform)
        ns = OrigNestedSampler(loglik_fn, prior_chain, sampler_name=self.sampler_name)
        results = ns(rng_sampling, self.num_live_points, collect_samples=True,
                     max_samples=self.max_samples, termination_frac=self.termination_frac)
        self._num_samples = int(device_get(results.num_samples))

        # transform base samples back to original domains
        base_samples = {k: v[:self._num_samples] for k, v in results.samples.items()}
        predictive = Predictive(reparam_model, base_samples,
                                return_sites=param_names + deterministics)
        self._samples = predictive(rng_predictive, *args, **kwargs)
        self._results = results._replace(samples={k: v for k, v in self._samples.items()
                                                  if k in param_names})

    def get_samples(self):
        return self._samples

    def get_results(self):
        return self._results

    def print_summary(self, prob=0.9, exclude_deterministic=True):
        samples = {k: v for k, v in self._samples.items()
                   if k in self._results.samples or not exclude_deterministic}
        print_summary(samples, prob=prob, group_by_chain=False)

    def diagnostics(self):
        plot_diagnostics(self._results)
        plot_cornerplot(self._results)
