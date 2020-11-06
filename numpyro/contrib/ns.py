from numpyro.handlers import reparam
from numpyro.infer.reparam import Reparam
from numpyro.infer import log_likelihood

from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain


class UnitCubeReparam(Reparam):
    # support distributions with quantiles implementation
    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        fn, batch_shape = self._unexpand(fn)
        # if fn is transformed distribution, get transform and base_dist, reparam the base_dist
        # TODO: extract quantiles function


class NestedSampling:
    def __init__(self, model, num_live_points, *, sampler_name='slice', **ns_kwargs):
        self.model = model
        self.num_live_points = num_live_points
        self.sampler_name = sampler_name
        self.ns_kwargs = ns_kwargs  # max_samples, termination_frac,...

    def run(self, rng_key, *args, **kwargs):
        # Step 1: reparam the model so that latent sites have Uniform(0, 1) priors
        reparam_model = reparam(self.model,
                                config=lambda msg: None if msg.get("is_observed") else UnitCubeReparam())

        # Step 2: compute the likelihood of the model
        def ll_fn(*params):
            return list(log_likelihood(reparam_model, params, *args, batch_ndims=0, **kwargs).values())[0]

        # Step 3: use NestedSampler with empty prior chain
        prior_chain = PriorChain()
        ns = NestedSampler(ll_fn, prior_chain, collect_samples=True, sampler_name=self.sampler_name)
        results = ns(rng_key, self.num_live_points)
        samples = results["samples"]
        # TODO: transform samples back or rerun reparam_model to get deterministic sites
        return samples
