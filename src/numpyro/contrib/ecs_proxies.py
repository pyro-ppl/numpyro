# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict, namedtuple
import warnings

from jax import hessian, jacobian, lax, numpy as jnp, random
from jax.flatten_util import ravel_pytree

from numpyro.distributions.transforms import biject_to
from numpyro.handlers import block, substitute, trace

TaylorTwoProxyState = namedtuple(
    "TaylorProxyState",
    "ref_subsample_log_liks,"
    "ref_subsample_log_lik_grads,"
    "ref_subsample_log_lik_hessians",
)

TaylorOneProxyState = namedtuple(
    "TaylorOneProxyState", "ref_subsample_log_liks," "ref_subsample_log_lik_grads,"
)


def perturbed_method(subsample_plate_sizes, proxy_fn):
    def estimator(likelihoods, params, gibbs_state):
        subsample_log_liks = defaultdict(float)
        for fn, value, name, subsample_dim in likelihoods.values():
            subsample_log_liks[name] += _sum_all_except_at_dim(
                fn.log_prob(value), subsample_dim
            )

        log_lik_sum = 0.0

        proxy_value_all, proxy_value_subsample = proxy_fn(
            params, subsample_log_liks.keys(), gibbs_state
        )

        for (
            name,
            subsample_log_lik,
        ) in subsample_log_liks.items():  # loop over all subsample sites
            n, m = subsample_plate_sizes[name]

            diff = subsample_log_lik - proxy_value_subsample[name]

            unbiased_log_lik = proxy_value_all[name] + n * jnp.mean(diff)
            variance = n**2 / m * jnp.var(diff)
            log_lik_sum += unbiased_log_lik - 0.5 * variance
        return log_lik_sum

    return estimator


def _sum_all_except_at_dim(x, dim):
    x = x.reshape((-1,) + x.shape[dim:]).sum(0)
    return x.reshape(x.shape[:1] + (-1,)).sum(-1)


def _update_block(rng_key, num_blocks, subsample_idx, plate_size):
    size, subsample_size = plate_size
    rng_key, subkey, block_key = random.split(rng_key, 3)
    block_size = (subsample_size - 1) // num_blocks + 1
    pad = block_size - (subsample_size - 1) % block_size - 1

    chosen_block = random.randint(block_key, shape=(), minval=0, maxval=num_blocks)
    new_idx = random.randint(subkey, minval=0, maxval=size, shape=(block_size,))
    subsample_idx_padded = jnp.pad(subsample_idx, (0, pad))
    start = chosen_block * block_size
    subsample_idx_padded = lax.dynamic_update_slice_in_dim(
        subsample_idx_padded, new_idx, start, 0
    )
    return rng_key, subsample_idx_padded[:subsample_size], pad, new_idx, start


def block_update(plate_sizes, num_blocks, rng_key, gibbs_sites, gibbs_state):
    u_new = {}
    for name, subsample_idx in gibbs_sites.items():
        rng_key, u_new[name], *_ = _update_block(
            rng_key, num_blocks, subsample_idx, plate_sizes[name]
        )
    return u_new, gibbs_state


def _block_update_proxy(num_blocks, rng_key, gibbs_sites, plate_sizes):
    u_new = {}
    pads = {}
    new_idxs = {}
    starts = {}
    for name, subsample_idx in gibbs_sites.items():
        rng_key, u_new[name], pads[name], new_idxs[name], starts[name] = _update_block(
            rng_key, num_blocks, subsample_idx, plate_sizes[name]
        )
    return u_new, pads, new_idxs, starts


def taylor_proxy(reference_params, degree):
    """Control variate for unbiased log likelihood estimation using a Taylor expansion around a reference
    parameter. Suggested for subsampling in [1].

    :param dict reference_params: Model parameterization at MLE or MAP-estimate.
    :param degree: number of terms in the Taylor expansion, either one or two.

    **References:**

    [1] On Markov chain Monte Carlo Methods For Tall Data
        Bardenet., R., Doucet, A., Holmes, C. (2017)
    """

    def construct_proxy_fn(
        prototype_trace,
        subsample_plate_sizes,
        model,
        model_args,
        model_kwargs,
        num_blocks=1,
    ):
        ref_params = {
            name: (
                biject_to(prototype_trace[name]["fn"].support).inv(value)
                if prototype_trace[name]["type"] == "sample"
                else value
            )
            for name, value in reference_params.items()
        }

        ref_params_flat, unravel_fn = ravel_pytree(ref_params)

        def log_likelihood(params_flat, subsample_indices=None):
            if subsample_indices is None:
                subsample_indices = {
                    k: jnp.arange(v[0]) for k, v in subsample_plate_sizes.items()
                }
            params = unravel_fn(params_flat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = {
                    name: (
                        biject_to(prototype_trace[name]["fn"].support)(value)
                        if prototype_trace[name]["type"] == "sample"
                        else value
                    )
                    for name, value in params.items()
                }
                with (
                    block(),
                    trace() as tr,
                    substitute(data=subsample_indices),
                    substitute(data=params),
                ):
                    model(*model_args, **model_kwargs)

            log_lik = {}
            for site in tr.values():
                if site["type"] == "sample" and site["is_observed"]:
                    for frame in site["cond_indep_stack"]:
                        if frame.name in log_lik:
                            log_lik[frame.name] += _sum_all_except_at_dim(
                                site["fn"].log_prob(site["value"]), frame.dim
                            )
                        elif frame.name in subsample_indices:
                            log_lik[frame.name] = _sum_all_except_at_dim(
                                site["fn"].log_prob(site["value"]), frame.dim
                            )
            return log_lik

        def log_likelihood_sum(params_flat, subsample_indices=None):
            return {
                k: v.sum()
                for k, v in log_likelihood(params_flat, subsample_indices).items()
            }

        if degree == 2:
            TPState = TaylorTwoProxyState
        elif 1:
            TPState = TaylorOneProxyState
        else:
            raise ValueError("Taylor proxy only defined for first and second degree.")

        # those stats are dict keyed by subsample names
        ref_sum_log_lik = log_likelihood_sum(ref_params_flat)
        ref_sum_log_lik_grads = jacobian(log_likelihood_sum)(ref_params_flat)

        if degree == 2:
            ref_sum_log_lik_hessians = hessian(log_likelihood_sum)(ref_params_flat)

        def gibbs_init(rng_key, gibbs_sites):
            ref_subsamples_taylor = [
                log_likelihood(ref_params_flat, gibbs_sites),
                jacobian(log_likelihood)(ref_params_flat, gibbs_sites),
            ]

            if degree == 2:
                ref_subsamples_taylor.append(
                    hessian(log_likelihood)(ref_params_flat, gibbs_sites)
                )

            return TPState(*ref_subsamples_taylor)

        def gibbs_update(rng_key, gibbs_sites, gibbs_state):
            u_new, pads, new_idxs, starts = _block_update_proxy(
                num_blocks, rng_key, gibbs_sites, subsample_plate_sizes
            )

            new_states = defaultdict(dict)
            new_ref_subsample_taylor = [
                log_likelihood(ref_params_flat, new_idxs),
                jacobian(log_likelihood)(ref_params_flat, new_idxs),
            ]

            if degree == 2:
                new_ref_subsample_taylor.append(
                    hessian(log_likelihood)(ref_params_flat, new_idxs)
                )

            last_ref_subsample_taylor = list(gibbs_state._asdict().values())

            for stat, new_block_values, last_values in zip(
                TPState._fields,
                new_ref_subsample_taylor,
                last_ref_subsample_taylor,
            ):
                for name, subsample_idx in gibbs_sites.items():
                    size, subsample_size = subsample_plate_sizes[name]
                    pad, start = pads[name], starts[name]
                    new_value = jnp.pad(
                        last_values[name],
                        [(0, pad)] + [(0, 0)] * (jnp.ndim(last_values[name]) - 1),
                    )
                    new_value = lax.dynamic_update_slice_in_dim(
                        new_value, new_block_values[name], start, 0
                    )
                    new_states[stat][name] = new_value[:subsample_size]

            gibbs_state = TPState(**new_states)
            return u_new, gibbs_state

        def proxy_fn(params, subsample_lik_sites, gibbs_state):
            params_flat, _ = ravel_pytree(params)
            params_diff = params_flat - ref_params_flat

            ref_subsample_log_liks = gibbs_state.ref_subsample_log_liks
            ref_subsample_log_lik_grads = gibbs_state.ref_subsample_log_lik_grads
            if degree == 2:
                ref_subsample_log_lik_hessians = (
                    gibbs_state.ref_subsample_log_lik_hessians
                )

            proxy_sum = defaultdict(float)
            proxy_subsample = defaultdict(float)
            for name in subsample_lik_sites:
                proxy_subsample[name] = ref_subsample_log_liks[name] + jnp.dot(
                    ref_subsample_log_lik_grads[name], params_diff
                )
                high_order_terms = 0.0
                if degree == 2:
                    high_order_terms = 0.5 * jnp.dot(
                        jnp.dot(ref_subsample_log_lik_hessians[name], params_diff),
                        params_diff,
                    )

                proxy_subsample[name] = proxy_subsample[name] + high_order_terms

                proxy_sum[name] = ref_sum_log_lik[name] + jnp.dot(
                    ref_sum_log_lik_grads[name], params_diff
                )

                high_order_terms = 0.0
                if degree == 2:
                    high_order_terms = 0.5 * jnp.dot(
                        jnp.dot(ref_sum_log_lik_hessians[name], params_diff),
                        params_diff,
                    )
                proxy_sum[name] = proxy_sum[name] + high_order_terms

            return proxy_sum, proxy_subsample

        return proxy_fn, gibbs_init, gibbs_update

    return construct_proxy_fn
