import argparse
import math
import time
from functools import partial
import pickle

import numpy as np

from jax import grad, vmap, jit
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from jax.lax import while_loop, stop_gradient

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, HMCGibbs, init_to_value
from numpyro.util import enable_x64


BETA_COV = 1.0


def forward(alpha, x):
    return x + (2.0 / 3.0) * alpha * jnp.power(x, 3.0) + 0.2 * jnp.square(alpha) * jnp.power(x, 5.0)

def cond_fn(val):
    return (val[2] > 1.0e-6) & (val[3] < 200)

def body_fn(alpha, val):
    x, y, _, i = val
    f = partial(forward, alpha)
    df = grad(f)
    delta = (f(x) - y) / df(x)
    x = x - delta
    return (x, y, jnp.fabs(delta), i + 1)

@jit
def inverse(alpha, y):
    return while_loop(cond_fn, partial(body_fn, alpha), (y, y, 9.9e9, 0))[0]

def jacobian_and_inverse(alpha, Y):
    inv = partial(inverse, alpha)
    Y_tilde = vmap(lambda y: stop_gradient(inv(y)))(Y)
    log_det_jacobian = -2.0 * jnp.sum(jnp.log(1.0 + alpha * jnp.square(Y_tilde)))
    return Y_tilde, log_det_jacobian


def model(X, Y):
    N, P = X.shape

    noise = numpyro.sample("noise", dist.LogNormal(0.0, 3.0))

    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), math.sqrt(BETA_COV) * jnp.ones(P)))
    #bias = numpyro.sample("bias", dist.Normal(0.0, 1.0))
    betaX = jnp.sum(beta * X, axis=-1)
    #betaX = bias + jnp.sum(beta * X, axis=-1)

    flow = True

    if flow:
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 0.5))
        #inv = partial(inverse, alpha)
        #Y_tilde = vmap(lambda y: stop_gradient(inv(y)))(Y)
        Y_tilde, jacobian = jacobian_and_inverse(alpha, Y)

        numpyro.factor("jacobian", jacobian)
        numpyro.sample("Y", dist.Normal(betaX, noise), obs=Y_tilde)
    else:
        numpyro.sample("Y", dist.Normal(betaX, noise), obs=Y)


def _gibbs_fn(X, Y, rng_key, gibbs_sites, hmc_sites):
    N, P = X.shape

    sigma = hmc_sites['noise']
    alpha = hmc_sites['alpha']
    inv = partial(inverse, alpha)
    Y_tilde = vmap(lambda y: stop_gradient(inv(y)))(Y)

    betaX = jnp.sum(gibbs_sites['beta'] * X, axis=-1)

    X_Y_tilde = jnp.sum(X * Y_tilde[:, None], axis=0)

    XX = np.matmul(np.transpose(X), X)

    sigma_sq = jnp.square(sigma)
    covar_inv = XX / sigma_sq + jnp.eye(P) / BETA_COV

    L = cho_factor(covar_inv, lower=True)[0]
    L_inv = solve_triangular(L, jnp.eye(P), lower=True)
    loc = cho_solve((L, True), X_Y_tilde) / sigma_sq

    beta_proposal = dist.MultivariateNormal(loc=loc, scale_tril=L_inv).sample(rng_key)

    return {'beta': beta_proposal}


def run_inference(args, rng_key, X, Y):
    #if False:
    #    samples = pickle.load(open('samples.pkl', "rb" ))
    #    values = {}
    #    for k, v in samples.items():
            #if k in ['mean_phi']: continue
    #        if k != 'phi': continue
    #        values[k] = np.mean(v, axis=0)
    #        print(k, values[k].shape)

    #init_strategy = init_to_value(values=values)
    #init_strategy = init_to_value(values={"kernel_var": 0.5, "kernel_noise": 0.1, "phi_sigma": 0.01,
    #                                      "kernel_length": 2.0, "phi": Y / 0.01})

    if args.strategy == "gibbs":
        gibbs_fn = partial(_gibbs_fn, X, Y)
        hmc_kernel = NUTS(model, max_tree_depth=6)#, init_strategy=init_strategy)
        kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['beta'])
        mcmc = MCMC(kernel, args.num_warmup, args.num_samples, progress_bar=True)
    else:
        hmc_kernel = NUTS(model, max_tree_depth=6)#, init_strategy=init_strategy)
        mcmc = MCMC(hmc_kernel, args.num_warmup, args.num_samples, progress_bar=True)

    start = time.time()
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary(exclude_deterministic=False)
    print('\nMCMC elapsed time:', time.time() - start)

    return mcmc.get_samples()


# create artificial regression dataset
def get_data(N=50, P=30, sigma_obs=0.02):
    np.random.seed(0)

    X = np.random.randn(N * P).reshape((N, P))
    Y = 1.0 * X[:, 0] - 0.5 * X[:, 1]
    #Y = np.power(2.4 * X[:, 0] + 1.2 * X[:, 1], 3.0)
    Y += sigma_obs * np.random.randn(N)
    Y = forward(0.5, Y)
    #Y -= jnp.mean(Y)
    #Y /= jnp.std(Y)

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y


def main(args):
    X, Y = get_data(N=args.num_data, P=args.P)

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(args, rng_key, X, Y)
    print(samples.keys())

    #with open('samples.pkl', "wb") as f:
    #    pickle.dump(samples, f)

    #mean_phi = np.mean(samples['mean_phi'], axis=0)

    #from scipy.stats import pearsonr

    #r = pearsonr(mean_phi, Y)
    #print("r",r)
    #print("mean_phi/Y", mean_phi/Y)



if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.4.1')
    parser = argparse.ArgumentParser(description="non-linear horseshoe")
    parser.add_argument("-n", "--num-samples", default=15000, type=int)
    parser.add_argument("--num-warmup", default=5000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--num-data", default=128, type=int)
    parser.add_argument("--strategy", default="gibbs", type=str, choices=["nuts", "gibbs"])
    parser.add_argument("--P", default=4, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    enable_x64()

    main(args)
