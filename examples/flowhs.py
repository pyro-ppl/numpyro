import argparse
import math
import time
from functools import partial
import pickle

import numpy as np

from jax import grad, vmap, jit
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_factor, cho_solve
from jax.lax import while_loop, stop_gradient, dynamic_slice_in_dim

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCGibbs, init_to_median
from numpyro.util import enable_x64

from taigapy import default_tc as tc

from sklearn.model_selection import KFold
from scipy.stats import pearsonr

from func import jacobian_and_inverse, forward


def linear_model(X, Y, mask):
    N, P = X.shape

    tau_obs = numpyro.sample("tau_obs", dist.Gamma(1.0, 2.0))
    sigma_obs = 1.0 / jnp.sqrt(tau_obs)

    if not mask:
        xi = numpyro.sample("xi", dist.InverseGamma(0.5, 1.0).mask(mask))
        tau = jnp.sqrt(numpyro.sample("tausq", dist.InverseGamma(0.5, 1.0 / xi).mask(mask)))

        nu = numpyro.sample("nu", dist.InverseGamma(0.5, jnp.ones(P)).mask(mask))
        lam = jnp.sqrt(numpyro.sample("lamsq", dist.InverseGamma(0.5, 1.0 / nu).mask(mask)))
    else:
        tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(P)))

    omega_scale = numpyro.deterministic("omega_scale", tau * lam)
    omega = numpyro.sample("omega", dist.Normal(0.0, omega_scale))
    omegaX = jnp.sum(omega * X, axis=-1)

    numpyro.sample("Y", dist.Normal(omegaX, sigma_obs), obs=Y)


def flow_model(X, Y, mask):
    N, P = X.shape

    tau_obs = numpyro.sample("tau_obs", dist.Gamma(1.0, 2.0))
    sigma_obs = 1.0 / jnp.sqrt(tau_obs)

    if not mask:
        xi = numpyro.sample("xi", dist.InverseGamma(0.5, 1.0).mask(mask))
        tau = jnp.sqrt(numpyro.sample("tausq", dist.InverseGamma(0.5, 1.0 / xi).mask(mask)))

        nu = numpyro.sample("nu", dist.InverseGamma(0.5, jnp.ones(P)).mask(mask))
        lam = jnp.sqrt(numpyro.sample("lamsq", dist.InverseGamma(0.5, 1.0 / nu).mask(mask)))
    else:
        tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(P)))

    omega_scale = numpyro.deterministic("omega_scale", tau * lam)
    omega = numpyro.sample("omega", dist.Normal(0.0, omega_scale))
    omegaX = jnp.sum(omega * X, axis=-1)

    alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5))
    Y_tilde, jacobian = jacobian_and_inverse(alpha, beta, Y)

    numpyro.deterministic("Y_tilde", Y_tilde)

    numpyro.factor("jacobian", jacobian)
    numpyro.sample("Y", dist.Normal(omegaX, sigma_obs), obs=Y_tilde)


def mvn_sample(rng_key, X, D, sigma, alpha, truncation=None):
    N, P = X.shape
    assert D.shape == (P,)
    assert alpha.shape == (N,)
    assert truncation is None or (truncation > 0 and truncation <= P)

    if truncation is not None and truncation < P:
        idx = jnp.argsort(D)
        idx = dynamic_slice_in_dim(idx, P - truncation, truncation)
        Xr = jnp.take(X, idx, -1)
        Dr = jnp.take(D, idx, -1)
    else:
        Xr = X
        Dr = D

    u = dist.Normal(0.0, jnp.sqrt(D)).sample(rng_key)
    delta = dist.Normal(0.0, jnp.ones(N)).sample(rng_key)

    v = jnp.matmul(X, u) / sigma + delta

    X_D_X = jnp.matmul(Xr, jnp.transpose(Xr) * Dr[:, None])
    precision = X_D_X / jnp.square(sigma) + jnp.eye(N)
    Z = jnp.median(jnp.diagonal(precision))

    L = cho_factor(precision / Z, lower=True)[0]
    w = cho_solve((L, True), alpha - v) / Z

    theta = jnp.matmul(jnp.transpose(X) * D[:, None], w) / sigma

    return theta + u

def _linear_gibbs_fn(X, Y, truncation, rng_key, gibbs_sites, hmc_sites):
    N, P = X.shape

    rng_keys = random.split(rng_key, 6)

    sigma_obs = 1.0 / jnp.sqrt(hmc_sites['tau_obs'])
    alpha = Y / sigma_obs
    D = jnp.square(hmc_sites['omega_scale'])
    omega = mvn_sample(rng_keys[0], X, D, sigma_obs, alpha, truncation=truncation)
    omegasq = jnp.square(omega)

    beta = 1.0 / gibbs_sites['nu'] + 0.5 * omegasq / gibbs_sites['tausq']
    lamsq_inverse = dist.Gamma(1.0, beta).sample(rng_keys[1])

    lamsq = 1.0 / lamsq_inverse
    nu = 1.0 / dist.Gamma(1.0, 1.0 + lamsq_inverse).sample(rng_keys[2])

    alpha = 0.5 + 0.5 * P
    beta = 1.0 / gibbs_sites['xi'] + 0.5 * jnp.sum(omegasq / lamsq)
    tausq_inverse = dist.Gamma(alpha, beta).sample(rng_keys[3])
    tausq = 1.0 / tausq_inverse
    xi = 1.0 / dist.Gamma(1.0, 1.0 + tausq_inverse).sample(rng_keys[4])

    #alpha = 1.0 + 0.5 * N
    #omega_mean = jnp.matmul(X, omega)
    #beta = 1.0 + 0.5 * jnp.sum(jnp.square(Y - omega_mean))
    #tau_obs = dist.Gamma(alpha, beta).sample(rng_keys[5])

    return {'omega': omega, 'lamsq': lamsq, 'nu': nu, 'xi': xi, 'tausq': tausq} #, 'tau_obs': tau_obs}


def _gibbs_fn(X, Y, truncation, rng_key, gibbs_sites, hmc_sites):
    N, P = X.shape

    rng_keys = random.split(rng_key, 6)

    sigma_obs = 1.0 / jnp.sqrt(hmc_sites['tau_obs'])
    alpha = hmc_sites['Y_tilde'] / sigma_obs
    D = jnp.square(hmc_sites['omega_scale'])
    omega = mvn_sample(rng_keys[0], X, D, sigma_obs, alpha, truncation=truncation)
    omegasq = jnp.square(omega)

    beta = 1.0 / gibbs_sites['nu'] + 0.5 * omegasq / gibbs_sites['tausq']
    lamsq_inverse = dist.Gamma(1.0, beta).sample(rng_keys[1])

    lamsq = 1.0 / lamsq_inverse
    nu = 1.0 / dist.Gamma(1.0, 1.0 + lamsq_inverse).sample(rng_keys[2])

    alpha = 0.5 + 0.5 * P
    beta = 1.0 / gibbs_sites['xi'] + 0.5 * jnp.sum(omegasq / lamsq)

    tausq_inverse = dist.Gamma(alpha, beta).sample(rng_keys[3])
    tausq = 1.0 / tausq_inverse
    xi = 1.0 / dist.Gamma(1.0, 1.0 + tausq_inverse).sample(rng_keys[4])

    #alpha = 1.0 + 0.5 * N
    #omega_mean = jnp.matmul(X, omega)
    #beta = 1.0 + 0.5 * jnp.sum(jnp.square(Y - omega_mean))
    #tau_obs_proposal = dist.Gamma(alpha, beta).sample(rng_keys[5])

    return {'omega': omega, 'lamsq': lamsq, 'nu': nu, 'xi': xi, 'tausq': tausq} #, 'tau_obs': tau_obs_proposal}


def run_inference(args, rng_key, X, Y):
    model = flow_model if args.model == 'flow' else linear_model
    gibbs_fn = _gibbs_fn if args.model == 'flow' else _linear_gibbs_fn
    if args.strategy == "gibbs":
        #init_strategy = init_to_median(num_samples=10)
        truncation = args.truncation if args.truncation > 0 else None
        gibbs_fn = partial(gibbs_fn, X, Y, truncation)
        hmc_kernel = NUTS(model, max_tree_depth=6, target_accept_prob=0.6, dense_mass=True)#, init_strategy=init_strategy)
        kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['omega', 'lamsq', 'nu', 'tausq', 'xi'])
        mcmc = MCMC(kernel, args.num_warmup, args.num_samples, progress_bar=True)
    else:
        hmc_kernel = NUTS(model, max_tree_depth=6, target_accept_prob=0.6)
        mcmc = MCMC(hmc_kernel, args.num_warmup, args.num_samples, progress_bar=True)

    start = time.time()
    mcmc.run(rng_key, X, Y, args.strategy != "gibbs")
    exclude_sites = ["lamsq", "nu", "omega"]
    exclude_sites = None
    mcmc.print_summary(exclude_deterministic=True, exclude_sites=exclude_sites)
    print('\nMCMC elapsed time:', time.time() - start)

    return mcmc.get_samples()


# create artificial regression dataset
def get_data(N=50, P=30, sigma_obs=0.03):
    np.random.seed(0)

    X = np.random.randn(N * P).reshape((N, P))
    Y = 1.0 * X[:, 0] - 0.5 * X[:, 1]
    # Y = np.power(2.4 * X[:, 0] + 1.2 * X[:, 1], 3.0)
    Y += sigma_obs * np.random.randn(N)
    alpha = 0.22
    beta = 0.11
    Y = forward(alpha, beta, Y)
    #Y -= jnp.mean(Y)
    #Y /= jnp.std(Y)

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y
    #return X[:N], Y[:N], X[N:], Y[N:]

def get_cancer_data(gene='SOX10 (6663)'):
#def get_cancer_data(gene='SOX9 (6662)'):
#def get_cancer_data(gene='BRAF (673)'):
    gene_effect = tc.get(name="avana-public-tentative-20q4-0828", file="gene_effect")
    expression = tc.get(name="depmap-a0ab", file="CCLE_expression")

    shared_lines = sorted(set(gene_effect.index) & set(expression.index))
    gene_effect = gene_effect.loc[shared_lines]
    expression = expression.loc[shared_lines]

    complete_lines = (np.sum(np.isnan(gene_effect.values), axis=1) == 0)
    gene_effect = gene_effect.loc[complete_lines]
    expression = expression.loc[complete_lines]

    Y = gene_effect[gene].values
    Y = (Y - Y.mean()) / Y.std()
    X = expression.values
    X = (X - X.mean(0)) / (1.0 + X.max(0) - X.min(0))

    print("X, Y", X.shape, Y.shape)

    #idx = np.random.permutation(X.shape[0])
    #idx_train = idx[:500]
    #idx_test = idx[500:]

    #X_train, Y_train = X[idx_train], Y[idx_train]
    #X_test, Y_test = X[idx_test], Y[idx_test]

    #return X_train, Y_train, X_test, Y_test, expression
    return X, Y, expression


def predict(X, rng_key, omega, alpha, beta, tau_obs):
    sigma_obs = 1.0 / jnp.sqrt(tau_obs)
    Y_tilde = jnp.matmul(X, omega) + dist.Normal(jnp.zeros(X.shape[0]), sigma_obs).sample(rng_key)
    Y = forward(alpha, beta, Y_tilde)
    return Y


def main(args):
    print(args)

    N = args.num_data
    X, Y = get_data(N=2 * N, P=args.P)
    #X, Y, expression = get_cancer_data()
    #X = X[:, :5000]
    #Y = Y[:500]
    #X = X[:500]

    #splits = KFold(args.num_folds, random_state=0, shuffle=True).split(X)
    splits = [(np.arange(N), np.arange(N, 2 * N))]
    out = np.zeros(X.shape[0])

    for split, (train, test) in enumerate(splits):
        if split != args.split:
            continue

        print("Starting split {}...".format(split))
        X_train, Y_train = X[train], Y[train]
        X_test, Y_test = X[test], Y[test]

        rng_key, rng_key_predict = random.split(random.PRNGKey(split))
        samples = run_inference(args, rng_key, X_train, Y_train)

        rng_key_predict = random.split(rng_key_predict, args.num_samples)

        alpha = samples['alpha'] if args.model == 'flow' else jnp.zeros(args.num_samples)
        beta = samples['beta'] if args.model == 'flow' else jnp.zeros(args.num_samples)
        predictions = vmap(partial(predict, X_test))(rng_key_predict, samples['omega'], alpha,
                                                     beta, samples['tau_obs'])
        test_mean = jnp.mean(predictions, axis=0)
        test_rmse = jnp.sqrt(jnp.mean(jnp.square(test_mean - Y_test)))
        print("[split {}] test_rmse".format(split), test_rmse)

        out[test] = test_mean

    #np.save('out.{}.npy'.format(args.split), out)

    #p = pearsonr(out, Y)[0]
    #print("PEARSON", p)

    import sys;sys.exit()


    lamsq = samples['lamsq']
    tausq = samples['tausq']
    lamtau = np.sqrt(np.median(lamsq * tausq[:, None], axis=0))
    idx = np.argsort(lamtau)[-20:]
    omega = np.mean(samples['omega'], axis=0)

    #import sys;sys.exit()

    for i in idx:
        print(expression.columns[i], omega[i], lamtau[i])

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
    parser.add_argument("-n", "--num-samples", default=4000, type=int)
    parser.add_argument("--num-warmup", default=1000, type=int)
    parser.add_argument("--truncation", default=0, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--num-data", default=256, type=int)
    parser.add_argument("--num-folds", default=1, type=int)
    parser.add_argument("--strategy", default="gibbs", type=str, choices=["nuts", "gibbs"])
    parser.add_argument("--P", default=128, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--model", default='flow', type=str, choices=['flow', 'linear'])
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    enable_x64()

    main(args)
