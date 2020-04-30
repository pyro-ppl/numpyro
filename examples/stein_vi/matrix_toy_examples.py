# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import jax
import jax.numpy as np
from jax import lax
from jax import scipy
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.contrib.autoguide import AutoDelta
from numpyro.infer.util import init_to_value, init_with_noise
import numpyro.infer.kernels as kernels
from numpyro.infer.stein import SVGD
from numpyro.infer import ELBO
import numpy as onp


# %%
rng_key = jax.random.PRNGKey(42)

# %% [markdown]
# # Sine Model

"""
# %%
class Sine(dist.Distribution):
    support = constraints.real
    def __init__(self, alpha=np.array(1.), sigma1=np.array(3e-3), sigma2=np.array(1.0), validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(alpha), np.shape(sigma1), np.shape(sigma2))
        self.alpha = np.broadcast_to(alpha, batch_shape)
        self.sigma1 = np.broadcast_to(sigma1, batch_shape)
        self.sigma2 = np.broadcast_to(sigma2, batch_shape)
        super(Sine, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        return - (value[..., 1] + np.sin(self.alpha * value[..., 0])) ** 2 / (2 * self.sigma1) -                (value[..., 0] ** 2 + value[..., 1] ** 2) / (2 * self.sigma2)

    def sample(self, key, sample_shape=()):
        xs = np.array(onp.linspace(-1, 1, num=100))
        ys = np.array(onp.linspace(-1, 1, num=100))
        zs = np.stack(np.meshgrid(xs, ys), axis=-1)
        logits = np.expand_dims(np.ravel(self.log_prob(zs)), axis=0)
        cs = dist.Categorical(logits=logits).sample(key, sample_shape)
        res = np.concatenate(np.divmod(cs, zs.shape[0]), axis=-1).astype('float') /              np.array([np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)]) + np.array([np.min(xs), np.min(ys)])
        return res


# %%
sine_xs = np.linspace(-1, 1, num=100)
sine_ys = np.linspace(-1, 1, num=100)
sine_zs = np.stack(np.meshgrid(sine_xs, sine_ys), axis=-1)
sine_lps = np.exp(Sine().log_prob(sine_zs))
plt.imshow(sine_lps, origin='lower', interpolation='bicubic', extent=[np.min(sine_xs), np.max(sine_xs), 
                                                                      np.min(sine_ys), np.max(sine_ys)])
plt.show()


# %%
num_iterations = 30


# %%
def model():
    numpyro.sample('x', Sine())
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value({'x': np.array([0., 0.])}), noise_scale=1.0))


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=0.5), ELBO(),
            kernels.RBFKernel(), num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(sine_lps, origin='lower', interpolation='bicubic', extent=[np.min(sine_xs), np.max(sine_xs), 
                                                                      np.min(sine_ys), np.max(sine_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(sine_xs), np.max(sine_xs)))
plt.ylim((np.min(sine_ys), np.max(sine_ys)))
plt.show()


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(), kernels.RBFKernel(mode='matrix'),
                                        precond_mode='const'),
            num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(sine_lps, origin='lower', interpolation='bicubic', extent=[np.min(sine_xs), np.max(sine_xs), 
                                                                      np.min(sine_ys), np.max(sine_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(sine_xs), np.max(sine_xs)))
plt.ylim((np.min(sine_ys), np.max(sine_ys)))
plt.show()


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=0.5), ELBO(),
            kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(), kernels.RBFKernel(mode='matrix'),
                                        precond_mode='anchor_points'),
            num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(sine_lps, origin='lower', interpolation='bicubic', extent=[np.min(sine_xs), np.max(sine_xs), 
                                                                      np.min(sine_ys), np.max(sine_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(sine_xs), np.max(sine_xs)))
plt.ylim((np.min(sine_ys), np.max(sine_ys)))
plt.show()
"""

# %% [markdown]
# # Double Banana Model

# %%
class DoubleBanana(dist.Distribution):
    support = constraints.real
    def __init__(self, y=np.log(30.), sigma1=np.array(1.0), sigma2=np.array(9e-2), validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(y), np.shape(sigma1), np.shape(sigma2))
        self.y = np.broadcast_to(y, batch_shape)
        self.sigma1 = np.broadcast_to(sigma1, batch_shape)
        self.sigma2 = np.broadcast_to(sigma2, batch_shape)
        super(DoubleBanana, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
    
    def log_prob(self, value):
        fx = np.log((1 - value[..., 0]) ** 2.0 + 100 * (value[..., 1] - value[..., 0] ** 2.0) ** 2.0)
        return - np.sqrt(value[..., 0] ** 2.0 + value[..., 1] ** 2.0) ** 2.0 / (2.0 * self.sigma1) -                 (self.y - fx) ** 2.0 / (2.0 * self.sigma2)

    
    def sample(self, key, sample_shape=()):
        xs = np.array(onp.linspace(-1.5, 1.5, num=100))
        ys = np.array(onp.linspace(-1, 2, num=100))
        zs = np.stack(np.meshgrid(xs, ys), axis=-1)
        logits = np.expand_dims(np.ravel(self.log_prob(zs)), axis=0)
        cs = dist.Categorical(logits=logits).sample(key, sample_shape)
        res = np.concatenate(np.divmod(cs, zs.shape[0]), axis=-1).astype('float') /              np.array([np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)]) + np.array([np.min(xs), np.min(ys)])
        return res


# %%
db_xs = np.linspace(-1.5, 1.5, num=100)
db_ys = np.linspace(-1, 2, num=100)
db_zs = np.stack(np.meshgrid(db_xs, db_ys), axis=-1)
db_lps = np.exp(DoubleBanana().log_prob(db_zs))
plt.imshow(db_lps, origin='lower', interpolation='bicubic', extent=[np.min(db_xs), np.max(db_xs), 
                                                                    np.min(db_ys), np.max(db_ys)])
plt.show()


# %%
num_iterations = 100

# %%
def model():
    numpyro.sample('x', DoubleBanana())
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value({'x': np.array([0., 0.])}), noise_scale=3.0))

"""

# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.RBFKernel(), num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(db_lps, origin='lower', interpolation='bicubic', extent=[np.min(db_xs), np.max(db_xs), 
                                                                    np.min(db_ys), np.max(db_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(db_xs), np.max(db_xs)))
plt.ylim((np.min(db_ys), np.max(db_ys)))
plt.show()


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(), kernels.RBFKernel(mode='matrix'),
                                        precond_mode='const'),
            num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.imshow(db_lps, origin='lower', interpolation='bicubic', extent=[np.min(db_xs), np.max(db_xs), 
                                                                    np.min(db_ys), np.max(db_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(db_xs), np.max(db_xs)))
plt.ylim((np.min(db_ys), np.max(db_ys)))
plt.show()


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=.8), ELBO(),
            kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(), kernels.RBFKernel(mode='matrix'),
                                        precond_mode='anchor_points'),
            num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.imshow(db_lps, origin='lower', interpolation='bicubic', extent=[np.min(db_xs), np.max(db_xs), 
                                                                    np.min(db_ys), np.max(db_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(db_xs), np.max(db_xs)))
plt.ylim((np.min(db_ys), np.max(db_ys)))
plt.show()

# %% [markdown]
# # Star Model

# %%
class Star(dist.Distribution):
    support = constraints.real
    def __init__(self, mu0=np.array([0., 1.5]), cov0=np.diag(np.array([1e-2, 0.25])), n_comp=5, validate_args=None):
        batch_shape = lax.broadcast_shapes(np.shape(mu0)[:-1], np.shape(cov0)[:-2])
        mu0 = np.broadcast_to(mu0, batch_shape + np.shape(mu0)[-1:])
        cov0 = np.broadcast_to(cov0, batch_shape + np.shape(cov0)[-2:])
        self.n_comp = n_comp
        mus = [mu0]
        covs = [cov0]
        theta = 2 * np.pi / n_comp
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for i in range(n_comp - 1):
            mui = rot @ mus[-1]
            covi = rot @ covs[-1] @ rot.transpose()
            mus.append(mui)
            covs.append(covi)
        self.mus = np.stack(mus)
        self.covs = np.stack(covs)
        super(Star, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
    
    def log_prob(self, value):
        lps = []
        for i in range(self.n_comp):
            lps.append(dist.MultivariateNormal(self.mus[i], self.covs[i]).log_prob(value))
        return scipy.special.logsumexp(np.stack(lps, axis=0), axis=0) / self.n_comp

    def sample(self, key, sample_shape=()):
        zs = dist.Categorical(probs=np.array([1/self.n_comp] * self.n_comp)).sample(key, sample_shape)
        xs = np.stack([dist.MultivariateNormal(self.mus[i], self.covs[i]).sample(key, sample_shape) 
                       for i in range(self.n_comp)], axis=0)
        return np.take_along_axis(xs, np.expand_dims(np.expand_dims(zs, axis=-1), axis=-1), axis=0)


# %%
star_xs = np.linspace(-3, 3, num=100)
star_ys = np.linspace(-3, 3, num=100)
star_zs = np.stack(np.meshgrid(star_xs, star_ys), axis=-1)
star_lps = np.exp(Star().log_prob(star_zs))
plt.imshow(star_lps, origin='lower', interpolation='bicubic', extent=[np.min(star_xs), np.max(star_xs),
                                                                      np.min(star_ys), np.max(star_ys)])
plt.show()


# %%
num_iterations = 60


# %%
def model():
    numpyro.sample('x', Star())
guide = AutoDelta(model, init_strategy=init_with_noise(init_to_value({'x': np.array([0., 0.])}), noise_scale=3.0))


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=1.0), ELBO(),
            kernels.RBFKernel(), num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(star_lps, origin='lower', interpolation='bicubic', extent=[np.min(star_xs), np.max(star_xs), 
                                                                      np.min(star_ys), np.max(star_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(star_xs), np.max(star_xs)))
plt.ylim((np.min(star_ys), np.max(star_ys)))
plt.show()


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=0.5), ELBO(),
            kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(), kernels.RBFKernel(mode='matrix'),
                                        precond_mode='const'),
            num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(star_lps, origin='lower', interpolation='bicubic', extent=[np.min(star_xs), np.max(star_xs), 
                                                                      np.min(star_ys), np.max(star_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(star_xs), np.max(star_xs)))
plt.ylim((np.min(star_ys), np.max(star_ys)))
plt.show()


# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=0.8), ELBO(),
            kernels.PrecondMatrixKernel(kernels.HessianPrecondMatrix(), kernels.RBFKernel(mode='matrix'),
                                        precond_mode='anchor_points'),
            num_stein_particles=50, num_loss_particles=1)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.imshow(star_lps, origin='lower', interpolation='bicubic', extent=[np.min(star_xs), np.max(star_xs), 
                                                                      np.min(star_ys), np.max(star_ys)])
plt.scatter(res[..., 0], res[..., 1], c='orange', marker='x')
plt.xlim((np.min(star_xs), np.max(star_xs)))
plt.ylim((np.min(star_ys), np.max(star_ys)))
plt.show()

"""
# %%
svgd = SVGD(model, guide, numpyro.optim.Adagrad(step_size=0.8), ELBO(),
            kernels.RBFKernel(),
            num_stein_particles=50, num_loss_particles=1,
            num_mcmc_particles=5)


# %%
svgd_state, loss = svgd.run(rng_key, num_iterations)
res = svgd.get_params(svgd_state)['auto_x']
plt.clf()
plt.scatter(res[..., 0], res[..., 1])


# %%


