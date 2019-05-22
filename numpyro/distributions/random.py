from numbers import Number

import jax.numpy as np
from jax import custom_transforms, device_get, jit, lax, random, vmap
from jax.config import config
from jax.interpreters import ad
from jax.util import partial

from numpyro.distributions.util import cumsum, promote_shapes


_DEFAULT_DTYPE = None  # depending on when the config is updated we can probably just assign the 
                       # dtype here or check from jax config.

def _get_default_dtype():
    global _DEFAULT_DTYPE
    if not _DEFAULT_DTYPE: 
        _DEFAULT_DTYPE = np.float64 if config.values['jax_enable_x64'] else np.float32
    return _DEFAULT_DTYPE


randint = random.randint
split = random.split
PRNGKey = random.PRNGKey


def bernoulli(key, p=0.5, shape=()):
    dtype = _get_default_dtype()
    return random.bernoulli(key, lax.convert_element_type(p, dtype), shape)


def cauchy(key, shape=()):
    dtype = _get_default_dtype()
    return random.cauchy(key, shape, dtype)


def exponential(key, shape=()):
    dtype = _get_default_dtype()
    return random.exponential(key, shape, dtype)


def normal(key, shape=()):
    dtype = _get_default_dtype()
    return random.normal(key, shape, dtype)


def pareto(key, b, shape=()):
    dtype = _get_default_dtype()
    return random.pareto(key, b, shape, dtype)


def uniform(key, shape=(), minval=0., maxval=1.):
    dtype = _get_default_dtype()
    return random.uniform(key, shape, dtype, minval, maxval)


def _gamma_one(key, alpha):
    # Marsaglia & Tsang's simple transformation-rejection method
    # Ref: https://dl.acm.org/citation.cfm?doid=358407.358414
    # https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables

    boost = np.where(alpha >= 1.0, 1.0, uniform(key, ()) ** (1.0 / alpha))
    key, = split(key, 1)  # NOTE: always split the key after calling random.foo
    alpha = np.where(alpha >= 1.0, alpha, alpha + 1.0)

    d = alpha - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)

    def _cond_fn(kXVU):
        _, X, V, U = kXVU
        # TODO: find a way to avoid evaluating second condition which involves log+log
        # note: lax.cond does not support batching rule yet
        return (U >= 1.0 - 0.0331 * X * X) & (np.log(U) >= 0.5 * X + d * (1.0 - V + np.log(V)))

    def _body_fn(kXVU):
        def _next_kxv(kxv):
            k = kxv[0]
            x = normal(k, ())
            k, = split(k, 1)
            v = 1.0 + c * x
            return k, x, v

        key = kXVU[0]
        key, x, v = lax.while_loop(lambda kxv: kxv[2] <= 0.0, _next_kxv, (key, 0.0, -1.0))
        X = x * x
        V = v * v * v
        U = uniform(key, ())
        key, = split(key, 1)
        return key, X, V, U

    _, _, V, _ = lax.while_loop(_cond_fn, _body_fn, (key, 1.0, 1.0, 2.0))
    z = d * V * boost
    return np.where(z == 0, np.finfo(z.dtype).tiny, z)


# TODO: use upstream implementation when available because it is 2x faster
def _gamma_impl(key, alpha):
    alphas = np.reshape(alpha, -1)
    keys = split(key, alphas.size)
    samples = vmap(_gamma_one)(keys, alphas)
    return samples.reshape(alpha.shape)


_bivariate_coef = [[0.16009398, -0.094634816, 0.025146379, -0.0030648348,
                    1, 0.3266811, 0.10406087, 0.0014179033],
                   [0.53487893, 0.12980707, 0.06573594, -0.0015649787,
                    0.16639465, 0.020070098, -0.0035938937, -0.00058392601],
                   [0.040121005, -0.0065914079, -0.002628604, -0.0013441777,
                    0.017050642, -0.0021309345, 0.00085092385, -1.5248239e-07]]


def _gamma_grad_one(z, alpha):
    # Ref 1: Pathwise Derivatives Beyond the Reparameterization Trick, Martin & Fritz
    # Ref 2: Case 4 follows https://github.com/fritzo/notebooks/blob/master/gamma-reparameterized.ipynb

    # TODO: use lax.cond instead of lax.while_loop when available
    def _case1(zagf):
        z, alpha, _, flag = zagf

        # dz = - dCDF(z; a) / pdf(z; a)
        # pdf = z^(a-1) * e^(-z) / Gamma(a)
        # CDF(z; a) = IncompleteGamma(a, z) / Gamma(a)
        # dCDF(z; a) = (dIncompleteGamma - IncompleteGamma * Digamma(a)) / Gamma(a)
        #            =: unnormalized_dCDF / Gamma(a)
        # IncompleteGamma ~ z^a [ 1/a - z/(a+1) + z^2/2!(a+2) - z^3/3!(a+3) + z^4/4!(a+4) - z^5/5!(a+5) ]
        #                 =: z^a * term1
        # dIncompleteGamma ~ z^a * log(z) * term1 - z^a [1/a^2 - z/(a+1)^2 + z^2/2!(a+2)^2
        #                                                - z^3/3!(a+3)^2 + z^4/4!(a+4)^2 - z^5/5!(a+5)^2 ]
        #                  =: z^a * log(z) * term1 - z^a * term2
        # unnormalized_dCDF = z^a { [log(z) - Digamma(a)] * term1 - term2 }
        zi = 1.0
        update = zi / alpha
        term1 = update
        term2 = update / alpha
        for i in range(1, 6):
            zi = -zi * z / i
            update = zi / (alpha + i)
            term1 = term1 + update
            term2 = term2 + update / (alpha + i)

        unnormalized_cdf_dot = np.power(z, alpha) * ((np.log(z) - lax.digamma(alpha)) * term1 - term2)
        unnormalized_pdf = np.power(z, alpha - 1) * np.exp(-z)
        grad = -unnormalized_cdf_dot / unnormalized_pdf

        return z, alpha, grad, ~flag

    def _cond2(zagf):
        z, alpha, _, flag = zagf
        return (~flag) & (alpha > 8.0) & ((z < 0.9 * alpha) | (z > 1.1 * alpha))

    def _case2(zagf):
        z, alpha, _, flag = zagf

        # Formula 58 of [1]
        sqrt_8a = np.sqrt(8 * alpha)
        z_minus_a = z - alpha
        log_z_div_a = np.log(z / alpha)
        sign = np.where(z < alpha, 1.0, -1.0)
        term1 = 4 * (z + alpha) / (sqrt_8a * z_minus_a * z_minus_a)
        term2 = log_z_div_a * (sqrt_8a / z_minus_a + sign * np.power(z_minus_a - alpha * log_z_div_a, -1.5))
        term3 = z * (1.0 + 1.0 / (12 * alpha) + 1.0 / (288 * alpha * alpha)) / sqrt_8a
        grad = (term1 + term2) * term3

        return z, alpha, grad, ~flag

    def _cond3(zagf):
        z, alpha, _, flag = zagf
        return (~flag) & (alpha > 8.0) & (z >= 0.9 * alpha) & (z <= 1.1 * alpha)

    def _case3(zagf):
        z, alpha, _, flag = zagf

        # Formula 59 of [1]
        z_div_a = np.divide(z, alpha)
        aa = alpha * alpha
        term1 = 1440 * alpha + 6 * z_div_a * (53 - 120 * z) - 65 * z_div_a * z_div_a + 3600 * z + 107
        term2 = 1244160 * alpha * aa
        term3 = 1 + 24 * alpha + 288 * aa
        grad = term1 * term3 / term2

        return z, alpha, grad, ~flag

    def _case4(zagf):
        z, alpha, _, flag = zagf

        # Ref [2]
        u = np.log(z / alpha)
        v = np.log(alpha)
        c = []
        for i in range(8):
            c.append(_bivariate_coef[0][i] + u * (_bivariate_coef[1][i] + u * _bivariate_coef[2][i]))
        p = c[0] + v * (c[1] + v * (c[2] + v * c[3]))
        q = c[4] + v * (c[5] + v * (c[6] + v * c[7]))
        grad = np.exp(p / np.maximum(q, 0.01))

        return z, alpha, grad, ~flag

    _, _, grad, flag = lax.while_loop(lambda zagf: (~zagf[3]) & (zagf[0] < 0.8), _case1, (z, alpha, 0.0, False))
    _, _, grad, flag = lax.while_loop(_cond2, _case2, (z, alpha, grad, flag))
    _, _, grad, flag = lax.while_loop(_cond3, _case3, (z, alpha, grad, flag))
    _, _, grad, flag = lax.while_loop(lambda zagf: ~zagf[3], _case4, (z, alpha, grad, flag))
    return grad


def _gamma_grad(sample, alpha):
    samples = np.reshape(sample, -1)
    alphas = np.reshape(alpha, -1)
    grads = vmap(_gamma_grad_one)(samples, alphas)
    return grads.reshape(alpha.shape)


@custom_transforms
def _gamma_p(key, alpha):
    return _gamma_impl(key, alpha)


ad.defjvp2(_gamma_p.primitive, None,
           lambda tangent, sample, key, alpha, **kwargs: tangent * _gamma_grad(sample, alpha))


@partial(jit, static_argnums=(2, 3))
def _gamma(key, alpha, shape, dtype):
    shape = shape or np.shape(alpha)
    alpha = lax.convert_element_type(alpha, dtype)
    if np.shape(alpha) != shape:
        alpha = np.broadcast_to(alpha, shape)
    return _gamma_p(key, alpha)


def gamma(key, alpha, shape=()):
    dtype = _get_default_dtype()
    return _gamma(key, alpha, shape, dtype)


# TODO: inefficient implementation; jit currently fails due to
# dynamic size of random.uniform.
@partial(jit, static_argnums=(2, 3))
def _binomial(key, p, n=1, shape=()):
    p, n = promote_shapes(p, n)
    shape = shape or lax.broadcast_shapes(np.shape(p), np.shape(n))
    n_max = int(np.max(n))
    uniforms = uniform(key, shape + (n_max,))
    n = np.expand_dims(n, axis=-1)
    p = np.expand_dims(p, axis=-1)
    mask = (np.arange(n_max) < n).astype(uniforms.dtype)
    p, uniforms = promote_shapes(p, uniforms)
    return np.sum(mask * lax.lt(uniforms, p), axis=-1, keepdims=False)


def binomial(key, p, n=1, shape=()):
    n = device_get(n)
    return _binomial(key, p, n, shape)


@partial(jit, static_argnums=(2,))
def _categorical(key, p, shape=()):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = cumsum(p)
    r = uniform(key, shape=shape + (1,))
    # FIXME: replace this computation by using binary search as suggested in the above
    # reference. A while_loop + vmap for a reshaped 2D array would be enough.
    return np.sum(s < r, axis=-1)


def categorical(key, p, shape=()):
    return _categorical(key, p, shape)


@partial(jit, static_argnums=(2,))
def _poisson(key, rate, shape=()):
    # Ref: https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    shape = shape or np.shape(rate)
    L = np.exp(-rate)
    k = np.zeros(shape)
    p = np.ones(shape)

    def body_fn(val):
        k, p, rng = val
        k = np.where(p > L, k + 1, k)
        rng, rng_u = split(rng)
        u = uniform(rng_u, shape)
        p = p * u
        return k, p, rng

    k, _, _ = lax.while_loop(lambda val: np.any(val[1] > L), body_fn, (k, p, key))
    return k - 1


def poisson(key, rate, shape):
    return _poisson(key, rate, shape)


def _scatter_add_one(operand, indices, updates):
    return lax.scatter_add(operand, indices, updates,
                           lax.ScatterDimensionNumbers(update_window_dims=(),
                                                       inserted_window_dims=(0,),
                                                       scatter_dims_to_operand_dims=(0,)))


@partial(jit, static_argnums=(2, 3))
def _multinomial(key, p, n, shape=()):
    if np.shape(n) != np.shape(p)[:-1]:
        broadcast_shape = lax.broadcast_shapes(np.shape(n), np.shape(p)[:-1])
        n = np.broadcast_to(n, broadcast_shape)
        p = np.broadcast_to(p, broadcast_shape + np.shape(p)[-1:])
    shape = shape or p.shape[:-1]
    n_max = int(np.max(n))
    # get indices from categorical distribution then gather the result
    indices = categorical(key, p, (n_max,) + shape)
    # mask out values when counts is heterogeneous
    if not isinstance(n, Number):
        mask = promote_shapes(np.arange(n_max) < np.expand_dims(n, -1), shape=shape + (n_max,))[0]
        mask = np.moveaxis(mask, -1, 0).astype(indices.dtype)
        excess = np.concatenate([np.expand_dims(n_max - n, -1), np.zeros(np.shape(n) + (p.shape[-1] - 1,))], -1)
    else:
        mask = 1
        excess = 0
    # NB: we transpose to move batch shape to the front
    indices_2D = (np.reshape(indices * mask, (n_max, -1,))).T
    samples_2D = vmap(_scatter_add_one, (0, 0, 0))(np.zeros((indices_2D.shape[0], p.shape[-1]),
                                                            dtype=indices.dtype),
                                                   np.expand_dims(indices_2D, axis=-1),
                                                   np.ones(indices_2D.shape, dtype=indices.dtype))
    return np.reshape(samples_2D, shape + p.shape[-1:]) - excess


def multinomial(key, p, n, shape=()):
    n = device_get(n)
    return _multinomial(key, p, n, shape)
