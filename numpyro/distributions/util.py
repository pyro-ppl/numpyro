from numbers import Number

import numpy as onp
import scipy.special as osp_special

import jax.numpy as np
from jax import canonicalize_dtype, custom_transforms, jit, lax, random, vmap
from jax.core import Primitive
from jax.interpreters import ad, partial_eval, xla
from jax.numpy.lax_numpy import _promote_args_like, _promote_shapes
from jax.util import partial


def _standard_gamma_one(key, alpha):
    # Marsaglia & Tsang's simple transformation-rejection method
    # Ref: https://dl.acm.org/citation.cfm?doid=358407.358414
    # https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables

    boost = np.where(alpha >= 1.0, 1.0, random.uniform(key, ()) ** (1.0 / alpha))
    key, = random.split(key, 1)  # NOTE: always split the key after calling random.foo
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
            x = random.normal(k, ())
            k, = random.split(k, 1)
            v = 1.0 + c * x
            return k, x, v

        key = kXVU[0]
        key, x, v = lax.while_loop(lambda kxv: kxv[2] <= 0.0, _next_kxv, (key, 0.0, -1.0))
        X = x * x
        V = v * v * v
        U = random.uniform(key, ())
        key, = random.split(key, 1)
        return key, X, V, U

    _, _, V, _ = lax.while_loop(_cond_fn, _body_fn, (key, 1.0, 1.0, 2.0))
    z = d * V * boost
    return np.where(z == 0, np.finfo(z.dtype).tiny, z)


def _standard_gamma_impl(key, alpha):
    alphas = np.reshape(alpha, -1)
    keys = random.split(key, alphas.size)
    samples = vmap(_standard_gamma_one)(keys, alphas)
    return samples.reshape(alpha.shape)


_bivariate_coef = [[0.16009398, -0.094634816, 0.025146379, -0.0030648348,
                    1, 0.3266811, 0.10406087, 0.0014179033],
                   [0.53487893, 0.12980707, 0.06573594, -0.0015649787,
                    0.16639465, 0.020070098, -0.0035938937, -0.00058392601],
                   [0.040121005, -0.0065914079, -0.002628604, -0.0013441777,
                    0.017050642, -0.0021309345, 0.00085092385, -1.5248239e-07]]


def _standard_gamma_grad_one(z, alpha):
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


def _standard_gamma_grad(sample, alpha):
    samples = np.reshape(sample, -1)
    alphas = np.reshape(alpha, -1)
    grads = vmap(_standard_gamma_grad_one)(samples, alphas)
    return grads.reshape(alpha.shape)


def _standard_gamma_abstract_eval(key, alpha, jaxpr, aval, consts):
    return lax.maybe_tracer_tuple_to_abstract_tuple(aval)


def _standard_gamma_translate(c, key, alpha, jaxpr, aval, consts):
    xla_computation = xla.jaxpr_computation(jaxpr, consts, (), c.GetShape(key), c.GetShape(alpha))
    return c.Call(xla_computation, (key, alpha))


# define primitive
standard_gamma_p = Primitive('standard_gamma')
standard_gamma_p.def_impl(partial(xla.apply_primitive, standard_gamma_p))
standard_gamma_p.def_abstract_eval(_standard_gamma_abstract_eval)
xla.translations[standard_gamma_p] = _standard_gamma_translate
ad.defjvp2(standard_gamma_p, None,
           lambda tangent, sample, key, alpha, **kwargs: tangent * _standard_gamma_grad(sample, alpha))


@partial(jit, static_argnums=(2, 3))
def standard_gamma(key, alpha, shape=(), dtype=np.float32):
    shape = shape or np.shape(alpha)
    alpha = lax.convert_element_type(alpha, dtype)
    if np.shape(alpha) != shape:
        alpha = np.broadcast_to(alpha, shape)
    jaxpr, out, consts = partial_eval.trace_unwrapped_to_jaxpr(_standard_gamma_impl,
                                                               tuple(lax._abstractify(o) for o in (key, alpha)))
    aval, _ = out
    return standard_gamma_p.bind(key, alpha, jaxpr=jaxpr, aval=aval, consts=consts)


# TODO @npradhan: move to jax repo, if possible.
def xlogy(x, y):
    jaxpr, out, consts = partial_eval.trace_unwrapped_to_jaxpr(xlogy_impl, tuple(lax._abstractify(o) for o in (x, y)))
    aval, _ = out
    return xlogy_p.bind(x, y, jaxpr=jaxpr, aval=aval, consts=consts)


def xlogy_impl(x, y):
    return x * np.where(x == 0., 0., np.log(y))


def xlogy_abstract_eval(x, y, jaxpr, aval, consts):
    return lax.maybe_tracer_tuple_to_abstract_tuple(aval)


def xlogy_translate(c, x, y, jaxpr, aval, consts):
    xla_computation = xla.jaxpr_computation(jaxpr, consts, (), c.GetShape(x), c.GetShape(y))
    return c.Call(xla_computation, (x, y))


def xlogy_jvp_lhs(g, x, y, jaxpr, aval, consts):
    x, y = _promote_args_like(osp_special.xlogy, x, y)
    g, y = _promote_args_like(osp_special.xlogy, g, y)
    return lax._safe_mul(lax._brcast(g, y), lax._brcast(lax.log(y), g))


def xlogy_jvp_rhs(g, x, y, jaxpr, aval, consts):
    x, y = _promote_args_like(osp_special.xlogy, x, y)
    g, x = _promote_args_like(osp_special.xlogy, g, x)
    jac = lax._safe_mul(lax._brcast(x, y), lax._brcast(lax.reciprocal(y), x))
    return lax.mul(lax._brcast(g, jac), jac)


xlogy_p = Primitive('xlogy')
xlogy_p.def_impl(partial(xla.apply_primitive, xlogy_p))
xlogy_p.def_abstract_eval(xlogy_abstract_eval)
xla.translations[xlogy_p] = xlogy_translate
ad.defjvp(xlogy_p, xlogy_jvp_lhs, xlogy_jvp_rhs)


def xlog1py(x, y):
    jaxpr, out, consts = partial_eval.trace_unwrapped_to_jaxpr(xlog1py_impl, tuple(lax._abstractify(o) for o in (x, y)))
    aval, _ = out
    return xlog1py_p.bind(x, y, jaxpr=jaxpr, aval=aval, consts=consts)


def xlog1py_impl(x, y):
    return x * np.where(x == 0., 0., np.log1p(y))


def xlog1py_jvp_lhs(g, x, y, jaxpr, aval, consts):
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    g, y = _promote_args_like(osp_special.xlog1py, g, y)
    return lax._safe_mul(lax._brcast(g, y), lax._brcast(lax.log1p(y), g))


def xlog1py_jvp_rhs(g, x, y, jaxpr, aval, consts):
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    g, x = _promote_args_like(osp_special.xlog1py, g, x)
    jac = lax._safe_mul(lax._brcast(x, y), lax._brcast(lax.reciprocal(1 + y), x))
    return lax.mul(lax._brcast(g, jac), jac)


xlog1py_p = Primitive('xlog1py')
xlog1py_p.def_impl(partial(xla.apply_primitive, xlog1py_p))
xlog1py_p.def_abstract_eval(xlogy_abstract_eval)
xla.translations[xlog1py_p] = xlogy_translate
ad.defjvp(xlog1py_p, xlog1py_jvp_lhs, xlog1py_jvp_rhs)


def entr(p):
    e = np.where(p > 0, -p * np.log(p), p)
    e = np.where(e == 0, 0, e)
    e = np.where(e < 0, -np.inf)
    return e


def binary_cross_entropy_with_logits(x, y):
    # compute -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))
    # Ref: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return np.clip(x, 0) + np.log1p(np.exp(-np.abs(x))) - x * y


@custom_transforms
def cumsum(x):
    return np.cumsum(x, axis=-1)


ad.defjvp(cumsum.primitive, lambda g, x: np.cumsum(g, axis=-1))


@custom_transforms
def cumprod(x):
    return np.cumprod(x, axis=-1)


# XXX this implementation does not address the case x=0, hence the result in that case will be nan
# Ref: https://stackoverflow.com/questions/40916955/how-to-compute-gradient-of-cumprod-safely
ad.defjvp2(cumprod.primitive, lambda g, ans, x: np.cumsum(g / x, axis=-1) * ans)


def promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [np.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [lax.reshape(arg, (1,) * (num_dims - len(s)) + s)
                if len(s) < num_dims else arg for arg, s in zip(args, shapes)]


def get_dtypes(*args):
    return [canonicalize_dtype(type(arg)) if isinstance(arg, Number)
            else canonicalize_dtype(onp.dtype(arg)) for arg in args]


# TODO: inefficient implementation; jit currently fails due to
# dynamic size of random.uniform.
# @jit
def binomial(key, p, n=1, shape=()):
    p, n = _promote_shapes(p, n)
    shape = shape or lax.broadcast_shapes(np.shape(p), np.shape(n))
    n_max = np.max(n)
    uniforms = random.uniform(key, shape + (n_max,))
    n = np.expand_dims(n, axis=-1)
    p = np.expand_dims(p, axis=-1)
    mask = (np.arange(n_max) > n).astype(uniforms.dtype)
    p, uniforms = promote_shapes(p, uniforms)
    return np.sum(mask * lax.lt(uniforms, p), axis=-1, keepdims=False)
