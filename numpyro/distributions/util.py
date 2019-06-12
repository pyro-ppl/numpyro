import math
from functools import update_wrapper
from numbers import Number

import numpy as onp
import scipy.special as osp_special

import jax.numpy as np
from jax import canonicalize_dtype, custom_transforms, device_get, jit, lax, random, vmap
from jax.interpreters import ad, batching
from jax.lib import xla_bridge
from jax.numpy.lax_numpy import _promote_args_like
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
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


# TODO: use upstream implementation when available because it is 2x faster
def _standard_gamma_impl(key, alpha):
    if key.ndim > 1:
        keys = vmap(lambda k: random.split(k, np.size(alpha[0])))(key)
    else:
        keys = random.split(key, alpha.size)
    alphas = np.reshape(alpha, -1)
    keys = np.reshape(keys, (-1, 2))
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


def _standard_gamma_batching_rule(batched_args, batch_dims):
    x, y = batched_args
    bx, by = batch_dims
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    x = batching.bdim_at_front(x, bx, size, force_broadcast=True)
    y = batching.bdim_at_front(y, by, size, force_broadcast=True)
    return _standard_gamma_p(x, y), 0


@custom_transforms
def _standard_gamma_p(key, alpha):
    return _standard_gamma_impl(key, alpha)


ad.defjvp2(_standard_gamma_p.primitive, None,
           lambda tangent, sample, key, alpha, **kwargs: tangent * _standard_gamma_grad(sample, alpha))
batching.primitive_batchers[_standard_gamma_p.primitive] = _standard_gamma_batching_rule


@partial(jit, static_argnums=(2, 3))
def _standard_gamma(key, alpha, shape, dtype):
    shape = shape or np.shape(alpha)
    alpha = lax.convert_element_type(alpha, dtype)
    if np.shape(alpha) != shape:
        alpha = np.broadcast_to(alpha, shape)
    return _standard_gamma_p(key, alpha)


def standard_gamma(key, alpha, shape=(), dtype=np.float64):
    dtype = xla_bridge.canonicalize_dtype(dtype)
    return _standard_gamma(key, alpha, shape, dtype)


# TODO: inefficient implementation; jit currently fails due to
# dynamic size of random.uniform.
@partial(jit, static_argnums=(2, 3))
def _binomial(key, p, n, shape):
    p, n = promote_shapes(p, n)
    shape = shape or lax.broadcast_shapes(np.shape(p), np.shape(n))
    n_max = int(np.max(n))
    uniforms = random.uniform(key, shape + (n_max,))
    n = np.expand_dims(n, axis=-1)
    p = np.expand_dims(p, axis=-1)
    mask = (np.arange(n_max) < n).astype(uniforms.dtype)
    p, uniforms = promote_shapes(p, uniforms)
    return np.sum(mask * lax.lt(uniforms, p), axis=-1, keepdims=False)


def binomial(key, p, n=1, shape=()):
    n = device_get(n)
    return _binomial(key, p, n, shape)


@partial(jit, static_argnums=(2,))
def _categorical(key, p, shape):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = cumsum(p)
    r = random.uniform(key, shape=shape + (1,))
    # FIXME: replace this computation by using binary search as suggested in the above
    # reference. A while_loop + vmap for a reshaped 2D array would be enough.
    return np.sum(s < r, axis=-1)


def categorical(key, p, shape=()):
    return _categorical(key, p, shape)


@partial(jit, static_argnums=(2,))
def _poisson(key, rate, shape):
    # Ref: https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    shape = shape or np.shape(rate)
    L = np.exp(-rate)
    k = np.zeros(shape)
    p = np.ones(shape)

    def body_fn(val):
        k, p, rng = val
        k = np.where(p > L, k + 1, k)
        rng, rng_u = random.split(rng)
        u = random.uniform(rng_u, shape)
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


def _xlogy_jvp_lhs(g, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(y))
    g = np.broadcast_to(g, shape)
    y = np.broadcast_to(y, shape)
    g, y = _promote_args_like(osp_special.xlogy, g, y)
    return lax._safe_mul(g, np.log(y))


def _xlogy_jvp_rhs(g, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(x))
    g = np.broadcast_to(g, shape)
    x = np.broadcast_to(x, shape)
    x, y = _promote_args_like(osp_special.xlogy, x, y)
    return g * lax._safe_mul(x, np.reciprocal(y))


@custom_transforms
def xlogy(x, y):
    x, y = _promote_args_like(osp_special.xlogy, x, y)
    return lax._safe_mul(x, np.log(y))


def _xlogy_batching_rule(batched_args, batch_dims):
    x, y = batched_args
    bx, by = batch_dims
    # promote shapes
    sx, sy = np.shape(x), np.shape(y)
    nx = len(sx) + int(bx is None)
    ny = len(sy) + int(by is None)
    nd = max(nx, ny)
    x = np.reshape(x, (1,) * (nd - len(sx)) + sx)
    y = np.reshape(y, (1,) * (nd - len(sy)) + sy)
    # correct bx, by due to promoting
    bx = bx + nd - len(sx) if bx is not None else nd - len(sx) - 1
    by = by + nd - len(sy) if by is not None else nd - len(sy) - 1
    # move bx, by to front
    x = batching.move_dim_to_front(x, bx)
    y = batching.move_dim_to_front(y, by)
    return xlogy(x, y), 0


ad.defjvp(xlogy.primitive, _xlogy_jvp_lhs, _xlogy_jvp_rhs)
batching.primitive_batchers[xlogy.primitive] = _xlogy_batching_rule


def _xlog1py_jvp_lhs(g, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(y))
    g = np.broadcast_to(g, shape)
    y = np.broadcast_to(y, shape)
    g, y = _promote_args_like(osp_special.xlog1py, g, y)
    return lax._safe_mul(g, np.log1p(y))


def _xlog1py_jvp_rhs(g, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(x))
    g = np.broadcast_to(g, shape)
    x = np.broadcast_to(x, shape)
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    return g * lax._safe_mul(x, np.reciprocal(1 + y))


def _xlog1py_batching_rule(batched_args, batch_dims):
    x, y = batched_args
    bx, by = batch_dims
    # promote shapes
    sx, sy = np.shape(x), np.shape(y)
    nx = len(sx) + int(bx is None)
    ny = len(sy) + int(by is None)
    nd = max(nx, ny)
    x = np.reshape(x, (1,) * (nd - len(sx)) + sx)
    y = np.reshape(y, (1,) * (nd - len(sy)) + sy)
    # correct bx, by due to promoting
    bx = bx + nd - len(sx) if bx is not None else nd - len(sx) - 1
    by = by + nd - len(sy) if by is not None else nd - len(sy) - 1
    # move bx, by to front
    x = batching.move_dim_to_front(x, bx)
    y = batching.move_dim_to_front(y, by)
    return xlog1py(x, y), 0


@custom_transforms
def xlog1py(x, y):
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    return lax._safe_mul(x, np.log1p(y))


ad.defjvp(xlog1py.primitive, _xlog1py_jvp_lhs, _xlog1py_jvp_rhs)
batching.primitive_batchers[xlog1py.primitive] = _xlog1py_batching_rule


def cholesky_inverse(matrix):
    # This formulation only takes the inverse of a triangular matrix
    # which is more numerically stable.
    # Refer to:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = np.swapaxes(np.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1)
    identity = np.broadcast_to(np.identity(matrix.shape[-1]), tril_inv.shape)
    return solve_triangular(tril_inv, identity, lower=True)


def entr(p):
    return np.where(p < 0, -np.inf, -xlogy(p))


def multigammaln(a, d):
    constant = 0.25 * d * (d - 1) * np.log(np.pi)
    res = np.sum(gammaln(np.expand_dims(a, axis=-1) - 0.5 * np.arange(d)), axis=-1)
    return res + constant


def binary_cross_entropy_with_logits(x, y):
    # compute -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))
    # Ref: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return np.clip(x, 0) + np.log1p(np.exp(-np.abs(x))) - x * y


def softmax(x, axis=-1):
    unnormalized = np.exp(x - np.max(x, axis, keepdims=True))
    return unnormalized / np.sum(unnormalized, axis, keepdims=True)


@custom_transforms
def cumsum(x):
    return np.cumsum(x, axis=-1)


ad.defjvp(cumsum.primitive, lambda g, x: np.cumsum(g, axis=-1))
batching.defvectorized(cumsum.primitive)


@custom_transforms
def cumprod(x):
    return np.cumprod(x, axis=-1)


# XXX this implementation does not address the case x=0, hence the result in that case will be nan
# Ref: https://stackoverflow.com/questions/40916955/how-to-compute-gradient-of-cumprod-safely
ad.defjvp2(cumprod.primitive, lambda g, ans, x: np.cumsum(g / x, axis=-1) * ans)
batching.defvectorized(cumprod.primitive)


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
    return [canonicalize_dtype(onp.result_type(arg)) for arg in args]


def sum_rightmost(x, dim):
    return np.sum(x, axis=-1) if dim == 1 else x


def matrix_to_tril_vec(x, diagonal=0):
    idxs = onp.tril_indices(x.shape[-1], diagonal)
    return x[..., idxs[0], idxs[1]]


def vec_to_tril_matrix(t, diagonal=0):
    # NB: the following formula only works for diagonal <= 0
    n = round((math.sqrt(1 + 8 * t.shape[-1]) - 1) / 2) - diagonal
    n2 = n * n
    idx = np.reshape(np.arange(n2), (n, n))[onp.tril_indices(n, diagonal)]
    x = lax.scatter_add(np.zeros(t.shape[:-1] + (n2,)), np.expand_dims(idx, axis=-1), t,
                        lax.ScatterDimensionNumbers(update_window_dims=range(t.ndim - 1),
                                                    inserted_window_dims=(t.ndim - 1,),
                                                    scatter_dims_to_operand_dims=(t.ndim - 1,)))
    return np.reshape(x, x.shape[:-1] + (n, n))


def signed_stick_breaking_tril(t):
    # make sure that t in (-1, 1)
    eps = np.finfo(t.dtype).eps
    t = np.clip(t, a_min=(-1 + eps), a_max=(1 - eps))
    # transform t to tril matrix with identity diagonal
    r = vec_to_tril_matrix(t, diagonal=-1)

    # apply stick-breaking on the squared values;
    # we omit the step of computing s = z * z_cumprod by using the fact:
    #     y = sign(r) * s = sign(r) * sqrt(z * z_cumprod) = r * sqrt(z_cumprod)
    z = r ** 2
    z1m_cumprod = cumprod(1 - z)
    z1m_cumprod_sqrt = np.sqrt(z1m_cumprod)

    pad_width = [(0, 0)] * z.ndim
    pad_width[-1] = (1, 0)
    z1m_cumprod_sqrt_shifted = np.pad(z1m_cumprod_sqrt[..., :-1], pad_width,
                                      mode="constant", constant_values=1.)
    y = (r + np.identity(r.shape[-1])) * z1m_cumprod_sqrt_shifted
    return y


# The is sourced from: torch.distributions.util.py
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


def clamp_probs(probs):
    finfo = np.finfo(get_dtypes(probs)[0])
    return np.clip(probs, a_min=finfo.tiny, a_max=1. - finfo.eps)
