from functools import update_wrapper
import math

import scipy.special as osp_special

from jax import custom_transforms, defjvp, jit, lax, random, vmap
from jax.lib.xla_bridge import canonicalize_dtype
import jax.numpy as np
from jax.numpy.lax_numpy import _promote_args_like
from jax.scipy.linalg import solve_triangular
from jax.util import partial


@partial(jit, static_argnums=(3, 4))
def _binomial(key, p, n, n_max, shape):
    p, n = promote_shapes(p, n)
    shape = shape or lax.broadcast_shapes(np.shape(p), np.shape(n))
    uniforms = random.uniform(key, shape + (n_max,))
    n = np.expand_dims(n, axis=-1)
    p = np.expand_dims(p, axis=-1)
    mask = (np.arange(n_max) < n).astype(uniforms.dtype)
    p, uniforms = promote_shapes(p, uniforms)
    return np.sum(mask * lax.lt(uniforms, p), axis=-1, keepdims=False)


def binomial(key, p, n=1, shape=()):
    n_max = int(np.max(n))
    return _binomial(key, p, n, n_max, shape)


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


@partial(jit, static_argnums=(2, 3))
def _poisson(key, rate, shape, dtype):
    # Ref: https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    shape = shape or np.shape(rate)
    L = np.exp(-rate)
    k = np.zeros(shape, dtype=dtype)
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


def poisson(key, rate, shape, dtype=np.int64):
    dtype = canonicalize_dtype(dtype)
    return _poisson(key, rate, shape, dtype)


def _scatter_add_one(operand, indices, updates):
    return lax.scatter_add(operand, indices, updates,
                           lax.ScatterDimensionNumbers(update_window_dims=(),
                                                       inserted_window_dims=(0,),
                                                       scatter_dims_to_operand_dims=(0,)))


@partial(jit, static_argnums=(3, 4))
def _multinomial(key, p, n, n_max, shape=()):
    if np.shape(n) != np.shape(p)[:-1]:
        broadcast_shape = lax.broadcast_shapes(np.shape(n), np.shape(p)[:-1])
        n = np.broadcast_to(n, broadcast_shape)
        p = np.broadcast_to(p, broadcast_shape + np.shape(p)[-1:])
    shape = shape or p.shape[:-1]
    # get indices from categorical distribution then gather the result
    indices = categorical(key, p, (n_max,) + shape)
    # mask out values when counts is heterogeneous
    if np.ndim(n) > 0:
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
    n_max = int(np.max(n))
    return _multinomial(key, p, n, n_max, shape)


def _xlogy_jvp_lhs(g, ans, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(y))
    g = np.broadcast_to(g, shape)
    y = np.broadcast_to(y, shape)
    g, y = _promote_args_like(osp_special.xlogy, g, y)
    return lax._safe_mul(g, np.log(y))


def _xlogy_jvp_rhs(g, ans, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(x))
    g = np.broadcast_to(g, shape)
    x = np.broadcast_to(x, shape)
    x, y = _promote_args_like(osp_special.xlogy, x, y)
    return g * lax._safe_mul(x, np.reciprocal(y))


@custom_transforms
def xlogy(x, y):
    x, y = _promote_args_like(osp_special.xlogy, x, y)
    return lax._safe_mul(x, np.log(y))


defjvp(xlogy, _xlogy_jvp_lhs, _xlogy_jvp_rhs)


def _xlog1py_jvp_lhs(g, ans, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(y))
    g = np.broadcast_to(g, shape)
    y = np.broadcast_to(y, shape)
    g, y = _promote_args_like(osp_special.xlog1py, g, y)
    return lax._safe_mul(g, np.log1p(y))


def _xlog1py_jvp_rhs(g, ans, x, y):
    shape = lax.broadcast_shapes(np.shape(g), np.shape(x))
    g = np.broadcast_to(g, shape)
    x = np.broadcast_to(x, shape)
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    return g * lax._safe_mul(x, np.reciprocal(1 + y))


@custom_transforms
def xlog1py(x, y):
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    return lax._safe_mul(x, np.log1p(y))


defjvp(xlog1py, _xlog1py_jvp_lhs, _xlog1py_jvp_rhs)


def cholesky_inverse(matrix):
    # This formulation only takes the inverse of a triangular matrix
    # which is more numerically stable.
    # Refer to:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = np.swapaxes(np.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1)
    identity = np.broadcast_to(np.identity(matrix.shape[-1]), tril_inv.shape)
    return solve_triangular(tril_inv, identity, lower=True)


# TODO: move upstream to jax.nn
def binary_cross_entropy_with_logits(x, y):
    # compute -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))
    # Ref: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return np.clip(x, 0) + np.log1p(np.exp(-np.abs(x))) - x * y


@custom_transforms
def cumsum(x):
    return np.cumsum(x, axis=-1)


defjvp(cumsum, lambda g, ans, x: np.cumsum(g, axis=-1))


@custom_transforms
def cumprod(x):
    return np.cumprod(x, axis=-1)


# XXX this implementation does not address the case x=0, hence the result in that case will be nan
# Ref: https://stackoverflow.com/questions/40916955/how-to-compute-gradient-of-cumprod-safely
defjvp(cumprod, lambda g, ans, x: np.cumsum(g / x, axis=-1) * ans)


def promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [np.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [lax.reshape(arg, (1,) * (num_dims - len(s)) + s)
                if len(s) < num_dims else arg for arg, s in zip(args, shapes)]


def get_dtype(x):
    return canonicalize_dtype(lax.dtype(x))


def sum_rightmost(x, dim):
    """
    Sum out ``dim`` many rightmost dimensions of a given tensor.
    """
    out_dim = np.ndim(x) - dim
    x = np.reshape(x[..., np.newaxis], np.shape(x)[:out_dim] + (-1,))
    return np.sum(x, axis=-1)


def matrix_to_tril_vec(x, diagonal=0):
    idxs = np.tril_indices(x.shape[-1], diagonal)
    return x[..., idxs[0], idxs[1]]


def vec_to_tril_matrix(t, diagonal=0):
    # NB: the following formula only works for diagonal <= 0
    n = round((math.sqrt(1 + 8 * t.shape[-1]) - 1) / 2) - diagonal
    n2 = n * n
    idx = np.reshape(np.arange(n2), (n, n))[np.tril_indices(n, diagonal)]
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


def clamp_probs(probs):
    finfo = np.finfo(get_dtype(probs))
    return np.clip(probs, a_min=finfo.tiny, a_max=1. - finfo.eps)


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


def validate_sample(log_prob_fn):
    def wrapper(self, *args, **kwargs):
        log_prob = log_prob_fn(self, *args, *kwargs)
        if self._validate_args:
            value = kwargs['value'] if 'value' in kwargs else args[0]
            mask = self._validate_sample(value)
            log_prob = np.where(mask, log_prob, -np.inf)
        return log_prob

    return wrapper
