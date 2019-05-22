import math
from functools import update_wrapper

import numpy as onp
import scipy.special as osp_special

import jax.numpy as np
from jax import canonicalize_dtype, custom_transforms, lax
from jax.interpreters import ad
from jax.numpy.lax_numpy import _promote_args_like
from jax.scipy.special import gammaln


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


ad.defjvp(xlogy.primitive, _xlogy_jvp_lhs, _xlogy_jvp_rhs)


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


@custom_transforms
def xlog1py(x, y):
    x, y = _promote_args_like(osp_special.xlog1py, x, y)
    return lax._safe_mul(x, np.log1p(y))


ad.defjvp(xlog1py.primitive, _xlog1py_jvp_lhs, _xlog1py_jvp_rhs)


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
