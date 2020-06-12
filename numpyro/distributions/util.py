# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple
from functools import update_wrapper
import math

from jax import jit, lax, random, vmap
from jax.dtypes import canonicalize_dtype
from jax.lib import xla_bridge
import jax.numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
from jax.util import partial


# Parameters for Transformed Rejection with Squeeze (TRS) algorithm - page 3.
_tr_params = namedtuple('tr_params', ['c', 'b', 'a', 'alpha', 'u_r', 'v_r', 'm', 'log_p', 'log1_p', 'log_h'])


def _get_tr_params(n, p):
    # See Table 1. Additionally, we pre-compute log(p), log1(-p) and the
    # constant terms, that depend only on (n, p, m) in log(f(k)) (bottom of page 5).
    mu = n * p
    spq = np.sqrt(mu * (1 - p))
    c = mu + 0.5
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    alpha = (2.83 + 5.1 / b) * spq
    u_r = 0.43
    v_r = 0.92 - 4.2 / b
    m = np.floor((n + 1) * p).astype(n.dtype)
    log_p = np.log(p)
    log1_p = np.log1p(-p)
    log_h = (m + 0.5) * (np.log((m + 1.) / (n - m + 1.)) + log1_p - log_p) + \
            (stirling_approx_tail(m) + stirling_approx_tail(n - m))
    return _tr_params(c, b, a, alpha, u_r, v_r, m, log_p, log1_p, log_h)


def stirling_approx_tail(k):
    precomputed = np.array([
        0.08106146679532726,
        0.04134069595540929,
        0.02767792568499834,
        0.02079067210376509,
        0.01664469118982119,
        0.01387612882307075,
        0.01189670994589177,
        0.01041126526197209,
        0.009255462182712733,
        0.008330563433362871,
    ])
    kp1 = k + 1
    kp1sq = (k + 1) ** 2
    return np.where(k < 10,
                    precomputed[k],
                    (1. / 12 - (1. / 360 - (1. / 1260) / kp1sq) / kp1sq) / kp1)


def _binomial_btrs(key, p, n):
    """
    Based on the transformed rejection sampling algorithm (BTRS) from the
    following reference:

    Hormann, "The Generation of Binonmial Random Variates"
    (https://core.ac.uk/download/pdf/11007254.pdf)
    """
    def _btrs_body_fn(val):
        _, key, _, _ = val
        key, key_u, key_v = random.split(key, 3)
        u = random.uniform(key_u)
        v = random.uniform(key_v)
        u = u - 0.5
        k = np.floor((2 * tr_params.a / (0.5 - np.abs(u)) + tr_params.b) * u + tr_params.c).astype(n.dtype)
        return k, key, u, v

    def _btrs_cond_fn(val):
        def accept_fn(k, u, v):
            # See acceptance condition in Step 3. (Page 3) of TRS algorithm
            # v <= f(k) * g_grad(u) / alpha

            m = tr_params.m
            log_p = tr_params.log_p
            log1_p = tr_params.log1_p
            # See: formula for log(f(k)) at bottom of Page 5.
            log_f = (n + 1.) * np.log((n - m + 1.) / (n - k + 1.)) + \
                    (k + 0.5) * (np.log((n - k + 1.) / (k + 1.)) + log_p - log1_p) + \
                    (stirling_approx_tail(k) - stirling_approx_tail(n - k)) + tr_params.log_h
            g = (tr_params.a / (0.5 - np.abs(u)) ** 2) + tr_params.b
            return np.log((v * tr_params.alpha) / g) <= log_f

        k, key, u, v = val
        early_accept = (np.abs(u) <= tr_params.u_r) & (v <= tr_params.v_r)
        early_reject = (k < 0) | (k > n)
        return lax.cond(early_accept | early_reject,
                        (),
                        lambda _: ~early_accept,
                        (k, u, v),
                        lambda x: ~accept_fn(*x))

    tr_params = _get_tr_params(n, p)
    ret = lax.while_loop(_btrs_cond_fn, _btrs_body_fn,
                         (-1, key, 1., 1.))  # use k=-1 initially so that cond_fn returns True
    return ret[0]


def _binomial_inversion(key, p, n):
    def _binom_inv_body_fn(val):
        i, key, geom_acc = val
        key, key_u = random.split(key)
        u = random.uniform(key_u)
        geom = np.floor(np.log1p(-u) / log1_p) + 1
        geom_acc = geom_acc + geom
        return i + 1, key, geom_acc

    def _binom_inv_cond_fn(val):
        i, _, geom_acc = val
        return geom_acc <= n

    log1_p = np.log1p(-p)
    ret = lax.while_loop(_binom_inv_cond_fn, _binom_inv_body_fn,
                         (-1, key, 0.))
    return ret[0]


def _binomial_dispatch(key, p, n):
    def dispatch(key, p, n):
        is_le_mid = p <= 0.5
        pq = np.where(is_le_mid, p, 1 - p)
        mu = n * pq
        k = lax.cond(mu < 10,
                     (key, pq, n),
                     lambda x: _binomial_inversion(*x),
                     (key, pq, n),
                     lambda x: _binomial_btrs(*x))
        return np.where(is_le_mid, k, n - k)

    # Return 0 for nan `p` or negative `n`, since nan values are not allowed for integer types
    cond0 = np.isfinite(p) & (n > 0) & (p > 0)
    return lax.cond(cond0 & (p < 1),
                    (key, p, n),
                    lambda x: dispatch(*x),
                    (),
                    lambda _: np.where(cond0, n, 0))


@partial(jit, static_argnums=(3,))
def _binomial(key, p, n, shape):
    shape = shape or lax.broadcast_shapes(np.shape(p), np.shape(n))
    # reshape to map over axis 0
    p = np.reshape(np.broadcast_to(p, shape), -1)
    n = np.reshape(np.broadcast_to(n, shape), -1)
    key = random.split(key, np.size(p))
    if xla_bridge.get_backend().platform == 'cpu':
        ret = lax.map(lambda x: _binomial_dispatch(*x),
                      (key, p, n))
    else:
        ret = vmap(lambda *x: _binomial_dispatch(*x))(key, p, n)
    return np.reshape(ret, shape)


def binomial(key, p, n=1, shape=()):
    return _binomial(key, p, n, shape)


@partial(jit, static_argnums=(2,))
def _categorical(key, p, shape):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = np.cumsum(p, axis=-1)
    r = random.uniform(key, shape=shape + (1,))
    # FIXME: replace this computation by using binary search as suggested in the above
    # reference. A while_loop + vmap for a reshaped 2D array would be enough.
    return np.sum(s < r, axis=-1)


def categorical(key, p, shape=()):
    return _categorical(key, p, shape)


# TODO: drop this for the next JAX release, see https://github.com/google/jax/pull/1855
def categorical_logits(key, logits, shape=()):
    shape = shape or logits.shape[:-1]
    return np.argmax(random.gumbel(key, shape + logits.shape[-1:], logits.dtype)
                     + logits, axis=-1)


# Ref https://github.com/numpy/numpy/blob/8a0858f3903e488495a56b4a6d19bbefabc97dca/
# numpy/random/src/distributions/distributions.c#L574
def _poisson_large(val):
    rng_key, lam = val
    slam = np.sqrt(lam)
    loglam = np.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)

    def cond_fn(val):
        _, V, us, k = val
        cond1 = (us >= 0.07) & (V <= vr)
        cond2 = (k < 0) | ((us < 0.013) & (V > us))
        cond3 = ((np.log(V) + np.log(invalpha) - np.log(a / (us * us) + b))
                 <= (-lam + k * loglam - gammaln(k + 1)))

        # lax.cond in _poisson_one apparently may still
        # execute _poisson_large for small lam:
        # additional condition to not iterate if that is the case
        cond4 = lam >= 10
        return (~cond1) & (cond2 | (~cond3)) & cond4

    def body_fn(val):
        rng_key, *_ = val
        rng_key, key_U, key_V = random.split(rng_key, 3)
        U = random.uniform(key_U) - 0.5
        V = random.uniform(key_V)
        us = 0.5 - np.abs(U)
        k = np.floor((2 * a / us + b) * U + lam + 0.43)
        return rng_key, V, us, k

    *_, k = lax.while_loop(cond_fn, body_fn, (rng_key, 0., 0., -1.))
    return k


def _poisson_small(val):
    rng_key, lam = val
    enlam = np.exp(-lam)

    def cond_fn(val):
        cond1 = val[1] > enlam

        # lax.cond in _poisson_one apparently may still
        # execute _poisson_small for large lam:
        # additional condition to not iterate if that is the case
        cond2 = lam < 10
        return cond1 & cond2

    def body_fn(val):
        rng_key, prod, k = val
        rng_key, key_U = random.split(rng_key)
        U = random.uniform(key_U)
        prod = prod * U
        return rng_key, prod, k + 1

    init = np.where(lam == 0., 0., -1.)
    *_, k = lax.while_loop(cond_fn, body_fn, (rng_key, 1., init))
    return k


def _poisson_one(val):
    return lax.cond(val[1] >= 10, val, _poisson_large, val, _poisson_small)


@partial(jit, static_argnums=(2, 3))
def _poisson(key, rate, shape, dtype):
    # Ref: https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    shape = shape or np.shape(rate)
    rate = lax.convert_element_type(rate, canonicalize_dtype(np.float64))
    rate = np.broadcast_to(rate, shape)
    rng_keys = random.split(key, np.size(rate))
    if xla_bridge.get_backend().platform == 'cpu':
        k = lax.map(_poisson_one, (rng_keys, np.reshape(rate, -1)))
    else:
        k = vmap(_poisson_one)((rng_keys, np.reshape(rate, -1)))
    k = lax.convert_element_type(k, dtype)
    return np.reshape(k, shape)


def poisson(key, rate, shape=(), dtype=np.int64):
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


def cholesky_of_inverse(matrix):
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


def cholesky_update(L, x, coef=1):
    """
    Finds cholesky of L @ L.T + coef * x @ x.T.

    **References;**

        1. A more efficient rank-one covariance matrix update for evolution strategies,
           Oswin Krause and Christian Igel
    """
    batch_shape = lax.broadcast_shapes(L.shape[:-2], x.shape[:-1])
    L = np.broadcast_to(L, batch_shape + L.shape[-2:])
    x = np.broadcast_to(x, batch_shape + x.shape[-1:])
    diag = np.diagonal(L, axis1=-2, axis2=-1)
    # convert to unit diagonal triangular matrix: L @ D @ T.t
    L = L / diag[..., None, :]
    D = np.square(diag)

    def scan_fn(carry, val):
        b, w = carry
        j, Dj, L_j = val
        wj = w[..., j]
        gamma = b * Dj + coef * np.square(wj)
        Dj_new = gamma / b
        b = gamma / Dj_new

        # update vectors w and L_j
        w = w - wj[..., None] * L_j
        L_j = L_j + (coef * wj / gamma)[..., None] * w
        return (b, w), (Dj_new, L_j)

    D, L = np.moveaxis(D, -1, 0), np.moveaxis(L, -1, 0)  # move scan dim to front
    _, (D, L) = lax.scan(scan_fn, (np.ones(batch_shape), x), (np.arange(D.shape[0]), D, L))
    D, L = np.moveaxis(D, 0, -1), np.moveaxis(L, 0, -1)  # move scan dim back
    return L * np.sqrt(D)[..., None, :]


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
    z1m_cumprod = np.cumprod(1 - z, axis=-1)
    z1m_cumprod_sqrt = np.sqrt(z1m_cumprod)

    pad_width = [(0, 0)] * z.ndim
    pad_width[-1] = (1, 0)
    z1m_cumprod_sqrt_shifted = np.pad(z1m_cumprod_sqrt[..., :-1], pad_width,
                                      mode="constant", constant_values=1.)
    y = (r + np.identity(r.shape[-1])) * z1m_cumprod_sqrt_shifted
    return y


def logmatmulexp(x, y):
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``.
    """
    x_shift = lax.stop_gradient(np.amax(x, -1, keepdims=True))
    y_shift = lax.stop_gradient(np.amax(y, -2, keepdims=True))
    xy = np.log(np.matmul(np.exp(x - x_shift), np.exp(y - y_shift)))
    return xy + x_shift + y_shift


def logsumexp(x, axis=0, keepdims=False):
    # TODO: remove when https://github.com/google/jax/pull/2260 merged upstream
    x_max = lax.stop_gradient(np.amax(x, axis=axis, keepdims=True))
    y = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return y if keepdims else y.squeeze(axis=axis)


def clamp_probs(probs):
    finfo = np.finfo(get_dtype(probs))
    return np.clip(probs, a_min=finfo.tiny, a_max=1. - finfo.eps)



@partial(jit, static_argnums=(2, 3))
def _von_mises_centered(key, concentration, shape, dtype):
    """ Compute centered von Mises samples using rejection sampling from [1] with wrapped Cauchy proposal.

        *** References ***
        [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag, 1986;
            Chapter 9, p. 473-476. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf

        :param sample_shape: shape of samples
        :param rng_key: random number generator key
        :return: centered samples from von Mises
    """

    # Cutoff from TensorFlow probability
    # (https://github.com/tensorflow/probability/blob/f051e03dd3cc847d31061803c2b31c564562a993/tensorflow_probability/python/distributions/von_mises.py#L567-L570)
    s_cutoff_map = {np.dtype(np.float16): 1.8e-1,
                    np.dtype(np.float32): 2e-2,
                    np.dtype(np.float64): 1.2e-4}

    r = 1. + np.sqrt(1. + 4. * concentration ** 2)
    rho = (r - np.sqrt(2. * r)) / (2. * concentration)
    s_exact = (1. + rho ** 2) / (2. * rho)

    s_approximate = 1. / concentration

    s_cutoff = s_cutoff_map.get(dtype)

    s = np.where(concentration > s_cutoff, s_exact, s_approximate)

    def body_function(i, *args):
        done, _, w = args[0]
        nonlocal key
        uni_ukey, uni_vkey, key = random.split(key, 3)

        u = random.uniform(key=uni_ukey, shape=shape, dtype=concentration.dtype, minval=-1., maxval=1.)
        z = np.cos(np.pi * u)
        w = np.where(done, w, (1. + s * z) / (s + z))  # Update where not done

        y = concentration * (s - w)
        v = random.uniform(key=uni_vkey, shape=shape, dtype=concentration.dtype, minval=-1., maxval=1.)

        accept = (y * (2. - y) >= v) | (np.log(y / v) + 1. >= y)

        return accept | done, u, w

    init_done = np.zeros(shape, dtype=bool)
    init_u = np.zeros(shape)
    init_w = np.zeros(shape)

    done, u, w = lax.fori_loop(
        lower=0,
        upper=100,
        body_fun=body_function,
        init_val=(init_done, init_u, init_w)
    )

    # cond_fun=lambda done, *_: np.logical_not(np.all(done)),
    return np.sign(u) * np.arccos(w)

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
