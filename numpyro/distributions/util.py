# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial, update_wrapper
import math

import numpy as np

import jax
from jax import jit, lax, random, vmap
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

# Parameters for Transformed Rejection with Squeeze (TRS) algorithm - page 3.
_tr_params = namedtuple(
    "tr_params", ["c", "b", "a", "alpha", "u_r", "v_r", "m", "log_p", "log1_p", "log_h"]
)


def _get_tr_params(n, p):
    # See Table 1. Additionally, we pre-compute log(p), log1(-p) and the
    # constant terms, that depend only on (n, p, m) in log(f(k)) (bottom of page 5).
    mu = n * p
    spq = jnp.sqrt(mu * (1 - p))
    c = mu + 0.5
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    alpha = (2.83 + 5.1 / b) * spq
    u_r = 0.43
    v_r = 0.92 - 4.2 / b
    m = jnp.floor((n + 1) * p).astype(n.dtype)
    log_p = jnp.log(p)
    log1_p = jnp.log1p(-p)
    log_h = (m + 0.5) * (jnp.log((m + 1.0) / (n - m + 1.0)) + log1_p - log_p) + (
        stirling_approx_tail(m) + stirling_approx_tail(n - m)
    )
    return _tr_params(c, b, a, alpha, u_r, v_r, m, log_p, log1_p, log_h)


def stirling_approx_tail(k):
    precomputed = jnp.array(
        [
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
        ]
    )
    kp1 = k + 1
    kp1sq = (k + 1) ** 2
    return jnp.where(
        k < 10,
        precomputed[k],
        (1.0 / 12 - (1.0 / 360 - (1.0 / 1260) / kp1sq) / kp1sq) / kp1,
    )


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
        k = jnp.floor(
            (2 * tr_params.a / (0.5 - jnp.abs(u)) + tr_params.b) * u + tr_params.c
        ).astype(n.dtype)
        return k, key, u, v

    def _btrs_cond_fn(val):
        def accept_fn(k, u, v):
            # See acceptance condition in Step 3. (Page 3) of TRS algorithm
            # v <= f(k) * g_grad(u) / alpha

            m = tr_params.m
            log_p = tr_params.log_p
            log1_p = tr_params.log1_p
            # See: formula for log(f(k)) at bottom of Page 5.
            log_f = (
                (n + 1.0) * jnp.log((n - m + 1.0) / (n - k + 1.0))
                + (k + 0.5) * (jnp.log((n - k + 1.0) / (k + 1.0)) + log_p - log1_p)
                + (stirling_approx_tail(k) - stirling_approx_tail(n - k))
                + tr_params.log_h
            )
            g = (tr_params.a / (0.5 - jnp.abs(u)) ** 2) + tr_params.b
            return jnp.log((v * tr_params.alpha) / g) <= log_f

        k, key, u, v = val
        early_accept = (jnp.abs(u) <= tr_params.u_r) & (v <= tr_params.v_r)
        early_reject = (k < 0) | (k > n)
        return lax.cond(
            early_accept | early_reject,
            (),
            lambda _: ~early_accept,
            (k, u, v),
            lambda x: ~accept_fn(*x),
        )

    tr_params = _get_tr_params(n, p)
    ret = lax.while_loop(
        _btrs_cond_fn, _btrs_body_fn, (-1, key, 1.0, 1.0)
    )  # use k=-1 initially so that cond_fn returns True
    return ret[0]


def _binomial_inversion(key, p, n):
    def _binom_inv_body_fn(val):
        i, key, geom_acc = val
        key, key_u = random.split(key)
        u = random.uniform(key_u)
        geom = jnp.floor(jnp.log1p(-u) / log1_p) + 1
        geom_acc = geom_acc + geom
        return i + 1, key, geom_acc

    def _binom_inv_cond_fn(val):
        i, _, geom_acc = val
        return geom_acc <= n

    log1_p = jnp.log1p(-p)
    ret = lax.while_loop(_binom_inv_cond_fn, _binom_inv_body_fn, (-1, key, 0.0))
    return ret[0]


def _binomial_dispatch(key, p, n):
    def dispatch(key, p, n):
        is_le_mid = p <= 0.5
        pq = jnp.where(is_le_mid, p, 1 - p)
        mu = n * pq
        k = lax.cond(
            mu < 10,
            (key, pq, n),
            lambda x: _binomial_inversion(*x),
            (key, pq, n),
            lambda x: _binomial_btrs(*x),
        )
        return jnp.where(is_le_mid, k, n - k)

    # Return 0 for nan `p` or negative `n`, since nan values are not allowed for integer types
    cond0 = jnp.isfinite(p) & (n > 0) & (p > 0)
    return lax.cond(
        cond0 & (p < 1),
        (key, p, n),
        lambda x: dispatch(*x),
        (),
        lambda _: jnp.where(cond0, n, 0),
    )


@partial(jit, static_argnums=(3,))
def _binomial(key, p, n, shape):
    shape = shape or lax.broadcast_shapes(jnp.shape(p), jnp.shape(n))
    # reshape to map over axis 0
    p = jnp.reshape(jnp.broadcast_to(p, shape), -1)
    n = jnp.reshape(jnp.broadcast_to(n, shape), -1)
    key = random.split(key, jnp.size(p))
    if jax.default_backend() == "cpu":
        ret = lax.map(lambda x: _binomial_dispatch(*x), (key, p, n))
    else:
        ret = vmap(lambda *x: _binomial_dispatch(*x))(key, p, n)
    return jnp.reshape(ret, shape)


def binomial(key, p, n=1, shape=()):
    return _binomial(key, p, n, shape)


@partial(jit, static_argnums=(2,))
def _categorical(key, p, shape):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = jnp.cumsum(p, axis=-1)
    r = random.uniform(key, shape=shape + (1,))
    # FIXME: replace this computation by using binary search as suggested in the above
    # reference. A while_loop + vmap for a reshaped 2D array would be enough.
    return jnp.sum(s < r, axis=-1)


def categorical(key, p, shape=()):
    return _categorical(key, p, shape)


def _scatter_add_one(operand, indices, updates):
    return lax.scatter_add(
        operand,
        indices,
        updates,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        ),
    )


@partial(jit, static_argnums=(3, 4))
def _multinomial(key, p, n, n_max, shape=()):
    if jnp.shape(n) != jnp.shape(p)[:-1]:
        broadcast_shape = lax.broadcast_shapes(jnp.shape(n), jnp.shape(p)[:-1])
        n = jnp.broadcast_to(n, broadcast_shape)
        p = jnp.broadcast_to(p, broadcast_shape + jnp.shape(p)[-1:])
    shape = shape or p.shape[:-1]
    if n_max == 0:
        return jnp.zeros(shape + p.shape[-1:], dtype=jnp.result_type(int))
    # get indices from categorical distribution then gather the result
    indices = categorical(key, p, (n_max,) + shape)
    # mask out values when counts is heterogeneous
    if jnp.ndim(n) > 0:
        mask = promote_shapes(
            jnp.arange(n_max) < jnp.expand_dims(n, -1), shape=shape + (n_max,)
        )[0]
        mask = jnp.moveaxis(mask, -1, 0).astype(indices.dtype)
        excess = jnp.concatenate(
            [
                jnp.expand_dims(n_max - n, -1),
                jnp.zeros(jnp.shape(n) + (p.shape[-1] - 1,)),
            ],
            -1,
        )
    else:
        mask = 1
        excess = 0
    # NB: we transpose to move batch shape to the front
    indices_2D = (jnp.reshape(indices * mask, (n_max, -1))).T
    samples_2D = vmap(_scatter_add_one, (0, 0, 0))(
        jnp.zeros((indices_2D.shape[0], p.shape[-1]), dtype=indices.dtype),
        jnp.expand_dims(indices_2D, axis=-1),
        jnp.ones(indices_2D.shape, dtype=indices.dtype),
    )
    return jnp.reshape(samples_2D, shape + p.shape[-1:]) - excess


def multinomial(key, p, n, shape=()):
    assert not isinstance(
        n, jax.core.Tracer
    ), "The total count parameter `n` should not be a jax abstract array."
    n_max = int(np.max(jax.device_get(n)))
    return _multinomial(key, p, n, n_max, shape)


def cholesky_of_inverse(matrix):
    # This formulation only takes the inverse of a triangular matrix
    # which is more numerically stable.
    # Refer to:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = jnp.swapaxes(
        jnp.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = jnp.broadcast_to(jnp.identity(matrix.shape[-1]), tril_inv.shape)
    return solve_triangular(tril_inv, identity, lower=True)


# TODO: move upstream to jax.nn
def binary_cross_entropy_with_logits(x, y):
    # compute -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))
    # Ref: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return jnp.clip(x, 0) + jnp.log1p(jnp.exp(-jnp.abs(x))) - x * y


def _reshape(x, shape):
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return np.reshape(x, shape)
    else:
        return jnp.reshape(x, shape)


def promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [jnp.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [
            _reshape(arg, (1,) * (num_dims - len(s)) + s) if len(s) < num_dims else arg
            for arg, s in zip(args, shapes)
        ]


def sum_rightmost(x, dim):
    """
    Sum out ``dim`` many rightmost dimensions of a given tensor.
    """
    out_dim = jnp.ndim(x) - dim
    x = jnp.reshape(jnp.expand_dims(x, -1), jnp.shape(x)[:out_dim] + (-1,))
    return jnp.sum(x, axis=-1)


def matrix_to_tril_vec(x, diagonal=0):
    idxs = jnp.tril_indices(x.shape[-1], diagonal)
    return x[..., idxs[0], idxs[1]]


def vec_to_tril_matrix(t, diagonal=0):
    # NB: the following formula only works for diagonal <= 0
    n = round((math.sqrt(1 + 8 * t.shape[-1]) - 1) / 2) - diagonal
    n2 = n * n
    idx = jnp.reshape(jnp.arange(n2), (n, n))[jnp.tril_indices(n, diagonal)]
    x = lax.scatter_add(
        jnp.zeros(t.shape[:-1] + (n2,)),
        jnp.expand_dims(idx, axis=-1),
        t,
        lax.ScatterDimensionNumbers(
            update_window_dims=range(t.ndim - 1),
            inserted_window_dims=(t.ndim - 1,),
            scatter_dims_to_operand_dims=(t.ndim - 1,),
        ),
    )
    return jnp.reshape(x, x.shape[:-1] + (n, n))


def cholesky_update(L, x, coef=1):
    """
    Finds cholesky of L @ L.T + coef * x @ x.T.

    **References;**

        1. A more efficient rank-one covariance matrix update for evolution strategies,
           Oswin Krause and Christian Igel
    """
    batch_shape = lax.broadcast_shapes(L.shape[:-2], x.shape[:-1])
    L = jnp.broadcast_to(L, batch_shape + L.shape[-2:])
    x = jnp.broadcast_to(x, batch_shape + x.shape[-1:])
    diag = jnp.diagonal(L, axis1=-2, axis2=-1)
    # convert to unit diagonal triangular matrix: L @ D @ T.t
    L = L / diag[..., None, :]
    D = jnp.square(diag)

    def scan_fn(carry, val):
        b, w = carry
        j, Dj, L_j = val
        wj = w[..., j]
        gamma = b * Dj + coef * jnp.square(wj)
        Dj_new = gamma / b
        b = gamma / Dj_new

        # update vectors w and L_j
        w = w - wj[..., None] * L_j
        L_j = L_j + (coef * wj / gamma)[..., None] * w
        return (b, w), (Dj_new, L_j)

    D, L = jnp.moveaxis(D, -1, 0), jnp.moveaxis(L, -1, 0)  # move scan dim to front
    _, (D, L) = lax.scan(
        scan_fn, (jnp.ones(batch_shape), x), (jnp.arange(D.shape[0]), D, L)
    )
    D, L = jnp.moveaxis(D, 0, -1), jnp.moveaxis(L, 0, -1)  # move scan dim back
    return L * jnp.sqrt(D)[..., None, :]


def signed_stick_breaking_tril(t):
    # make sure that t in (-1, 1)
    eps = jnp.finfo(t.dtype).eps
    t = jnp.clip(t, a_min=(-1 + eps), a_max=(1 - eps))
    # transform t to tril matrix with identity diagonal
    r = vec_to_tril_matrix(t, diagonal=-1)

    # apply stick-breaking on the squared values;
    # we omit the step of computing s = z * z_cumprod by using the fact:
    #     y = sign(r) * s = sign(r) * sqrt(z * z_cumprod) = r * sqrt(z_cumprod)
    z = r ** 2
    z1m_cumprod_sqrt = jnp.cumprod(jnp.sqrt(1 - z), axis=-1)

    pad_width = [(0, 0)] * z.ndim
    pad_width[-1] = (1, 0)
    z1m_cumprod_sqrt_shifted = jnp.pad(
        z1m_cumprod_sqrt[..., :-1], pad_width, mode="constant", constant_values=1.0
    )
    y = (r + jnp.identity(r.shape[-1])) * z1m_cumprod_sqrt_shifted
    return y


def logmatmulexp(x, y):
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``.
    """
    x_shift = lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift


def clamp_probs(probs):
    finfo = jnp.finfo(jnp.result_type(probs))
    return jnp.clip(probs, a_min=finfo.tiny, a_max=1.0 - finfo.eps)


def is_identically_zero(x):
    """
    Check if argument is exactly the number zero. True for the number zero;
    false for other numbers; false for ndarrays.
    """
    if isinstance(x, (int, float)):
        return x == 0
    else:
        return False


def is_identically_one(x):
    """
    Check if argument is exactly the number one. True for the number one;
    false for other numbers; false for ndarrays.
    """
    if isinstance(x, (int, float)):
        return x == 1
    else:
        return False


def von_mises_centered(key, concentration, shape=(), dtype=jnp.float64):
    """Compute centered von Mises samples using rejection sampling from [1] with wrapped Cauchy proposal.

    *** References ***
    [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag, 1986;
        Chapter 9, p. 473-476. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf


    :param key: random number generator key
    :param concentration: concentration of distribution
    :param shape: shape of samples
    :param dtype: float precesions for choosing correct s cutfoff
    :return: centered samples from von Mises
    """
    shape = shape or jnp.shape(concentration)
    dtype = jnp.result_type(dtype)
    concentration = lax.convert_element_type(concentration, dtype)
    concentration = jnp.broadcast_to(concentration, shape)
    return _von_mises_centered(key, concentration, shape, dtype)


@partial(jit, static_argnums=(2, 3))
def _von_mises_centered(key, concentration, shape, dtype):
    # Cutoff from TensorFlow probability
    # (https://github.com/tensorflow/probability/blob/f051e03dd3cc847d31061803c2b31c564562a993/tensorflow_probability/python/distributions/von_mises.py#L567-L570)
    s_cutoff_map = {
        jnp.dtype(jnp.float16): 1.8e-1,
        jnp.dtype(jnp.float32): 2e-2,
        jnp.dtype(jnp.float64): 1.2e-4,
    }
    s_cutoff = s_cutoff_map.get(dtype)

    r = 1.0 + jnp.sqrt(1.0 + 4.0 * concentration ** 2)
    rho = (r - jnp.sqrt(2.0 * r)) / (2.0 * concentration)
    s_exact = (1.0 + rho ** 2) / (2.0 * rho)

    s_approximate = 1.0 / concentration

    s = jnp.where(concentration > s_cutoff, s_exact, s_approximate)

    def cond_fn(*args):
        """check if all are done or reached max number of iterations"""
        i, _, done, _, _ = args[0]
        return jnp.bitwise_and(i < 100, jnp.logical_not(jnp.all(done)))

    def body_fn(*args):
        i, key, done, _, w = args[0]
        uni_ukey, uni_vkey, key = random.split(key, 3)

        u = random.uniform(
            key=uni_ukey,
            shape=shape,
            dtype=concentration.dtype,
            minval=-1.0,
            maxval=1.0,
        )
        z = jnp.cos(jnp.pi * u)
        w = jnp.where(done, w, (1.0 + s * z) / (s + z))  # Update where not done

        y = concentration * (s - w)
        v = random.uniform(key=uni_vkey, shape=shape, dtype=concentration.dtype)

        accept = (y * (2.0 - y) >= v) | (jnp.log(y / v) + 1.0 >= y)

        return i + 1, key, accept | done, u, w

    init_done = jnp.zeros(shape, dtype=bool)
    init_u = jnp.zeros(shape)
    init_w = jnp.zeros(shape)

    _, _, done, u, w = lax.while_loop(
        cond_fun=cond_fn,
        body_fun=body_fn,
        init_val=(jnp.array(0), key, init_done, init_u, init_w),
    )

    return jnp.sign(u) * jnp.arccos(w)


def scale_and_mask(x, scale=None, mask=None):
    """
    Scale and mask a tensor, broadcasting and avoiding unnecessary ops.
    """
    if is_identically_zero(x):
        return x
    if not (scale is None or is_identically_one(scale)):
        x = x * scale
    if mask is None:
        return x
    else:
        return jnp.where(mask, x, 0.0)


# TODO: use funsor implementation
def periodic_repeat(x, size, dim):
    """
    Repeat a ``period``-sized array up to given ``size``.
    """
    assert isinstance(size, int) and size >= 0
    assert isinstance(dim, int)
    if dim >= 0:
        dim -= jnp.ndim(x)

    period = jnp.shape(x)[dim]
    repeats = (size + period - 1) // period
    result = jnp.repeat(x, repeats, axis=dim)
    result = result[(Ellipsis, slice(None, size)) + (slice(None),) * (-1 - dim)]
    return result


def safe_normalize(x, *, p=2):
    """
    Safely project a vector onto the sphere wrt the ``p``-norm. This avoids the
    singularity at zero by mapping zero to the uniform unit vector proportional
    to ``[1, 1, ..., 1]``.

    :param numpy.ndarray x: A vector
    :param float p: The norm exponent, defaults to 2 i.e. the Euclidean norm.
    :returns: A normalized version ``x / ||x||_p``.
    :rtype: numpy.ndarray
    """
    assert isinstance(p, (float, int))
    assert p >= 0
    norm = jnp.linalg.norm(x, p, axis=-1, keepdims=True)
    x = x / jnp.clip(norm, a_min=jnp.finfo(x).tiny)
    # Avoid the singularity.
    mask = jnp.all(x == 0, axis=-1, keepdims=True)
    x = jnp.where(mask, x.shape[-1] ** (-1 / p), x)
    return x


# src: https://github.com/google/jax/blob/5a41779fbe12ba7213cd3aa1169d3b0ffb02a094/jax/_src/random.py#L95
def is_prng_key(key):
    try:
        return key.shape == (2,) and key.dtype == np.uint32
    except AttributeError:
        return False


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

    # This is to prevent warnings from sphinx
    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

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
            value = kwargs["value"] if "value" in kwargs else args[0]
            mask = self._validate_sample(value)
            log_prob = jnp.where(mask, log_prob, -jnp.inf)
        return log_prob

    return wrapper
