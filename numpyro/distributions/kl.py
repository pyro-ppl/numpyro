# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# The implementation follows the design in PyTorch: torch.distributions.kl.py
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

from multipledispatch import dispatch

from jax import lax
import jax.numpy as jnp
from jax.scipy.special import betaln, digamma, gammaln

from numpyro.distributions.continuous import (
    Beta,
    Dirichlet,
    Gamma,
    Kumaraswamy,
    Normal,
    Weibull,
)
from numpyro.distributions.distribution import (
    Delta,
    Distribution,
    ExpandedDistribution,
    Independent,
    MaskedDistribution,
)
from numpyro.distributions.util import scale_and_mask, sum_rightmost


def kl_divergence(p, q):
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.
    """
    raise NotImplementedError


################################################################################
# KL Divergence Implementations
################################################################################


@dispatch(Distribution, ExpandedDistribution)
def kl_divergence(p, q):
    kl = kl_divergence(p, q.base_dist)
    shape = lax.broadcast_shapes(p.batch_shape, q.batch_shape)
    return jnp.broadcast_to(kl, shape)


@dispatch(ExpandedDistribution, Distribution)
def kl_divergence(p, q):
    kl = kl_divergence(p.base_dist, q)
    shape = lax.broadcast_shapes(p.batch_shape, q.batch_shape)
    return jnp.broadcast_to(kl, shape)


@dispatch(ExpandedDistribution, ExpandedDistribution)
def kl_divergence(p, q):
    kl = kl_divergence(p.base_dist, q.base_dist)
    shape = lax.broadcast_shapes(p.batch_shape, q.batch_shape)
    return jnp.broadcast_to(kl, shape)


@dispatch(Delta, Distribution)
def kl_divergence(p, q):
    return -q.log_prob(p.v)


@dispatch(Delta, ExpandedDistribution)
def kl_divergence(p, q):
    return -q.log_prob(p.v)


@dispatch(Independent, Independent)
def kl_divergence(p, q):
    shared_ndims = min(p.reinterpreted_batch_ndims, q.reinterpreted_batch_ndims)
    p_ndims = p.reinterpreted_batch_ndims - shared_ndims
    q_ndims = q.reinterpreted_batch_ndims - shared_ndims
    p = Independent(p.base_dist, p_ndims) if p_ndims else p.base_dist
    q = Independent(q.base_dist, q_ndims) if q_ndims else q.base_dist
    kl = kl_divergence(p, q)
    if shared_ndims:
        kl = sum_rightmost(kl, shared_ndims)
    return kl


@dispatch(MaskedDistribution, MaskedDistribution)
def kl_divergence(p, q):
    if p._mask is False or q._mask is False:
        mask = False
    elif p._mask is True:
        mask = q._mask
    elif q._mask is True:
        mask = p._mask
    elif p._mask is q._mask:
        mask = p._mask
    else:
        mask = p._mask & q._mask

    if mask is False:
        return 0.0
    if mask is True:
        return kl_divergence(p.base_dist, q.base_dist)
    kl = kl_divergence(p.base_dist, q.base_dist)
    return scale_and_mask(kl, mask=mask)


@dispatch(Normal, Normal)
def kl_divergence(p, q):
    var_ratio = jnp.square(p.scale / q.scale)
    t1 = jnp.square((p.loc - q.loc) / q.scale)
    return 0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio))


@dispatch(Beta, Beta)
def kl_divergence(p, q):
    # From https://en.wikipedia.org/wiki/Beta_distribution#Quantities_of_information_(entropy)
    a, b = p.concentration1, p.concentration0
    alpha, beta = q.concentration1, q.concentration0
    a_diff = alpha - a
    b_diff = beta - b
    t1 = betaln(alpha, beta) - betaln(a, b)
    t2 = a_diff * digamma(a) + b_diff * digamma(b)
    t3 = (a_diff + b_diff) * digamma(a + b)
    return t1 - t2 + t3


@dispatch(Dirichlet, Dirichlet)
def kl_divergence(p, q):
    # From http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    sum_p_concentration = p.concentration.sum(-1)
    sum_q_concentration = q.concentration.sum(-1)
    t1 = gammaln(sum_p_concentration) - gammaln(sum_q_concentration)
    t2 = (gammaln(p.concentration) - gammaln(q.concentration)).sum(-1)
    t3 = p.concentration - q.concentration
    t4 = digamma(p.concentration) - digamma(sum_p_concentration)[..., None]
    return t1 - t2 + (t3 * t4).sum(-1)


@dispatch(Gamma, Gamma)
def kl_divergence(p, q):
    # From https://en.wikipedia.org/wiki/Gamma_distribution#Kullback%E2%80%93Leibler_divergence
    a, b = p.concentration, p.rate
    alpha, beta = q.concentration, q.rate
    b_ratio = beta / b
    t1 = gammaln(alpha) - gammaln(a)
    t2 = (a - alpha) * digamma(a)
    t3 = alpha * jnp.log(b_ratio)
    t4 = a * (b_ratio - 1)
    return t1 + t2 - t3 + t4


@dispatch(Weibull, Gamma)
def kl_divergence(p, q):
    # From https://arxiv.org/abs/1401.6853 Formula (28)
    a, b = p.concentration, p.scale
    alpha, beta = q.concentration, q.rate
    a_reciprocal = 1 / a
    b_beta = b * beta
    t1 = jnp.log(a) + gammaln(alpha)
    t2 = alpha * (jnp.euler_gamma * a_reciprocal - jnp.log(b_beta))
    t3 = b_beta * jnp.exp(gammaln(a_reciprocal + 1))
    return t1 + t2 + t3 - (jnp.euler_gamma + 1)


@dispatch(Kumaraswamy, Beta)
def kl_divergence(p, q):
    # From https://arxiv.org/abs/1605.06197 Formula (12)
    a, b = p.concentration1, p.concentration0
    alpha, beta = q.concentration1, q.concentration0
    b_reciprocal = jnp.reciprocal(b)
    a_b = a * b
    t1 = (alpha / a - 1) * (jnp.euler_gamma + digamma(b) + b_reciprocal)
    t2 = jnp.log(a_b) + betaln(alpha, beta) + (b_reciprocal - 1)
    a_ = jnp.expand_dims(a, -1)
    b_ = jnp.expand_dims(b, -1)
    a_b_ = jnp.expand_dims(a_b, -1)
    m = jnp.arange(1, p.KL_KUMARASWAMY_BETA_TAYLOR_ORDER + 1)
    t3 = (beta - 1) * b * (jnp.exp(betaln(m / a_, b_)) / (m + a_b_)).sum(-1)
    return t1 + t2 + t3
