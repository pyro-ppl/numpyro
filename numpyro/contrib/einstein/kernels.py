# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.random as npr

import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats

from numpyro.contrib.einstein.utils import posdef, safe_norm, sqrth
import numpyro.distributions as dist


class PrecondMatrix(ABC):
    @abstractmethod
    def compute(self, particles: jnp.ndarray, loss_fn: Callable[[jnp.ndarray], float]):
        """
        Computes a preconditioning matrix for a given set of particles and a loss function

        :param particles: The Stein particles to compute the preconditioning matrix from
        :param loss_fn: Loss function given particles
        """
        raise NotImplementedError


class SteinKernel(ABC):
    @property
    @abstractmethod
    def mode(self):
        """
        Returns the type of kernel, either 'norm' or 'vector' or 'matrix'.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        particles: jnp.ndarray,
        particle_info: Dict[str, Tuple[int, int]],
        loss_fn: Callable[[jnp.ndarray], float],
    ):
        """
        Computes the kernel function given the input Stein particles

        :param particles: The Stein particles to compute the kernel from
        :param particle_info: A mapping from parameter names to the position in the particle matrix
        :param loss_fn: Loss function given particles
        :return: The kernel_fn to compute kernel for pair of particles.
                 Modes: norm `(d,) (d,)-> ()`,  vector `(d,) (d,) -> (d)`, or matrix `(d,) (d,) -> (d,d)`
        """
        raise NotImplementedError


class RBFKernel(SteinKernel):
    """
    Calculates the Gaussian RBF kernel function, from [1],
    :math: `k(x,y) = \\exp(\\frac{1}{h} \\|x-y\\|^2)`,
    where the bandwidth h is computed using the median heuristic
    :math: `h = \\frac{1}{\\log(n)} \\med(\\|x-y\\|)`.

    ** References: **
    1. *Stein Variational Gradient Descent*  Liu and Wang

    :param str mode: Either 'norm' (default) specifying to take the norm of each particle, 'vector' to return a
                 component-wise kernel or 'matrix' to return a matrix-valued kernel
    :param str matrix_mode: Either 'norm_diag' (default) for diagonal filled with the norm kernel or 'vector_diag'
                        for diagonal of vector-valued kernel
    :param bandwidth_factor: A multiplier to the bandwidth based on data size n (default 1/log(n))
    """

    def __init__(
        self,
        mode="norm",
        matrix_mode="norm_diag",
        bandwidth_factor: Callable[[float], float] = lambda n: 1 / jnp.log(n),
    ):
        assert mode == "norm" or mode == "vector" or mode == "matrix"
        assert matrix_mode == "norm_diag" or matrix_mode == "vector_diag"
        self._mode = mode
        self.matrix_mode = matrix_mode
        self.bandwidth_factor = bandwidth_factor

    def _normed(self):
        return self._mode == "norm" or (
            self.mode == "matrix" and self.matrix_mode == "norm_diag"
        )

    def compute(self, particles, particle_info, loss_fn):
        diffs = jnp.expand_dims(particles, axis=0) - jnp.expand_dims(
            particles, axis=1
        )  # N x N (x D)
        if self._normed() and particles.ndim == 2:
            diffs = safe_norm(diffs, ord=2, axis=-1)  # N x D -> N
        diffs = jnp.reshape(diffs, (diffs.shape[0] * diffs.shape[1], -1))  # N * N (x D)
        factor = self.bandwidth_factor(particles.shape[0])
        if diffs.ndim == 2:
            diff_norms = safe_norm(diffs, ord=2, axis=-1)
        else:
            diff_norms = diffs
        bandwidth = jnp.median(diff_norms) ** 2 * factor + 1e-5

        def kernel(x, y):
            diff = safe_norm(x - y, ord=2) if self._normed() and x.ndim >= 1 else x - y
            kernel_res = jnp.exp(-(diff ** 2) / bandwidth)
            if self._mode == "matrix":
                if self.matrix_mode == "norm_diag":
                    return kernel_res * jnp.identity(x.shape[0])
                else:
                    return jnp.diag(kernel_res)
            else:
                return kernel_res

        return kernel

    @property
    def mode(self):
        return self._mode


class IMQKernel(SteinKernel):
    """
    Calculates the IMQ kernel
    :math:`k(x,y) = (c^2 + \\|x+y\\|^2_2)^{\\beta},`
    from [1].

    ** References: **
    1. *Measuring Sample Quality with Kernels* by Gorham and Mackey

    :param str mode: Either 'norm' (default) specifying to take the norm of each particle,
                 or 'vector' to return a component-wise kernel
    :param float const: Positive multi-quadratic constant (c)
    :param float expon: Inverse exponent (beta) between (-1, 0)
    """

    def __init__(self, mode="norm", const=1.0, expon=-0.5):
        assert mode == "norm" or mode == "vector"
        assert 0.0 < const
        assert -1.0 < expon < 0.0
        self._mode = mode
        self.const = const
        self.expon = expon

    @property
    def mode(self):
        return self._mode

    def compute(self, particles, particle_info, loss_fn):
        def kernel(x, y):
            diff = safe_norm(x - y, ord=2, axis=-1) if self._mode == "norm" else x - y
            return (self.const ** 2 + diff ** 2) ** self.expon

        return kernel


class LinearKernel(SteinKernel):
    """
    Calculates the linear kernel
    :math: `k(x,y) = x \\cdot y + 1`
    from [1].

    ** References **
    1. Stein Variational Gradient Descent as Moment Matching" by Liu and Wang
    """

    def __init__(self, mode="norm"):
        assert mode == "norm"
        self._mode = "norm"

    @property
    def mode(self):
        return self._mode

    def compute(self, particles: jnp.ndarray, particle_info, loss_fn):
        def kernel(x, y):
            if x.ndim == 1:
                return x @ y + 1
            else:
                return x * y + 1

        return kernel


class RandomFeatureKernel(SteinKernel):
    """
    Calculates the random kernel
    :math:`k(x,y)= 1/m\\sum_{l=1}^{m}\\phi(x,w_l)\\phi(y,w_l),
    from [1].


    ** References: **
    1. *Stein Variational Gradient Descent as Moment Matching* by Liu and Wang

    :param bandwidth_subset: How many particles should be used to calculate the bandwidth?
                             (default None, meaning all particles)
    :param random_indices: The set of indices which to do random feature expansion on.
                           (default None, meaning all indices)
    :param bandwidth_factor: A multiplier to the bandwidth based on data size n (default 1/log(n))

    """

    def __init__(
        self,
        mode="norm",
        bandwidth_subset=None,
        random_indices=None,
        bandwidth_factor: Callable[[float], float] = lambda n: 1 / jnp.log(n),
    ):
        assert bandwidth_subset is None or bandwidth_subset > 0
        assert mode == "norm"
        self._mode = "norm"
        self.bandwidth_subset = bandwidth_subset
        self.random_indices = None
        self.bandwidth_factor = bandwidth_factor
        self._random_weights = None
        self._random_biases = None

    @property
    def mode(self):
        return self._mode

    def compute(self, particles, particle_info, loss_fn):
        if self._random_weights is None:
            self._random_weights = jnp.array(npr.randn(*particles.shape))
            self._random_biases = jnp.array(npr.rand(*particles.shape) * 2 * np.pi)
        factor = self.bandwidth_factor(particles.shape[0])
        if self.bandwidth_subset is not None:
            particles = particles[npr.choice(particles.shape[0], self.bandwidth_subset)]
        diffs = jnp.expand_dims(particles, axis=0) - jnp.expand_dims(
            particles, axis=1
        )  # N x N x D
        if particles.ndim == 2:
            diffs = safe_norm(diffs, ord=2, axis=-1)  # N x N x D -> N x N
        diffs = jnp.reshape(diffs, (diffs.shape[0] * diffs.shape[1], -1))  # N * N x 1
        if diffs.ndim == 2:
            diff_norms = safe_norm(diffs, ord=2, axis=-1)
        else:
            diff_norms = diffs
        median = jnp.argsort(diff_norms)[int(diffs.shape[0] / 2)]
        bandwidth = jnp.abs(diffs)[median] ** 2 * factor + 1e-5

        def feature(x, w, b):
            return jnp.sqrt(2) * jnp.cos((x @ w + b) / bandwidth)

        def kernel(x, y):
            ws = (
                self._random_weights
                if self.random_indices is None
                else self._random_weights[self.random_indices]
            )
            bs = (
                self._random_biases
                if self.random_indices is None
                else self._random_biases[self.random_indices]
            )
            return jnp.sum(
                jax.vmap(lambda w, b: feature(x, w, b) * feature(y, w, b))(ws, bs)
            )

        return kernel


class MixtureKernel(SteinKernel):
    """
    Calculates a mixture of multiple kernels
    :math: `k(x,y) = \\sum_i w_ik_i(x,y)`

    ** Reference **
    1. *Stein Variational Gradient Descent as Moment Matching* by Liu and Wang

    :param ws: Weight of each kernel in the mixture
    :param kernel_fns: Different kernel functions to mix together
    """

    def __init__(self, ws: List[float], kernel_fns: List[SteinKernel], mode="norm"):
        assert len(ws) == len(kernel_fns)
        assert len(kernel_fns) > 1
        assert all(kf.mode == mode for kf in kernel_fns)
        self.ws = ws
        self.kernel_fns = kernel_fns

    @property
    def mode(self):
        return self.kernel_fns[0].mode

    def compute(self, particles, particle_info, loss_fn):
        kernels = [
            kf.compute(particles, particle_info, loss_fn) for kf in self.kernel_fns
        ]

        def kernel(x, y):
            res = self.ws[0] * kernels[0](x, y)
            for w, k in zip(self.ws[1:], kernels[1:]):
                res = res + w * k(x, y)
            return res

        return kernel


class HessianPrecondMatrix(PrecondMatrix):
    """
    Calculates the constant precondition matrix based on the negative Hessian of the loss from [1].

    ** References: **
    1. *Stein Variational Gradient Descent with Matrix-Valued Kernels* by Wang, Tang, Bajaj and Liu
    """

    def compute(self, particles, loss_fn):
        hessian = -jax.vmap(jax.hessian(loss_fn))(particles)
        return hessian


class PrecondMatrixKernel(SteinKernel):
    """
    Calculates the const preconditioned kernel
    :math: `k(x,y) = Q^{-\\frac{1}{2}}k(Q^{\\frac{1}{2}}x, Q^{\\frac{1}{2}}y)Q^{-\\frac{1}{2}},`
    or anchor point preconditioned kernel
    :math: `k(x,y) = \\sum_{l=1}^m k_{Q_l}(x,y)w_l(x)w_l(y)`
    both from [1].

    ** References: **
    1. "Stein Variational Gradient Descent with Matrix-Valued Kernels" by Wang, Tang, Bajaj and Liu

    :param precond_matrix_fn: The constant preconditioning matrix
    :param inner_kernel_fn: The inner kernel function
    :param precond_mode: How to use the precondition matrix, either constant ('const')
                         or as mixture with anchor points ('anchor_points')
    """

    def __init__(
        self,
        precond_matrix_fn: PrecondMatrix,
        inner_kernel_fn: SteinKernel,
        precond_mode="anchor_points",
    ):
        assert inner_kernel_fn.mode == "matrix"
        assert precond_mode == "const" or precond_mode == "anchor_points"
        self.precond_matrix_fn = precond_matrix_fn
        self.inner_kernel_fn = inner_kernel_fn
        self.precond_mode = precond_mode

    @property
    def mode(self):
        return "matrix"

    def compute(self, particles, particle_info, loss_fn):
        qs = self.precond_matrix_fn.compute(particles, loss_fn)
        if self.precond_mode == "const":
            qs = jnp.expand_dims(jnp.mean(qs, axis=0), axis=0)
        qs_inv = jnp.linalg.inv(qs)
        qs_sqrt = sqrth(qs)
        qs_inv_sqrt = sqrth(qs_inv)
        inner_kernel = self.inner_kernel_fn.compute(particles, particle_info, loss_fn)

        def kernel(x, y):
            if self.precond_mode == "const":
                wxs = jnp.array([1.0])
                wys = jnp.array([1.0])
            else:
                wxs = jax.nn.softmax(
                    jax.vmap(
                        lambda z, q_inv: dist.MultivariateNormal(
                            z, posdef(q_inv)
                        ).log_prob(x)
                    )(particles, qs_inv)
                )
                wys = jax.nn.softmax(
                    jax.vmap(
                        lambda z, q_inv: dist.MultivariateNormal(
                            z, posdef(q_inv)
                        ).log_prob(y)
                    )(particles, qs_inv)
                )
            return jnp.sum(
                jax.vmap(
                    lambda qs, qis, wx, wy: wx
                    * wy
                    * (qis @ inner_kernel(qs @ x, qs @ y) @ qis.transpose())
                )(qs_sqrt, qs_inv_sqrt, wxs, wys),
                axis=0,
            )

        return kernel


class GraphicalKernel(SteinKernel):
    """
    Calculates graphical kernel
    :math: `k(x,y) = diag({K^(l)(x,y)}_l)
    from [1].

    ** References: **
    1. *Stein Variational Message Passing for Continuous Graphical Models* by Wang, Zheng and Liu

    :param local_kernel_fns: A mapping between parameters and a choice of kernel function for that parameter
                             (default to default_kernel_fn for each parameter)
    :param default_kernel_fn: The default choice of kernel function when none is specified for a particular parameter
    """

    def __init__(
        self,
        mode="matrix",
        local_kernel_fns: Dict[str, SteinKernel] = None,
        default_kernel_fn: SteinKernel = RBFKernel(),
    ):
        assert mode == "matrix"

        self.local_kernel_fns = local_kernel_fns if local_kernel_fns is not None else {}
        self.default_kernel_fn = default_kernel_fn

    @property
    def mode(self):
        return "matrix"

    def compute(self, particles, particle_info, loss_fn):
        def pk_loss_fn(start, end):
            def fn(ps):
                return loss_fn(
                    jnp.concatenate(
                        [particles[:, :start], ps, particles[:, end:]], axis=-1
                    )
                )

            return fn

        local_kernels = []
        for pk, (start_idx, end_idx) in particle_info.items():
            pk_kernel_fn = self.local_kernel_fns.get(pk, self.default_kernel_fn)
            pk_kernel = pk_kernel_fn.compute(
                particles[:, start_idx:end_idx],
                {pk: (0, end_idx - start_idx)},
                pk_loss_fn(start_idx, end_idx),
            )
            local_kernels.append((pk_kernel, pk_kernel_fn.mode, start_idx, end_idx))

        def kernel(x, y):
            kernel_res = []
            for kernel, mode, start_idx, end_idx in local_kernels:
                v = kernel(x[start_idx:end_idx], y[start_idx:end_idx])
                if mode == "norm":
                    v = v * jnp.identity(end_idx - start_idx)
                elif mode == "vector":
                    v = jnp.diag(v)
                kernel_res.append(v)
            return jax.scipy.linalg.block_diag(*kernel_res)

        return kernel
