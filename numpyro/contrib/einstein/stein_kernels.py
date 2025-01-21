# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from jax import grad, numpy as jnp, random, vmap
from jax.lax import stop_gradient
import jax.scipy.linalg
import jax.scipy.stats

from numpyro.contrib.einstein.stein_util import median_bandwidth
from numpyro.distributions import biject_to
from numpyro.infer.autoguide import AutoNormal


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
        rng_key,
        particles: jnp.ndarray,
        particle_info: dict[str, tuple[int, int]],
        loss_fn: Callable[[jnp.ndarray], float],
    ):
        """Computes the kernel function given the input Stein particles

        :param particles: The Stein particles to compute the kernel from
        :param particle_info: A mapping from parameter names to the position in the
            particle matrix
        :param loss_fn: Loss function given particles
        :return: The kernel_fn to compute kernel for pair of particles.
            Modes: norm `(d,) (d,)-> ()`, vector `(d,) (d,) -> (d)`, or matrix
            `(d,) (d,) -> (d,d)`
        """
        raise NotImplementedError

    def init(self, rng_key, particles_shape):
        """
        Initializes the kernel
        :param rng_key: a JAX PRNGKey to initialize the kernel
        :param tuple particles_shape: shape of the input `particles` in :meth:`compute`
        """
        pass


class RBFKernel(SteinKernel):
    """Calculates the Gaussian RBF kernel function used in [1]. The kernel is given by

        :math:`k(x,y) = \\exp(\\frac{-1}{h} \\|x-y\\|^2)`,

    where the bandwidth :math:`h` is computed using the median heuristic

        :math:`h = \\frac{1}{\\log(m)} \\text{med}(\\|x-y\\|)`.

    In the above :math:`m` is the number of particles.

    :param str mode: Either 'norm' (default) specifying to take the norm of each
        particle, 'vector' to return a component-wise kernel or 'matrix' to return a
        matrix-valued kernel
    :param str matrix_mode: Either 'norm_diag' (default) for diagonal filled with the
        norm kernel or 'vector_diag' for diagonal of vector-valued kernel
    :param bandwidth_factor: A multiplier to the bandwidth based on data size n
        (default 1/log(n))

    **References:**

    1. Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm."
        Advances in neural information processing systems 29 (2016).
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

    def compute(self, rng_key, particles, particle_info, loss_fn):
        bandwidth = median_bandwidth(particles, self.bandwidth_factor)

        def kernel(x, y):
            reduce = jnp.sum if self._normed() else lambda x: x
            kernel_res = jnp.exp(-reduce((x - y) ** 2) / stop_gradient(bandwidth))
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
    """Calculates the IMQ kernel from Theorem 8 of [1]. The kernel is given by

        :math:`k(x,y) = (c^2 + \\|x-y\\|^2_2)^{\\beta},`

    where :math:`c\\in \\mathcal\\{R\\}` and :math:`\\beta \\in (-1,0)`.


    :param str mode: Either 'norm' (default) specifying to take the norm
        of each particle, or 'vector' to return a component-wise kernel
    :param float const: Positive multi-quadratic constant (c)
    :param float expon: Inverse exponent (beta) between (-1, 0)

    **References:**

    1. Gorham, Jackson, and Lester Mackey. "Measuring Sample Quality with Kernels."
        International Conference on Machine Learning. PMLR, 2017.
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

    def _normed(self):
        return self._mode == "norm"

    def compute(self, rng_key, particles, particle_info, loss_fn):
        def kernel(x, y):
            reduce = jnp.sum if self._normed() else lambda x: x
            return (self.const**2 + reduce((x - y) ** 2)) ** self.expon

        return kernel


class LinearKernel(SteinKernel):
    """
    Calculates the linear kernel from Theorem 3.3 in [1]. The kernel is given by

        :math:`k(x,y) = x^T y + 1`.

    **References:**

    1. Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent as Moment Matching."
        Advances in Neural Information Processing Systems 31 (2018).
    """

    def __init__(self, mode="norm"):
        assert mode == "norm"
        self._mode = "norm"

    @property
    def mode(self):
        return self._mode

    def compute(self, rng_key, particles: jnp.ndarray, particle_info, loss_fn):
        def kernel(x, y):
            if x.ndim == 1:
                return x @ y + 1
            else:
                return x * y + 1

        return kernel


class RandomFeatureKernel(SteinKernel):
    """Calculates the Gaussian variate of random kernel in eq. 5 and 6 of [1]. The kernel is given by

        :math:`k(x,y)= \\frac{1}{m}\\sum_{l=1}^{m}\\phi(x,w_l)\\phi(y,w_l)`,

    where :math:`\\phi(\\cdot, w)` are the Gaussian random feature maps in eq. 6. The maps are given by

        :math:`\\phi(z, w) = \\sqrt{2}\\left(h^{-1}w_1^Tz + w_0\\right)`,

    where :math:`h` is the bandwidth computed using the median trick from :class:`~numpyro.constrib.einstein.RBFKernel`,
    :math:`w_0\\sim\\text{Uni}([0,2\\pi])` and :math:`w_1\\sim\\mathcal{N}(0,I)`.

    :param bandwidth_subset: How many particles should be used to calculate the bandwidth?
                             (default None, meaning all particles)
    :param random_indices: The set of indices which to do random feature expansion on.
                           (default None, meaning all indices)
    :param bandwidth_factor: A multiplier to the bandwidth based on data size n (default 1/log(n))

    **References:**

    1. Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent as Moment Matching."
        Advances in Neural Information Processing Systems 31 (2018).
    """

    def __init__(
        self,
        mode="norm",
        bandwidth_subset=None,
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
        self._bandwidth_subset_indices = None

    @property
    def mode(self):
        return self._mode

    def init(self, rng_key, particles_shape):
        rng_key, rng_weight, rng_bias = random.split(rng_key, 3)
        self._random_weights = random.normal(rng_weight, shape=particles_shape)
        self._random_biases = random.uniform(
            rng_bias, shape=particles_shape, maxval=(2 * np.pi)
        )
        if self.bandwidth_subset is not None:
            self._bandwidth_subset_indices = random.choice(
                rng_key, particles_shape[0], (self.bandwidth_subset,)
            )

    def compute(self, rng_key, particles, particle_info, loss_fn):
        if self._random_weights is None:
            raise RuntimeError(
                "The `.init` method should be called first to initialize the"
                " random weights, biases and subset indices."
            )
        if particles.shape != self._random_weights.shape:
            raise ValueError(
                "Shapes of `particles` and the random weights are mismatched, got {}"
                " and {}.".format(particles.shape, self._random_weights.shape)
            )
        if self.bandwidth_subset is not None:
            particles = particles[self._bandwidth_subset_indices]

        bandwidth = median_bandwidth(particles, self.bandwidth_factor)

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
    """Calculates a mixture of multiple kernels from eq. 1 of [1]. The kernel is given by

        :math:`k(x,y) = \\sum_i w_ik_i(x,y)`,

    where :math:`k_i` is a reproducing kernel and :math:`w_i \\in (0,\\infty)`.

    :param ws: Weight of each kernel in the mixture.
    :param kernel_fns: Different kernel functions to mix together.

    **References:**

    1. Ai, Qingzhong, et al. "Stein variational gradient descent with multiple kernels."
        Cognitive Computation 15.2 (2023): 672-682.
    """

    def __init__(self, ws: list[float], kernel_fns: list[SteinKernel], mode="norm"):
        assert len(ws) == len(kernel_fns)
        assert len(kernel_fns) > 1
        assert all(kf.mode == mode for kf in kernel_fns)
        self.ws = ws
        self.kernel_fns = kernel_fns

    @property
    def mode(self):
        return self.kernel_fns[0].mode

    def compute(self, rng_key, particles, particle_info, loss_fn):
        kernels = [
            kf.compute(rng_key, particles, particle_info, loss_fn)
            for kf in self.kernel_fns
        ]

        def kernel(x, y):
            res = self.ws[0] * kernels[0](x, y)
            for w, k in zip(self.ws[1:], kernels[1:]):
                res = res + w * k(x, y)
            return res

        return kernel

    def init(self, rng_key, particles_shape):
        for kf in self.kernel_fns:
            rng_key, krng_key = random.split(rng_key)
            kf.init(krng_key, particles_shape)


class GraphicalKernel(SteinKernel):
    """Calculates the graphical kernel, also called the coordinate-wise kernel, from Theorem 1 in [1].
    The kernel is given by

        :math:`k(x,y) = diag({k_l(x_l,y_l)})`,

    for coordinate-wise kernels :math:`k_l`.

    :param local_kernel_fns: A mapping between parameters and a choice of kernel
        function for that parameter (default to default_kernel_fn for each parameter)
    :param default_kernel_fn: The default choice of kernel function when none is
        specified for a particular parameter

    **References:**

    1. Wang, Dilin, Zhe Zeng, and Qiang Liu. "Stein variational message passing for continuous graphical models."
        International Conference on Machine Learning. PMLR, 2018.
    """

    def __init__(
        self,
        mode="matrix",
        local_kernel_fns: dict[str, SteinKernel] = None,
        default_kernel_fn: SteinKernel = RBFKernel(),
    ):
        assert mode == "matrix"

        self.local_kernel_fns = local_kernel_fns if local_kernel_fns is not None else {}
        self.default_kernel_fn = default_kernel_fn

    @property
    def mode(self):
        return "matrix"

    def compute(self, rng_key, particles, particle_info, loss_fn):
        def pk_loss_fn(start, end):
            def fn(ps):
                return loss_fn(
                    jnp.concatenate(
                        [particles[:, :start], ps, particles[:, end:]], axis=-1
                    )
                )

            return fn

        local_kernels = []
        keys = random.split(rng_key, len(particle_info))
        for key, (pk, (start_idx, end_idx)) in zip(keys, particle_info.items()):
            pk_kernel_fn = self.local_kernel_fns.get(pk, self.default_kernel_fn)
            pk_kernel = pk_kernel_fn.compute(
                key,
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


class ProbabilityProductKernel(SteinKernel):
    """**EXPERIMENTAL** Compute the unormalized probability product kernel for Gaussians given by eq. 5 in [1].

    **References**:

    1. Jebara, Tony, Risi Kondor, and Andrew Howard. "Probability product kernels."
        The Journal of Machine Learning Research 5 (2004): 819-844.
    """

    def __init__(self, guide, scale=1.0, mode="norm"):
        assert mode == "norm"
        self._mode = "norm"
        self.guide = guide
        self.scale = scale
        assert isinstance(guide, AutoNormal), "PPK only implemented for AutoNormal"

    def compute(
        self,
        rng_key,
        particles: jnp.ndarray,
        particle_info: dict[str, tuple[int, int]],
        loss_fn: Callable[[jnp.ndarray], float],
    ):
        loc_idx = jnp.concatenate(
            [
                jnp.arange(*idx)
                for name, idx in particle_info.items()
                if name.endswith(f"{self.guide.prefix}_loc")
            ]
        )
        scale_idx = jnp.concatenate(
            [
                jnp.arange(*idx)
                for name, idx in particle_info.items()
                if name.endswith(f"{self.guide.prefix}_scale")
            ]
        )

        def kernel(x, y):
            biject = biject_to(self.guide.scale_constraint)
            x_loc = x[loc_idx]
            x_scale = biject(x[scale_idx])
            x_quad = ((x_loc / x_scale) ** 2).sum()

            y_loc = y[loc_idx]
            y_scale = biject(y[scale_idx])
            y_quad = ((y_loc / y_scale) ** 2).sum()

            cross_loc = x_loc * x_scale**-2 + y_loc * y_scale**-2
            cross_var = 1 / (y_scale**-2 + x_scale**-2)
            cross_quad = (cross_loc**2 * cross_var).sum()

            quad = jnp.exp(-self.scale / 2 * (x_quad + y_quad - cross_quad))

            return quad

        return kernel

    @property
    def mode(self):
        return self._mode


class RadialGaussNewtonKernel(SteinKernel):
    r"""The Radial Gauss-Newton Kernel [1,2], also called the scaled Hessian kernel, is a scalar kernel defined as

        .. math::
            k(x,y) = \exp\left(-\frac{1}{2d}(x-y)^T M (x-y)\right),

    where :math:`x,y \in R^d` are particles and :math:`M` is a metric matrix. :math:`M` approximates the expected
    curvature of the log posterior using Hessian approximations :math:`A(x)`.

    The matrix :math:`M` is computed using :math:`m` particles as follows [2, Eq. 19, p.5]:

        .. math::
            M= \frac{1}{m} \sum_{i=1}^m A(x_i)

    with the Hessian approximation given by:

        .. math::
            A(x) = J(x) J(x)^T,

    where :math:`J(x)` is the Jacobian of an ELBO at :math:`x`.

    **References**:

    1. Maken, Fahira Afzal, Fabio Ramos, and Lionel Ott. "Stein Particle Filter for Nonlinear,
        **Non-Gaussian State Estimation."** IEEE Robotics and Automation Letters 7.2 (2022).
    2. Detommaso, Gianluca, et al. "A Stein variational Newton method."
        Advances in Neural Information Processing Systems 31 (2018).
    """

    def __init__(self):
        self._mode = "norm"

    def compute(self, rng_key, particles, particle_info, loss_fn):
        n, d = particles.shape

        Jx = vmap(grad(loss_fn, argnums=1))(
            random.split(rng_key, n), particles, jnp.arange(n)
        )

        def kernel(x, y):
            dist = jnp.dot(stop_gradient(Jx), (x - y)) ** 2

            kernel_res = jnp.exp(-dist.mean() / (2 * d))
            return kernel_res

        return kernel

    @property
    def mode(self):
        return self._mode
