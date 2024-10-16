# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import lax
import jax.numpy as jnp

from numpyro.distributions.constraints import real_vector
from numpyro.distributions.transforms import Transform
from numpyro.util import fori_loop


def _clamp_preserve_gradients(x, min, max):
    return x + lax.stop_gradient(jnp.clip(x, min, max) - x)


# adapted from https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/transforms/iaf.py
class InverseAutoregressiveTransform(Transform):
    """
    An implementation of Inverse Autoregressive Flow, using Eq (10) from Kingma et al., 2016,

        :math:`\\mathbf{y} = \\mu_t + \\sigma_t\\odot\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, :math:`\\mu_t,\\sigma_t`
    are calculated from an autoregressive network on :math:`\\mathbf{x}`, and :math:`\\sigma_t>0`.

    **References**

    1. *Improving Variational Inference with Inverse Autoregressive Flow* [arXiv:1606.04934],
       Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling
    """

    domain = real_vector
    codomain = real_vector

    def __init__(
        self, autoregressive_nn, log_scale_min_clip=-5.0, log_scale_max_clip=3.0
    ):
        """
        :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
            mean and log scale as a tuple
        """
        self.arn = autoregressive_nn
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def __call__(self, x):
        """
        :param numpy.ndarray x: the input into the transform
        """
        return self.call_with_intermediates(x)[0]

    def call_with_intermediates(self, x):
        mean, log_scale = self.arn(x)
        log_scale = _clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        scale = jnp.exp(log_scale)
        return scale * x + mean, log_scale

    def _inverse(self, y):
        """
        :param numpy.ndarray y: the output of the transform to be inverted
        """

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        def _update_x(i, x):
            mean, log_scale = self.arn(x)
            inverse_scale = jnp.exp(
                -_clamp_preserve_gradients(
                    log_scale, min=self.log_scale_min_clip, max=self.log_scale_max_clip
                )
            )
            x = (y - mean) * inverse_scale
            return x

        x = fori_loop(0, y.shape[-1], _update_x, jnp.zeros(y.shape))
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """
        Calculates the elementwise determinant of the log jacobian.

        :param numpy.ndarray x: the input to the transform
        :param numpy.ndarray y: the output of the transform
        """
        if intermediates is None:
            log_scale = self.arn(x)[1]
            log_scale = _clamp_preserve_gradients(
                log_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )
            return log_scale.sum(-1)
        else:
            log_scale = intermediates
            return log_scale.sum(-1)

    def tree_flatten(self):
        return (self.log_scale_min_clip, self.log_scale_max_clip), (
            ("log_scale_min_clip", "log_scale_max_clip"),
            {"arn": self.arn},
        )

    def __eq__(self, other):
        if not isinstance(other, InverseAutoregressiveTransform):
            return False
        return (
            (self.arn is other.arn)
            & jnp.array_equal(self.log_scale_min_clip, other.log_scale_min_clip)
            & jnp.array_equal(self.log_scale_max_clip, other.log_scale_max_clip)
        )


class BlockNeuralAutoregressiveTransform(Transform):
    """
    An implementation of Block Neural Autoregressive flow.

    **References**

    1. *Block Neural Autoregressive Flow*,
       Nicola De Cao, Ivan Titov, Wilker Aziz
    """

    domain = real_vector
    codomain = real_vector

    def __init__(self, bn_arn):
        self.bn_arn = bn_arn

    def __call__(self, x):
        """
        :param numpy.ndarray x: the input into the transform
        """
        return self.call_with_intermediates(x)[0]

    def call_with_intermediates(self, x):
        y, logdet = self.bn_arn(x)
        return y, logdet

    def _inverse(self, y):
        raise NotImplementedError(
            "Block neural autoregressive transform does not have an analytic"
            " inverse implemented."
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """
        Calculates the elementwise determinant of the log jacobian.

        :param numpy.ndarray x: the input to the transform
        :param numpy.ndarray y: the output of the transform
        """
        if intermediates is None:
            logdet = self.bn_arn(x)[1]
            return logdet.sum(-1)
        else:
            logdet = intermediates
            return logdet.sum(-1)

    def tree_flatten(self):
        return (), ((), {"bn_arn": self.bn_arn})

    def __eq__(self, other):
        return (
            isinstance(other, BlockNeuralAutoregressiveTransform)
            and self.bn_arn is other.bn_arn
        )
