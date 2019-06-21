# adapted from https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/transforms/iaf.py

from __future__ import absolute_import, division, print_function

from numpyro.distributions.constraints import Transform
from jax.lax import stop_gradient
import jax.numpy as np


def _clamp_preserve_gradients(x, min, max):
    return x + stop_gradient(np.clip(x, a_min=min, a_max=max) - x)


class InverseAutoregressiveTransform(Transform):
    """
    An implementation of Inverse Autoregressive Flow, using Eq (10) from Kingma et al., 2016,

        :math:`\\mathbf{y} = \\mu_t + \\sigma_t\\odot\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, :math:`\\mu_t,\\sigma_t`
    are calculated from an autoregressive network on :math:`\\mathbf{x}`, and :math:`\\sigma_t>0`.

    References

    1. Improving Variational Inference with Inverse Autoregressive Flow [arXiv:1606.04934]
    Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling
    """
    def __init__(self, autoregressive_nn, params, log_scale_min_clip=-5., log_scale_max_clip=3.):
        """
        :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
            mean and log scale as a tuple
        :param params: the parameters for the autoregressive neural network
        :type list:
        """
        self.arn = autoregressive_nn
        self.params = params
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

        # TODO event_dim logic?

    def __call__(self, x):
        """
        :param x: the input into the transform
        :type x: numpy array
        """
        out = self.arn.apply_fun(self.params, x)
        mean, log_scale = out[..., 0, :], out[..., 1, :]
        log_scale = _clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        scale = np.exp(log_scale)
        return scale * x + mean

    def inv(self, y):
        """
        :param y: the output of the transform to be inverted
        :type y: numpy array
        """
        x = [np.zeros(y.shape[:-1])] * y.shape[-1]

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for idx in self.arn.permutation:
            out = self.arn.apply_fun(self.params, np.stack(x, axis=-1))
            mean, log_scale = out[..., 0, :], out[..., 1, :]
            inverse_scale = np.exp(-_clamp_preserve_gradients(
                log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            x[idx] = (y[..., idx] - mean) * inverse_scale

        x = np.stack(x, axis=-1)
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        :param x: the input to the transform
        :type x: numpy array
        :param y: the output of the transform
        :type y: numpy array
        """
        log_scale = self.arn.apply_fun(self.params, x)[..., 1, :]  # TODO: this shouldn't be recomputed here
        log_scale = _clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)
