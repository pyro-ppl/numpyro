from jax import lax, ops
import jax.numpy as np

from numpyro.distributions.constraints import Transform, real_vector


def _clamp_preserve_gradients(x, min, max):
    return x + lax.stop_gradient(np.clip(x, a_min=min, a_max=max) - x)


# adapted from https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/transforms/iaf.py
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
    domain = real_vector
    codomain = real_vector
    event_dim = 1

    def __init__(self, autoregressive_nn, params, caching=False,
                 log_scale_min_clip=-5., log_scale_max_clip=3.):
        """
        :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
            mean and log scale as a tuple
        :param params: the parameters for the autoregressive neural network
        :type list:
        :param bool caching: whether to cache results during forward pass
        """
        self.arn = autoregressive_nn
        self.params = params
        self.caching = caching
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def __call__(self, x):
        """
        :param x: the input into the transform
        :type x: numpy array
        """
        out = self.arn.apply_fun(self.params, x)
        mean, log_scale = out[..., 0, :], out[..., 1, :]
        log_scale = _clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        scale = np.exp(log_scale)
        if self.caching:
            self._cached_x = x
            self._cached_log_scale = log_scale
        return scale * x + mean

    def inv(self, y):
        """
        :param y: the output of the transform to be inverted
        :type y: numpy array
        """
        if self.caching:
            return self._cached_x

        x = np.zeros(y.shape)

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        def _update_x(i, x):
            idx = self.arn.permutation[i]
            out = self.arn.apply_fun(self.params, x)
            mean, log_scale = out[..., 0, :], out[..., 1, :]
            inverse_scale = np.exp(-_clamp_preserve_gradients(
                log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            x = ops.index_update(x, (..., idx), (y[..., idx] - mean) * inverse_scale)
            return x

        x = lax.fori_loop(0, y.shape[-1], _update_x, x)

        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        :param x: the input to the transform
        :type x: numpy array
        :param y: the output of the transform
        :type y: numpy array
        """
        if self.caching:
            log_scale = self._cached_log_scale
        else:
            log_scale = self.arn.apply_fun(self.params, x)[..., 1, :]  # TODO: this shouldn't be recomputed here
            log_scale = _clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)
