from jax import ops, random
from jax.nn.initializer import glorot_uniform, normal, uniform

from numpyro.distributions.util import softplus


def BlockMaskedDense(num_blocks, in_factor, out_factor, bias=True, W_init=glorot_uniform()):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.

    :param int num_blocks: Number of block matrices.
    :param int in_factor: number of rows in each block.
    :param int out_factor: number of columns in each block.
    :param W_init: initialization method for the weights.
    :return: an (`init_fn`, `update_fn`) pair.
    """
    input_dim, out_dim = num_blocks * in_factor, num_blocks * out_factor
    # construct mask_d, mask_o for formula (8) of Ref [1]
    # Diagonal block mask
    mask_d = np.identity(num_block)[..., None]
    mask_d = np.tile(mask_d, (1, in_factor, out_factor)).reshape(input_dim, out_dim)
    # Off-diagonal block mask for lower triangular weight matrix
    mask_o = vec_to_tril_matrix(np.ones(dim * (dim - 1) // 2), diagonal=-1)[..., None]
    mask_o = np.tile(mask_o, (1, in_factor, out_factor)).reshape(input_dim, out_dim)

    def init_fun(rng, input_shape):
        assert input_dim == input_shape[-1]
        *k1, k2, k3 = random.split(rng, num_blocks + 2)

        # Initialize each column block using W_init
        W = np.zeros((input_dim, out_dim)
        for i in range(num_blocks):
            W = ops.index_add(
                W,
                ops.index[:(i + 1) * in_factor, i * out_factor:(i + 1) * out_factor],
                W_init(k1[i], ((i + 1) * in_factor, out_factor))
            )

        # initialize weight scale
        ws = np.log(uniform(1.)(k2, (out_dim,)))

        if bias:
            b = (uniform(1.)(k3, mask.shape[-1:]) - 0.5) * (2 / np.sqrt(out_dim))
            params = (W, ws, b)
        else:
            params = (W, ws)
        return input_shape[:-1] + (outdim,), params

    def apply_fun(params, inputs, **kwargs):
        x, logdet = inputs
        if bias:
            W, ws, b = params
        else:
            W, ws = params

        # Form block weight matrix, making sure it's positive on diagonal!
        w = np.exp(W) * mask_d + W * mask_o

        # Compute norm of each column (i.e. each output features)
        w_norm = np.linalg.norm(w ** 2, axis=-2, keepdims=True)

        # Normalize weight and rescale
        w = np.exp(ws) * w / w_norm

        out = np.dot(x, w)
        if bias:
            out = out + b

        dense_logdet = ws + W - np.log(w_norm)
        # logdet of block diagonal
        dense_logdet = logdet[mask_d.astype(bool)].reshape(dim, in_factor, out_factor)
        logdet = dense_logdet if logdet is None else logmatmulexp(logdet, dense_logdet)
        return out, logdet

    return init_fun, apply_fun


def Tanh():
    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        x, logdet = inputs
        out = np.tanh(x)
        tanh_logdet = -2 * (s + softplus(-2 * s)- np.log(2.))
        # logdet.shape = batch_shape + (num_blocks, in_factor, out_factor)
        # tanh_logdet.shape = batch_shape + (num_blocks x out_factor,)
        # so we need to reshape tanh_logdet to: batch_shape + (num_blocks, 1, out_factor)
        tanh_logdet = tanh_logdet.reshape(logdet.shape[:-2] + (1, logdet.shape[-1]))
        return out, logdet + tanh_logdet


def FanInNormal():
    """
    Similar to stax.FanInSum but also keeps track of log determinant of Jacobian.
    """
    # x + f(x)
    def init_fun(rng, input_shape):
        pass

    def apply_fun(params, inputs, **kwargs):
        pass


def FanInGated(gate_init=normal(1.)):
    """
    Similar to FanInNormal uses a learnable parameter `gate` to interpolate two fan-in branches.
    """
    # a * x + (1 - a) * f(x)
    def init_fun(rng, input_shape):
        pass

    def apply_fun(params, inputs, **kwargs):
        pass


def BlockNeuralAutoregressiveNN(input_dim, hidden_factors, residual=None):
    layers = []
    in_factor = 1
    for i, hidden_factor in enumerate(hidden_factors):
        layers.append(BlockMaskedDense(input_dim, in_factor, hidden_factor))
        layers.append(Tanh())
        in_factor = hidden_factor
    layers.append(BlockMaskedDense(input_dim, in_factor, 1))
    arn = stax.serial(*layers)
    if residual is None
        return arn
    else:
        return stax.serial(stax.FanOut(2), stax.parallel(stax.Identity, arn), residual())
