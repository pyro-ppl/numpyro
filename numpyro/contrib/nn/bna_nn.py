from numpyro.distributions.util import softplus


def MaskedBlockDense():
    def init_fun(rng, input_shape):
        return output_shape, params

    def apply_fun(params, inputs, **kwargs):
        x, logdet = inputs
        # TODO: implement
        return o, logdet


def Tanh():
    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        x, logdet = inputs
        y = np.tanh(x)
        tanh_logdet = - 2 * (x - np.log(2.) + softplus(-2 * x))
        return y, logdet + tanh_logdet  # TODO: reshape?


def stax_serial_with_jacobian(*layers):
    """
    Works like `stax.serial` but also forward the log determinant of Jacobian.
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
          rng, layer_rng = random.split(rng)
          input_shape, param = init_fun(layer_rng, input_shape)
          params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for fun, param, rng in zip(apply_funs, params, rngs):
          inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    return init_fun, apply_fun


def BlockNeuralAutoregressiveNN(input_dim, hidden_factor):
    # TODO: support multi hidden factors
    layers = [MaskedBlockDense(hidden_factor), Tanh(), MaskedBlockDense(input_dim)]
    return stax_serial_with_jacobian(*layers)
