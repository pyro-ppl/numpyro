class BlockNeuralAutoregressiveTransform(Transform):
    event_dim = 1

    def __init__(self, bna_nn, params):
        self.bna_nn = bna_nn
        self.params = params

    def __call__(self, x):
        self._cached_x = x
        y, self._cached_logdet = self.bna_nn(self.params, x)
        return y

    def inv(self, y, caching=True):
        if caching:
            return self._cached_x
        else:
            raise ValueError("Block neural autoregressive transform does not have an analytic"
                             " inverse implemented.")

    def log_abs_det_jacobian(self, x, y, caching=True):
        if caching:
            return self._cached_logdet
        else:
            _, logdet = self.bna_nn(self.params, x)
            return logdet
