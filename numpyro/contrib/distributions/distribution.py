class Distribution(object):
    arg_constraints = {}
    support = None
    is_reparametrized = False
    _validate_args = False

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is None:
            self._validate_args = validate_args
        super(Distribution, self).__init__()

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    def sample(self, key, size=()):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError
