import numpy as np
from numpyro.callbacks.callback import Callback


class TerminateOnNaN(Callback):
    def __init__(self, check_params=False):
        super().__init__()
        self.check_params = check_params

    def on_train_step_end(self, step, train_info):
        if np.any(np.isnan(train_info['loss'])):
            raise StopIteration
        if self.check_params:
            for param_val in self.vi.get_params(train_info['state']).values():
                if np.any(np.isnan(param_val)):
                    raise StopIteration
