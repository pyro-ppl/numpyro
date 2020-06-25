from typing import Optional

from tqdm import trange, tqdm

from numpyro.callbacks import Callback


class Progbar(Callback):
    def __init__(self):
        super().__init__()
        self.progbar: Optional[tqdm] = None

    def on_train_begin(self, train_info):
        self.progbar = trange(train_info['num_steps'])

    def on_train_step_end(self, step, train_info):
        if self.progbar is not None:
            self.progbar.set_description('{} {:.5}'.format(self.vi.name, train_info['loss']), refresh=False)
            self.progbar.update()

    def on_train_end(self, train_info):
        if self.progbar is not None:
            self.progbar.close()
            self.progbar = None
