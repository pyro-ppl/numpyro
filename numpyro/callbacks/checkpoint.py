import pickle
from datetime import datetime

from numpyro.callbacks.callback import Callback


class Checkpoint(Callback):
    STRF_TIME = "%Y%m%d_%H%M%S"

    def __init__(self, file_path: str,
                 loss_mode='training',
                 save_mode='best',
                 frequency='epoch'):
        assert loss_mode in {'training', 'validation'}
        assert frequency in {'step', 'epoch'}
        assert save_mode in {'best', 'all'}
        super().__init__()
        self.loss_mode = loss_mode
        self.frequency = frequency
        self.save_mode = save_mode
        self.file_path = file_path
        self.best_loss = float('inf')
        self.time = datetime.utcnow().strftime(Checkpoint.STRF_TIME)

    def on_validation_end(self, val_step, val_info):
        if self.loss_mode == 'validation':
            self._checkpoint('val', val_step, val_info['loss'], val_info['state'])

    def on_train_epoch_end(self, epoch, train_info):
        if self.loss_mode == 'training' and self.frequency == 'epoch':
            self._checkpoint('epoch', epoch, train_info['loss'], train_info['state'])

    def on_train_step_end(self, step, train_info):
        if self.loss_mode == 'training' and self.frequency == 'step':
            self._checkpoint('step', step, train_info['loss'], train_info['state'])

    def _checkpoint(self, prefix, number, loss, state):
        if self.save_mode == 'best' and self.best_loss < loss:
            return
        self.best_loss = loss
        if self.save_mode != 'best' and '{num}' in self.file_path:
            file_path = self.file_path.replace('{num}', "{}_{}".format(prefix, number))
        else:
            file_path = self.file_path.replace('{num}', "{}_{}".format(prefix, 'best'))
        if '{time}' in self.file_path:
            file_path = file_path.replace('{time}', self.time)
        with open(file_path, 'w') as fp:
            pickle.dump(state, fp)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as fp:
            return pickle.load(fp)
