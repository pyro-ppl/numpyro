import os
import pickle
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from jax.experimental.optimizers import pack_optimizer_state, unpack_optimizer_state
from tqdm import tqdm, trange

__all__ = [
    "Checkpoint",
    "EarlyStopping",
    "History",
    "Progbar",
    "ReduceLROnPlateau",
    "TerminateOnNaN"
]


class Callback(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is Callback:
            raise TypeError("Callback is not directly instantiable")
        return super().__new__(cls)

    def __init__(self, vi=None):
        self.vi = vi

    def on_train_begin(self, train_info):
        pass

    def on_train_end(self, train_info):
        pass

    def on_train_step_begin(self, step, train_info):
        pass

    def on_train_step_end(self, step, train_info):
        pass

    def on_train_epoch_begin(self, epoch, train_info):
        pass

    def on_train_epoch_end(self, epoch, train_info):
        pass

    def on_validation_begin(self, val_step, val_info):
        pass

    def on_validation_end(self, val_step, val_info):
        pass


class Checkpoint(Callback):
    STRF_TIME = "%Y%m%d_%H%M%S"

    def __init__(
            self, file_path: str, loss_mode="training", save_mode="best", frequency="epoch"
    ):
        assert loss_mode in {"training", "validation"}
        assert frequency in {"step", "epoch"}
        assert save_mode in {"best", "all"}
        super().__init__()
        self.loss_mode = loss_mode
        self.frequency = frequency
        self.save_mode = save_mode
        self.file_path = file_path
        self.best_loss = float("inf")
        self.time = datetime.utcnow().strftime(Checkpoint.STRF_TIME)

    def on_validation_end(self, val_step, val_info):
        if self.loss_mode == "validation":
            self._checkpoint("val", val_step, val_info["loss"], val_info["state"])

    def on_train_epoch_end(self, epoch, train_info):
        if self.loss_mode == "training" and self.frequency == "epoch":
            self._checkpoint("epoch", epoch, train_info["loss"], train_info["state"])

    def on_train_step_end(self, step, train_info):
        if self.loss_mode == "training" and self.frequency == "step":
            self._checkpoint("step", step, train_info["loss"], train_info["state"])

    def _checkpoint(self, prefix, number, loss, state):
        if self.save_mode == "best" and self.best_loss < loss:
            return
        self.best_loss = loss
        if self.save_mode != "best" and "{num}" in self.file_path:
            file_path = self.file_path.replace("{num}", "{}_{}".format(prefix, number))
        else:
            file_path = self.file_path.replace("{num}", "{}_{}".format(prefix, "best"))
        if "{time}" in self.file_path:
            file_path = file_path.replace("{time}", self.time)
        step, opt_state = state.optim_state

        with open(file_path, "wb") as fp:
            pickle.dump(
                (step, unpack_optimizer_state(opt_state), state.rng_key, loss), fp
            )

    def latest(self):
        path = Path(self.file_path)
        restore_files = path.parent.glob("*" + "".join(path.suffixes))
        try:
            return max(restore_files, key=os.path.getctime)
        except ValueError:
            return None

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as fp:
            step, opt_state, rng_key, loss = pickle.load(fp)
        return (step, pack_optimizer_state(opt_state)), rng_key, loss


class EarlyStopping(Callback):
    def __init__(
            self,
            patience=10,
            min_delta=0.0,
            smoothing="dexp",
            data_smoothing_factor=0.6,
            trend_smoothing_factor=0.6,
            loss_mode="training",
    ):
        super().__init__()
        assert smoothing in {"none", "exp", "dexp"}
        assert loss_mode in {"training", "validation"}
        self.patience = patience
        self.smoothing = smoothing
        self.min_delta = min_delta
        self.waiting = 0
        self.best_loss = float("inf")
        self.prev_loss = None
        self.curr_loss = None
        self.trend = None
        self.data_smoothing_factor = data_smoothing_factor
        self.trend_smoothing_factor = trend_smoothing_factor
        self.loss_mode = loss_mode

    def on_train_begin(self, train_info):
        if self.loss_mode == "training":
            self.best_loss = train_info["loss"]
            self.curr_loss = train_info["loss"]

    def on_train_step_end(self, step, train_info):
        if self.loss_mode == "training":
            self.update_and_early_stop(train_info["loss"])

    def on_validation_end(self, val_step, val_info):
        if self.loss_mode == "validation":
            self.update_and_early_stop(val_info["loss"])

    def update_and_early_stop(self, loss):
        if self.smoothing == "dexp":
            self.prev_loss = self.curr_loss
            self.curr_loss = self.data_smoothing_factor * loss + (
                    1 - self.data_smoothing_factor
            ) * (self.curr_loss + self.trend)
            if self.trend is None:
                self.trend = loss - self.curr_loss
            else:
                self.trend = (
                        self.trend_smoothing_factor * (self.curr_loss - self.prev_loss)
                        + (1 - self.trend_smoothing_factor) * self.trend
                )
        elif self.smoothing == "exp":
            self.curr_loss = (
                    self.data_smoothing_factor * loss
                    + (1 - self.data_smoothing_factor) * self.curr_loss
            )
        else:
            self.curr_loss = loss
        if self.curr_loss - self.best_loss < self.min_delta:
            self.best_loss = self.curr_loss
            self.waiting = 0
        elif self.waiting >= self.patience:
            raise StopIteration
        else:
            self.waiting += 1


class History(Callback):
    def __init__(self):
        super().__init__()
        self.training_history = []
        self.validation_history = []

    def on_train_begin(self, train_info):
        self.training_history.append(train_info["loss"])

    def on_train_step_end(self, step, train_info):
        self.training_history.append(train_info["loss"])

    def on_validation_end(self, val_step, val_info):
        self.validation_history.append(val_info["loss"])


class Progbar(Callback):
    def __init__(self):
        super().__init__()
        self.progbar: Optional[tqdm] = None

    def on_train_begin(self, train_info):
        self.progbar = trange(train_info["num_steps"])

    def on_train_step_end(self, step, train_info):
        if self.progbar is not None:
            self.progbar.set_description(
                "{} {:.5}".format(self.vi.name, train_info["loss"]), refresh=False
            )
            self.progbar.update()

    def on_train_end(self, train_info):
        if self.progbar is not None:
            self.progbar.close()
            self.progbar = None


class ReduceLROnPlateau(Callback):
    def __init__(
            self,
            initial_lr=1e-3,
            factor=0.1,
            patience=10,
            min_delta=1e-3,
            min_lr=1e-5,
            loss_mode="training",
            frequency="epoch",
    ):
        assert loss_mode in {"training", "validation"}
        assert frequency in {"step", "epoch"}
        super().__init__()
        self.best_loss = float("inf")
        self.waiting = 0
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.loss_mode = loss_mode
        self.frequency = frequency

    def __call__(self, epoch):
        return self.lr

    def on_validation_end(self, val_step, val_info):
        if self.loss_mode == "validation":
            self._reduce_lr_on_plateau(val_info["loss"])

    def on_train_step_end(self, step, train_info):
        if self.loss_mode == "train" and self.frequency == "step":
            self._reduce_lr_on_plateau(train_info["loss"])

    def on_train_epoch_end(self, epoch, train_info):
        if self.loss_mode == "train" and self.frequency == "epoch":
            self._reduce_lr_on_plateau(train_info["loss"])

    def _reduce_lr_on_plateau(self, loss):
        if loss - self.best_loss < self.min_delta:
            self.best_loss = loss
            self.waiting = 0
        elif self.waiting >= self.patience:
            self.lr = max(self.min_lr, self.factor * self.lr)
            self.waiting = 0
        else:
            self.waiting += 1


class TerminateOnNaN(Callback):
    def __init__(self, check_params=False):
        super().__init__()
        self.check_params = check_params

    def on_train_step_end(self, step, train_info):
        if np.any(np.isnan(train_info["loss"])):
            raise StopIteration
        if self.check_params:
            for param_val in self.vi.get_params(train_info["state"]).values():
                if np.any(np.isnan(param_val)):
                    raise StopIteration
