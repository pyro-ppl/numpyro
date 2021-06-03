from numpyro.contrib.einstein.callbacks.callback import Callback


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
