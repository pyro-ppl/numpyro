from abc import ABC


class Callback(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is Callback:
            raise TypeError("Callback is not directly instantiable")
        return super().__new__(cls)

    def __init__(self, vi: "VI" = None):
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
