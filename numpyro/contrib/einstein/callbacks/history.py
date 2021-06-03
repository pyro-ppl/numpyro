from numpyro.contrib.einstein.callbacks import Callback


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
