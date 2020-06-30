from numpyro.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(self, patience=10, min_delta=0.0, smoothing='dexp',
                 data_smoothing_factor=0.6, trend_smoothing_factor=0.6,
                 loss_mode='training'):
        super().__init__()
        assert smoothing in {'none', 'exp', 'dexp'}
        assert loss_mode in {'training', 'validation'}
        self.patience = patience
        self.smoothing = smoothing
        self.min_delta = min_delta
        self.waiting = 0
        self.best_loss = float('inf')
        self.prev_loss = None
        self.curr_loss = None
        self.trend = None
        self.data_smoothing_factor = data_smoothing_factor
        self.trend_smoothing_factor = trend_smoothing_factor
        self.loss_mode = loss_mode

    def on_train_begin(self, train_info):
        if self.loss_mode == 'training':
            self.best_loss = train_info['loss']
            self.curr_loss = train_info['loss']

    def on_train_step_end(self, step, train_info):
        if self.loss_mode == 'training':
            self.update_and_early_stop(train_info['loss'])

    def on_validation_end(self, val_step, val_info):
        if self.loss_mode == 'validation':
            self.update_and_early_stop(val_info['loss'])

    def update_and_early_stop(self, loss):
        if self.smoothing == 'dexp':
            self.prev_loss = self.curr_loss
            self.curr_loss = (self.data_smoothing_factor * loss +
                              (1 - self.data_smoothing_factor) * (self.curr_loss + self.trend))
            if self.trend is None:
                self.trend = loss - self.curr_loss
            else:
                self.trend = (self.trend_smoothing_factor * (self.curr_loss - self.prev_loss) +
                              (1 - self.trend_smoothing_factor) * self.trend)
        elif self.smoothing == 'exp':
            self.curr_loss = (self.data_smoothing_factor * loss +
                              (1 - self.data_smoothing_factor) * self.curr_loss)
        else:
            self.curr_loss = loss
        if self.curr_loss - self.best_loss < self.min_delta:
            self.best_loss = self.curr_loss
            self.waiting = 0
        elif self.waiting >= self.patience:
            raise StopIteration
        else:
            self.waiting += 1
