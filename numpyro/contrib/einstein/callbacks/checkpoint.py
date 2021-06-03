from datetime import datetime
import os
from pathlib import Path
import pickle

from jax.experimental.optimizers import pack_optimizer_state, unpack_optimizer_state

from numpyro.contrib.einstein.callbacks.callback import Callback


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


if __name__ == "__main__":
    # TODO: transfer to test!
    import jax

    import numpyro
    from numpyro.contrib.autoguide import AutoDelta
    import numpyro.distributions as dist
    from numpyro.infer import ELBO, Stein
    from numpyro.infer.initialization import init_to_value, init_with_noise
    from numpyro.infer.kernels import RBFKernel
    from numpyro.infer.stein import SteinState
    from numpyro.optim import Adam

    def model():
        numpyro.sample(
            "x",
            dist.MultivariateNormal(
                loc=jax.numpy.array([5.0, 10.0]),
                covariance_matrix=[[3.0, 5.0], [5.0, 10.0]],
            ),
        )

    guide = AutoDelta(
        model,
        init_strategy=init_with_noise(
            init_to_value(values={"x": jax.numpy.array([-10.0, 30])}), noise_scale=1.0
        ),
    )

    optim = Adam(0.1)
    loss = ELBO()
    kernel = RBFKernel()

    stein = Stein(model, guide, optim, loss, kernel)
    state = stein.init(jax.random.PRNGKey(0))
    Checkpoint("tmp.plk")._checkpoint("", 0.0, 0.0, state)
    optim_state, rng_key, loss = Checkpoint.load("tmp.plk")
    loaded_state = SteinState(optim_state, rng_key)
    print(loaded_state, loss)
