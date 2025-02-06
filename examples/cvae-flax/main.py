# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

from data import load_dataset
import matplotlib.pyplot as plt
from train_baseline import train_baseline
from train_cvae import train_cvae

import numpyro
from numpyro.examples.datasets import MNIST

from models import BaselineNet, Decoder, Encoder, cvae_guide, cvae_model  # isort:skip


def main(args):
    train_init, train_fetch = load_dataset(
        MNIST, batch_size=args.batch_size, split="train", seed=args.rng_seed
    )
    test_init, test_fetch = load_dataset(
        MNIST, batch_size=args.batch_size, split="test", seed=args.rng_seed
    )

    num_train, train_idx = train_init()
    num_test, test_idx = test_init()

    baseline = BaselineNet()
    baseline_params = train_baseline(
        baseline,
        num_train,
        train_idx,
        train_fetch,
        num_test,
        test_idx,
        test_fetch,
        n_epochs=args.num_epochs,
    )

    cvae_params = train_cvae(
        cvae_model,
        cvae_guide,
        baseline_params,
        num_train,
        train_idx,
        train_fetch,
        num_test,
        test_idx,
        test_fetch,
        n_epochs=args.num_epochs,
    )

    x_test, y_test = test_fetch(0, test_idx)

    baseline = BaselineNet()
    recognition_net = Encoder()
    generation_net = Decoder()

    y_hat_base = baseline.apply({"params": cvae_params["baseline$params"]}, x_test)
    z_loc, z_scale = recognition_net.apply(
        {"params": cvae_params["recognition_net$params"]}, x_test, y_hat_base
    )
    y_hat_vae = generation_net.apply(
        {"params": cvae_params["generation_net$params"]}, z_loc
    )

    fig, axs = plt.subplots(4, 10, figsize=(15, 5))
    for i in range(10):
        axs[0][i].imshow(x_test[i])
        axs[1][i].imshow(y_test[i])
        axs[2][i].imshow(y_hat_base[i])
        axs[3][i].imshow(y_hat_vae[i])
        for j, label in enumerate(["Input", "Truth", "Baseline", "CVAE"]):
            axs[j][i].set_xticks([])
            axs[j][i].set_yticks([])
            if i == 0:
                axs[j][i].set_ylabel(label, rotation="horizontal", labelpad=40)
    plt.show()


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.17.0")
    parser = argparse.ArgumentParser(
        description="Conditional Variational Autoencoder on MNIST using Flax"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rng_seed", type=int, default=23)
    parser.add_argument("--num-epochs", type=int, default=10)

    args = parser.parse_args()
    main(args)
