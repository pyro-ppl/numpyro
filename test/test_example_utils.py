import jax.numpy as np
from jax import lax

from numpyro.examples.datasets import BASEBALL, MNIST, load_dataset


def test_mnist_data_load():
    def mean_pixels(i, mean_pix):
        batch, _ = fetch(i, idx)
        return mean_pix + np.sum(batch) / batch.size

    init, fetch = load_dataset(MNIST, batch_size=128, split='train')
    num_batches, idx = init()
    assert lax.fori_loop(0, num_batches, mean_pixels, np.float32(0.)) / num_batches < 0.15


def test_baseball_data_load():
    init, fetch = load_dataset(BASEBALL, split='train', shuffle=False)
    num_batches, idx = init()
    dataset = fetch(0, idx)
    assert np.shape(dataset[0]) == (18, 2)
    assert np.shape(dataset[1]) == (18,)
