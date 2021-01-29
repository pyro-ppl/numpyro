import os
import pickle
import tarfile
from time import time
from urllib.request import urlretrieve

import numpy as np
from flax import nn
from flax.nn.activation import selu, softmax
from jax import random, device_get

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, init_to_median


def cifar10(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape((-1, 32, 32, 3)).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    return train_images, train_labels, test_images, test_labels


def summary(dataset, name, mcmc, sample_time, svi_time=0., plates={}):
    n_eff_mean = np.mean([numpyro.diagnostics.effective_sample_size(device_get(v))
                          for k, v in mcmc.get_samples(True).items() if k not in plates])
    pickle.dump(mcmc.get_samples(True), open(f'{dataset}/{name}_posterior_samples.pkl', 'wb'))
    step_field = 'num_steps' if name == 'hmc' else 'hmc_state.num_steps'
    num_step = np.sum(mcmc.get_extra_fields()[step_field])
    accpt_prob = 1.

    with open(f'{dataset}/{name}_chain_stats.txt', 'w') as f:
        print('sample_time', 'svi_time', 'n_eff_mean', 'gibbs_accpt_prob', 'tot_num_steps', 'time_per_step',
              'time_per_eff',
              sep=',', file=f)
        print(sample_time, svi_time, n_eff_mean, accpt_prob, num_step, sample_time / num_step, sample_time / n_eff_mean,
              sep=',', file=f)


class Network(nn.Module):
    """ Scaling Hamiltonian Monte Carlo Inference for Bayesian Neural Networks with Symmetric Splitting
        Adam D. Cobb, Brian Jalaian (2020) """

    def apply(self, x, out_channels):
        c1 = selu(nn.Conv(x, features=6, kernel_size=(4, 4)))
        max1 = nn.max_pool(c1, window_shape=(2, 2))
        c2 = nn.activation.selu(nn.Conv(max1, features=16, kernel_size=(4, 4)))
        max2 = nn.max_pool(c2, window_shape=(2, 2))
        l1 = selu(nn.Dense(max2.reshape(x.shape[0], -1), features=400))
        l2 = selu(nn.Dense(l1, features=120))
        l3 = selu(nn.Dense(l2, features=84))
        l4 = softmax(nn.Dense(l3, features=out_channels))
        return l4


def model(data, obs):
    module = Network.partial(out_channels=10)
    net = random_flax_module('conv_nn', module, dist.Normal(0, 1.), input_shape=data.shape)

    if obs is not None:
        obs = obs[..., None]
    numpyro.sample('obs', dist.Categorical(logits=net(data)), obs=obs)


def hmc(dataset, data, obs):
    kernel = NUTS(model, init_strategy=init_to_median)
    mcmc = MCMC(kernel, 100, 100)
    mcmc._compile(random.PRNGKey(0), data, obs, extra_fields=("num_steps",))
    start = time()
    mcmc.run(random.PRNGKey(0), data, obs, extra_fields=('num_steps',))
    summary(dataset, 'hmc', mcmc, time() - start)


def main():
    train_data, train_labels, test_data, test_labels = cifar10()
    hmc('cifar10', train_data[:1000], train_labels[:1000])


if __name__ == '__main__':
    main()
