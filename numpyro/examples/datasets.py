import gzip
import os
import struct
from collections import namedtuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

from jax import device_put, lax

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '.data'))
os.makedirs(DATA_DIR, exist_ok=True)


dset = namedtuple('dset', ['name', 'urls'])


MNIST = dset('mnist', [
    'https://d2fefpcigoriu7.cloudfront.net/datasets/mnist/train-images-idx3-ubyte.gz',
    'https://d2fefpcigoriu7.cloudfront.net/datasets/mnist/train-labels-idx1-ubyte.gz',
    'https://d2fefpcigoriu7.cloudfront.net/datasets/mnist/t10k-images-idx3-ubyte.gz',
    'https://d2fefpcigoriu7.cloudfront.net/datasets/mnist/t10k-labels-idx1-ubyte.gz',
])


def _download(dset):
    for url in dset.urls:
        file = os.path.basename(urlparse(url).path)
        out_path = os.path.join(DATA_DIR, file)
        if not os.path.exists(out_path):
            print('Downloading - {}.'.format(url))
            urlretrieve(url, out_path)
            print('Download complete.')


def _load_mnist():
    _download(MNIST)

    def read_label(file):
        with gzip.open(file, 'rb') as f:
            f.read(8)
            data = np.frombuffer(f.read(), dtype=np.int8) / np.float32(255.)
            return device_put(data)

    def read_img(file):
        with gzip.open(file, 'rb') as f:
            _, _, nrows, ncols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8) / np.float32(255.)
            return device_put(data.reshape(-1, nrows, ncols))

    files = [os.path.join(DATA_DIR, os.path.basename(urlparse(url).path))
             for url in MNIST.urls]
    return {'train': (read_img(files[0]), read_label(files[1])),
            'test': (read_img(files[2]), read_label(files[3]))}


def _load(dset):
    if dset == MNIST:
        return _load_mnist()
    raise ValueError('Dataset - {} not found.'.format(dset.name))


def iter_dataset(dset, batch_size=None, split='train', shuffle=False):
    arrays = _load(dset)[split]
    num_records = len(arrays[0])
    idxs = np.arange(num_records)
    if not batch_size:
        batch_size = num_records
    if shuffle:
        idxs = np.random.permutation(idxs)
    for i in range(num_records // batch_size):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_records)
        yield tuple(a[idxs[start_idx:end_idx]] for a in arrays)


def load_dataset(dset, batch_size=None, split='train', shuffle=True):
    arrays = _load(dset)[split]
    num_records = len(arrays[0])
    idxs = np.arange(num_records)
    if not batch_size:
        batch_size = num_records

    def init():
        return num_records // batch_size, np.random.permutation(idxs) if shuffle else idxs

    def get_batch(i, idxs):
        ret_idx = lax.dynamic_slice_in_dim(idxs, (i + 1) * batch_size, batch_size)
        return tuple(lax.index_take(a, (ret_idx,), axes=(0,)) for a in arrays)

    return init, get_batch
