# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import csv
import gzip
import io
import os
import pickle
import struct
from urllib.parse import urlparse
from urllib.request import urlretrieve
import warnings
import zipfile

import numpy as np

from jax import lax

from numpyro.util import find_stack_level

if "CI" in os.environ:
    DATA_DIR = os.path.expanduser("~/.data")
else:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".data"))
os.makedirs(DATA_DIR, exist_ok=True)

dset = namedtuple("dset", ["name", "urls"])

BASEBALL = dset(
    "baseball", ["https://d2hg8soec8ck9v.cloudfront.net/datasets/EfronMorrisBB.txt"]
)

BOSTON_HOUSING = dset(
    "boston_housing",
    ["https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"],
)

COVTYPE = dset(
    "covtype", ["https://d2hg8soec8ck9v.cloudfront.net/datasets/covtype.zip"]
)

DIPPER_VOLE = dset(
    "dipper_vole",
    ["https://github.com/pyro-ppl/datasets/blob/master/dipper_vole.zip?raw=true"],
)

MNIST = dset(
    "mnist",
    [
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/train-images-idx3-ubyte.gz",
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/t10k-images-idx3-ubyte.gz",
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/t10k-labels-idx1-ubyte.gz",
    ],
)

SP500 = dset("SP500", ["https://d2hg8soec8ck9v.cloudfront.net/datasets/SP500.csv"])

UCBADMIT = dset(
    "ucbadmit", ["https://d2hg8soec8ck9v.cloudfront.net/datasets/UCBadmit.csv"]
)

LYNXHARE = dset(
    "lynxhare", ["https://d2hg8soec8ck9v.cloudfront.net/datasets/LynxHare.txt"]
)

JSB_CHORALES = dset(
    "jsb_chorales",
    ["https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/jsb_chorales.pickle"],
)

HIGGS = dset(
    "higgs",
    ["https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"],
)

NINE_MERS = dset(
    "9mers",
    ["https://github.com/pyro-ppl/datasets/blob/master/9mers_data.pkl?raw=true"],
)

MORTALITY = dset(
    "mortality",
    [
        "https://github.com/pyro-ppl/datasets/blob/master/simulated_mortality.csv?raw=true"
    ],
)


def _download(dset):
    for url in dset.urls:
        file = os.path.basename(urlparse(url).path)
        out_path = os.path.join(DATA_DIR, file)
        if not os.path.exists(out_path):
            print("Downloading - {}.".format(url))
            urlretrieve(url, out_path)
            print("Download complete.")


def _load_baseball():
    _download(BASEBALL)

    def train_test_split(file):
        train, test, player_names = [], [], []
        with open(file, "r") as f:
            csv_reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                player_names.append(row["FirstName"] + " " + row["LastName"])
                at_bats, hits = row["At-Bats"], row["Hits"]
                train.append(np.array([int(at_bats), int(hits)]))
                season_at_bats, season_hits = row["SeasonAt-Bats"], row["SeasonHits"]
                test.append(np.array([int(season_at_bats), int(season_hits)]))
        return np.stack(train), np.stack(test), player_names

    train, test, player_names = train_test_split(
        os.path.join(DATA_DIR, "EfronMorrisBB.txt")
    )
    return {"train": (train, player_names), "test": (test, player_names)}


def _load_boston_housing():
    _download(BOSTON_HOUSING)
    file_path = os.path.join(DATA_DIR, "housing.data")
    data = np.loadtxt(file_path)
    return {"train": (data[:, :-1], data[:, -1])}


def _load_covtype():
    _download(COVTYPE)

    file_path = os.path.join(DATA_DIR, "covtype.zip")
    data = np.load(file_path)

    return {"train": (data["data"], data["target"])}


def _load_dipper_vole():
    _download(DIPPER_VOLE)

    file_path = os.path.join(DATA_DIR, "dipper_vole.zip")
    data = {}
    with zipfile.ZipFile(file_path) as zipper:
        data["dipper"] = (
            np.genfromtxt(zipper.open("dipper_capture_history.csv"), delimiter=",")[
                :, 1:
            ].astype(int),
            np.genfromtxt(zipper.open("dipper_sex.csv"), delimiter=",")[:, 1].astype(
                int
            ),
        )
        data["vole"] = (
            np.genfromtxt(
                zipper.open("meadow_voles_capture_history.csv"), delimiter=","
            )[:, 1:],
        )

    return data


def _load_mnist():
    _download(MNIST)

    def read_label(file):
        with gzip.open(file, "rb") as f:
            f.read(8)
            data = np.frombuffer(f.read(), dtype=np.int8)
            return data

    def read_img(file):
        with gzip.open(file, "rb") as f:
            _, _, nrows, ncols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8) / np.float32(255.0)
            return data.reshape(-1, nrows, ncols)

    files = [
        os.path.join(DATA_DIR, os.path.basename(urlparse(url).path))
        for url in MNIST.urls
    ]
    return {
        "train": (read_img(files[0]), read_label(files[1])),
        "test": (read_img(files[2]), read_label(files[3])),
    }


def _load_sp500():
    _download(SP500)

    date, value = [], []
    with open(os.path.join(DATA_DIR, "SP500.csv"), "r") as f:
        csv_reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            date.append(row["DATE"])
            value.append(float(row["VALUE"]))
    value = np.stack(value)

    return {"train": (date, value)}


def _load_ucbadmit():
    _download(UCBADMIT)

    dept, male, applications, admit = [], [], [], []
    with open(os.path.join(DATA_DIR, "UCBadmit.csv")) as f:
        csv_reader = csv.DictReader(
            f,
            delimiter=";",
            fieldnames=["index", "dept", "gender", "admit", "reject", "applications"],
        )
        next(csv_reader)  # skip the first row
        for row in csv_reader:
            dept.append(ord(row["dept"]) - ord("A"))
            male.append(row["gender"] == "male")
            applications.append(int(row["applications"]))
            admit.append(int(row["admit"]))

    return {
        "train": (
            np.stack(dept),
            np.stack(male),
            np.stack(applications),
            np.stack(admit),
        )
    }


def _load_lynxhare():
    _download(LYNXHARE)

    file_path = os.path.join(DATA_DIR, "LynxHare.txt")
    data = np.loadtxt(file_path)

    return {"train": (data[:, 0].astype(int), data[:, 1:])}


def _pad_sequence(sequences):
    # like torch.nn.utils.rnn.pad_sequence with batch_first=True
    max_length = max(x.shape[0] for x in sequences)
    padded_sequences = []
    for x in sequences:
        pad = [(0, 0)] * np.ndim(x)
        pad[0] = (0, max_length - x.shape[0])
        padded_sequences.append(np.pad(x, pad))
    return np.stack(padded_sequences)


def _load_jsb_chorales():
    _download(JSB_CHORALES)

    file_path = os.path.join(DATA_DIR, "jsb_chorales.pickle")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # XXX: we might expose those in `load_dataset` keywords
    min_note = 21
    note_range = 88
    processed_dataset = {}
    for split, data_split in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]["sequence_lengths"] = np.zeros(n_seqs, dtype=int)
        processed_dataset[split]["sequences"] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]["sequence_lengths"][seq] = seq_length
            processed_sequence = np.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = np.array(list(data_split[seq][t])) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_sequence[t, note_slice] = np.ones(slice_length)
            processed_dataset[split]["sequences"].append(processed_sequence)

    for k, v in processed_dataset.items():
        lengths = v["sequence_lengths"]
        sequences = v["sequences"]
        processed_dataset[k] = (lengths, _pad_sequence(sequences).astype("int32"))
    return processed_dataset


def _load_higgs(num_datapoints):
    warnings.warn(
        "Higgs is a 2.6 GB dataset",
        stacklevel=find_stack_level(),
    )
    _download(HIGGS)

    file_path = os.path.join(DATA_DIR, "HIGGS.csv.gz")
    with io.TextIOWrapper(gzip.open(file_path, "rb")) as f:
        csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE)
        obs = []
        data = []
        for i, row in enumerate(csv_reader):
            obs.append(int(float(row[0])))
            data.append([float(v) for v in row[1:]])
            if num_datapoints and i > num_datapoints:
                break
    obs = np.stack(obs)
    data = np.stack(data)
    (n,) = obs.shape

    return {
        "train": (data[: -(n // 20)], obs[: -(n // 20)]),
        "test": (data[-(n // 20) :], obs[-(n // 20) :]),
    }  # standard split -500_000: as test


def _load_9mers():
    _download(NINE_MERS)
    file_path = os.path.join(DATA_DIR, "9mers_data.pkl")
    return pickle.load(open(file_path, "rb"))


def _load_mortality():
    _download(MORTALITY)

    a, s1, s2, t, deaths, population = [], [], [], [], [], []
    with open(os.path.join(DATA_DIR, "simulated_mortality.csv")) as f:
        csv_reader = csv.DictReader(
            f,
            fieldnames=[
                "age_group",
                "year",
                "a",
                "s1",
                "s2",
                "t",
                "deaths",
                "population",
            ],
        )
        next(csv_reader)  # skip the first row
        for row in csv_reader:
            a.append(int(row["a"]))
            s1.append(int(row["s1"]))
            s2.append(int(row["s2"]))
            t.append(int(row["t"]))
            deaths.append(int(row["deaths"]))
            population.append(int(row["population"]))

    return {
        "train": (
            np.stack(a),
            np.stack(s1),
            np.stack(s2),
            np.stack(t),
            np.stack(deaths),
            np.stack(population),
        )
    }


def _load(dset, num_datapoints=-1):
    if dset == BASEBALL:
        return _load_baseball()
    elif dset == BOSTON_HOUSING:
        return _load_boston_housing()
    elif dset == COVTYPE:
        return _load_covtype()
    elif dset == DIPPER_VOLE:
        return _load_dipper_vole()
    elif dset == MNIST:
        return _load_mnist()
    elif dset == SP500:
        return _load_sp500()
    elif dset == UCBADMIT:
        return _load_ucbadmit()
    elif dset == LYNXHARE:
        return _load_lynxhare()
    elif dset == JSB_CHORALES:
        return _load_jsb_chorales()
    elif dset == HIGGS:
        return _load_higgs(num_datapoints)
    elif dset == NINE_MERS:
        return _load_9mers()
    elif dset == MORTALITY:
        return _load_mortality()
    raise ValueError("Dataset - {} not found.".format(dset.name))


def iter_dataset(dset, batch_size=None, split="train", shuffle=True):
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


def load_dataset(
    dset,
    batch_size=None,
    split="train",
    shuffle=True,
    num_datapoints=None,
):
    data = _load(dset, num_datapoints)
    if isinstance(data, dict):
        arrays = data[split]
    num_records = len(arrays[0])
    idxs = np.arange(num_records)
    if not batch_size:
        batch_size = num_records

    def init():
        return (
            num_records // batch_size,
            np.random.permutation(idxs) if shuffle else idxs,
        )

    def get_batch(i=0, idxs=idxs):
        ret_idx = lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        return tuple(
            np.take(a, ret_idx, axis=0)
            if isinstance(a, list)
            else lax.index_take(a, (ret_idx,), axes=(0,))
            for a in arrays
        )

    return init, get_batch
