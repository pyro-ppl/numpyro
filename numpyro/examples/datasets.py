# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Generator
import csv
import gzip
from http import HTTPStatus
import io
import os
import pickle
import shutil
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
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

BART = dset(
    "bart",
    [
        "https://github.com/pyro-ppl/datasets/blob/master/bart/bart_0.npz?raw=true",
        "https://github.com/pyro-ppl/datasets/blob/master/bart/bart_1.npz?raw=true",
        "https://github.com/pyro-ppl/datasets/blob/master/bart/bart_2.npz?raw=true",
        "https://github.com/pyro-ppl/datasets/blob/master/bart/bart_3.npz?raw=true",
    ],
)

BASEBALL = dset(
    "baseball",
    ["https://github.com/pyro-ppl/datasets/blob/master/EfronMorrisBB.txt?raw=true"],
)

BOSTON_HOUSING = dset(
    "boston_housing",
    ["https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"],
)

COVTYPE = dset(
    "covtype", ["https://github.com/pyro-ppl/datasets/blob/master/covtype.npz?raw=true"]
)

DIPPER_VOLE = dset(
    "dipper_vole",
    ["https://github.com/pyro-ppl/datasets/blob/master/dipper_vole.zip?raw=true"],
)

MNIST = dset(
    "mnist",
    [
        "https://github.com/pyro-ppl/datasets/blob/master/mnist/train-images-idx3-ubyte.gz?raw=true",
        "https://github.com/pyro-ppl/datasets/blob/master/mnist/train-labels-idx1-ubyte.gz?raw=true",
        "https://github.com/pyro-ppl/datasets/blob/master/mnist/t10k-images-idx3-ubyte.gz?raw=true",
        "https://github.com/pyro-ppl/datasets/blob/master/mnist/t10k-labels-idx1-ubyte.gz?raw=true",
    ],
)

SP500 = dset(
    "SP500", ["https://github.com/pyro-ppl/datasets/blob/master/SP500.csv?raw=true"]
)

UCBADMIT = dset(
    "ucbadmit",
    ["https://github.com/pyro-ppl/datasets/blob/master/UCBadmit.csv?raw=true"],
)

LYNXHARE = dset(
    "lynxhare",
    ["https://github.com/pyro-ppl/datasets/blob/master/LynxHare.txt?raw=true"],
)

JSB_CHORALES = dset(
    "jsb_chorales",
    [
        "https://github.com/pyro-ppl/datasets/blob/master/polyphonic/jsb_chorales.pickle?raw=true"
    ],
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


_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_BACKOFF_BASE_SECONDS = 1.0
_DOWNLOAD_BACKOFF_MAX_SECONDS = 8.0
_TRANSIENT_HTTP_STATUS_CODES = {
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


def _github_raw_fallback_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return None
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 4 or path_parts[2] != "blob":
        return None
    owner, repo = path_parts[0], path_parts[1]
    branch_and_path = path_parts[3:]
    return "https://raw.githubusercontent.com/{}".format(
        "/".join([owner, repo, *branch_and_path])
    )


def _is_retryable_error(exc: HTTPError | URLError) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS_CODES
    return isinstance(exc, URLError)


def _detached_download_error(exc: HTTPError | URLError) -> HTTPError | URLError:
    if isinstance(exc, HTTPError):
        return HTTPError(exc.url, exc.code, exc.msg, exc.headers, fp=None)
    if isinstance(exc, URLError):
        return URLError(exc.reason)
    return exc


def _download_backoff_delay(exc: HTTPError | URLError, attempt: int) -> float:
    if isinstance(exc, HTTPError):
        retry_after = exc.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(_DOWNLOAD_BACKOFF_MAX_SECONDS, max(0.0, float(retry_after)))
            except (TypeError, ValueError):
                pass
    return min(
        _DOWNLOAD_BACKOFF_MAX_SECONDS, _DOWNLOAD_BACKOFF_BASE_SECONDS * (2**attempt)
    )


def _download_with_retries(url: str, out_path: str) -> None:
    last_exc = None
    for attempt in range(_DOWNLOAD_MAX_RETRIES):
        try:
            request = Request(url, headers={"User-Agent": "numpyro-datasets"})
            with urlopen(request) as response, open(out_path, "wb") as f:
                shutil.copyfileobj(response, f)
            if isinstance(last_exc, HTTPError):
                last_exc.close()
            return
        except (HTTPError, URLError) as exc:
            retryable = _is_retryable_error(exc)
            delay = _download_backoff_delay(exc, attempt) if retryable else None
            detached_exc = _detached_download_error(exc)
            if isinstance(exc, HTTPError):
                exc.close()
            if isinstance(last_exc, HTTPError):
                last_exc.close()
            last_exc = detached_exc
            if not retryable:
                if isinstance(detached_exc, HTTPError):
                    detached_exc.close()
                raise detached_exc
            if attempt < _DOWNLOAD_MAX_RETRIES - 1:
                print(
                    "Download failed with {}. Retrying in {:.1f}s.".format(
                        type(detached_exc).__name__, delay
                    )
                )
                time.sleep(delay)
    if isinstance(last_exc, HTTPError):
        last_exc.close()
    raise last_exc


def _download(dset: dset) -> None:
    for url in dset.urls:
        file = os.path.basename(urlparse(url).path)
        out_path = os.path.join(DATA_DIR, file)
        if os.path.exists(out_path):
            continue

        print("Downloading - {}.".format(url))
        fallback_url = _github_raw_fallback_url(url)
        download_urls = [url, fallback_url] if fallback_url else [url]
        last_exc = None
        for i, download_url in enumerate(download_urls):
            try:
                _download_with_retries(download_url, out_path)
            except (HTTPError, URLError) as exc:
                last_exc = exc
                if os.path.exists(out_path):
                    os.remove(out_path)
                if i < len(download_urls) - 1:
                    print(
                        "Download failed for {} with {}. Trying fallback URL.".format(
                            download_url, type(exc).__name__
                        )
                    )
                continue

            print("Download complete.")
            break
        else:
            if isinstance(last_exc, HTTPError):
                last_exc.close()
            raise last_exc


def load_bart_od() -> dict:
    _download(BART)

    filenames = [os.path.join(DATA_DIR, f"bart_{i}.npz") for i in range(4)]
    stations = None
    start_date = None
    counts_list = []
    for filename in filenames:
        with np.load(filename, allow_pickle=True) as dataset:
            if stations is None:
                stations = dataset["stations"]
                start_date = dataset["start_date"]
            counts_list.append(dataset["counts"])
    counts = np.vstack(counts_list)
    return {
        "stations": stations,
        "start_date": start_date,
        "counts": counts,
    }


def _load_baseball() -> dict:
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


def _load_boston_housing() -> dict:
    _download(BOSTON_HOUSING)
    file_path = os.path.join(DATA_DIR, "housing.data")
    data = np.loadtxt(file_path)
    return {"train": (data[:, :-1], data[:, -1])}


def _load_covtype() -> dict:
    _download(COVTYPE)

    file_path = os.path.join(DATA_DIR, "covtype.npz")
    with np.load(file_path) as data:
        return {"train": (data["data"], data["target"])}


def _load_dipper_vole() -> dict:
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


def _load_mnist() -> dict:
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


def _load_sp500() -> dict:
    _download(SP500)

    date, value = [], []
    with open(os.path.join(DATA_DIR, "SP500.csv"), "r") as f:
        csv_reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            date.append(row["DATE"])
            value.append(float(row["VALUE"]))
    value = np.stack(value)

    return {"train": (date, value)}


def _load_ucbadmit() -> dict:
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


def _load_lynxhare() -> dict:
    _download(LYNXHARE)

    file_path = os.path.join(DATA_DIR, "LynxHare.txt")
    data = np.loadtxt(file_path)

    return {"train": (data[:, 0].astype(int), data[:, 1:])}


def _pad_sequence(sequences: list[np.ndarray]) -> np.ndarray:
    # like torch.nn.utils.rnn.pad_sequence with batch_first=True
    max_length = max(x.shape[0] for x in sequences)
    padded_sequences = []
    for x in sequences:
        pad = [(0, 0)] * np.ndim(x)
        pad[0] = (0, max_length - x.shape[0])
        padded_sequences.append(np.pad(x, pad))
    return np.stack(padded_sequences)


def _load_jsb_chorales() -> dict:
    _download(JSB_CHORALES)

    file_path = os.path.join(DATA_DIR, "jsb_chorales.pickle")
    with open(file_path, "rb") as f:
        # Filter numpy deprecation warning from loading legacy pickle file
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="dtype.*align should be passed as Python or NumPy boolean",
                category=np.exceptions.VisibleDeprecationWarning,
            )
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


def _load_higgs(num_datapoints: int) -> dict:
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


def _load_9mers() -> dict:
    _download(NINE_MERS)
    file_path = os.path.join(DATA_DIR, "9mers_data.pkl")
    return pickle.load(open(file_path, "rb"))


def _load_mortality() -> dict:
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


def _load(dset: dset, num_datapoints: int = -1) -> dict:
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


def iter_dataset(
    dset: dset,
    batch_size: int | None = None,
    split: str = "train",
    shuffle: bool = True,
) -> Generator[tuple[np.ndarray, ...], None, None]:
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
    dset: dset,
    batch_size: int | None = None,
    split: str = "train",
    shuffle: bool = True,
    num_datapoints: int | None = None,
) -> tuple[Callable, Callable]:
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
