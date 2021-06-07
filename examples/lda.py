import sys

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

import jax
from jax.experimental import stax
import jax.numpy as jnp

import numpyro
from numpyro.contrib.einstein import RBFKernel, Stein, WrappedGuide
from numpyro.contrib.einstein.callbacks import Progbar
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.infer import Trace_ELBO
from numpyro.optim import Adam

numpyro.set_platform('cpu')


def lda(
    doc_words,
    lengths,
    num_topics=20,
    num_words=100,
    num_max_elements=10,
    num_hidden=100,
):
    num_docs = doc_words.shape[0]
    topic_word_probs = (
        numpyro.sample(
            "topic_word_probs",
            dist.Dirichlet(jnp.ones((num_topics, num_words)) / num_words).to_event(1),
        )
        + 1e-7
    )
    element_plate = numpyro.plate("words", num_max_elements, dim=-1)
    with numpyro.plate("documents", num_docs, dim=-2):
        document_topic_probs = numpyro.sample(
            "topic_probs", dist.Dirichlet(jnp.ones(num_topics) / num_topics)
        )
        with element_plate:
            word_topic = numpyro.sample(
                "word_topic", dist.Categorical(document_topic_probs)
            )
            numpyro.sample(
                "word",
                dist.Categorical(Vindex(topic_word_probs)[word_topic]),
                obs=doc_words,
            )


def lda_guide(
    doc_words,
    lengths,
    num_topics=20,
    num_words=100,
    num_max_elements=10,
    num_hidden=100,
):
    num_docs = doc_words.shape[0]
    topic_word_probs_val = numpyro.param(
        "topic_word_probs_val",
        jnp.ones((num_topics, num_words)),
        constraint=dist.constraints.simplex,
    )
    _topic_word_probs = numpyro.sample(
        "topic_word_probs", dist.Delta(topic_word_probs_val).to_event(1)
    )
    amortize_nn = numpyro.module(
        "amortize_nn",
        stax.serial(
            stax.Dense(num_hidden), stax.Relu, stax.Dense(num_topics), stax.Softmax
        ),
        (num_docs, num_max_elements),
    )
    document_topic_probs_vals = amortize_nn(doc_words)[..., None, :] + 1e-7
    _document_topic_probs = numpyro.sample(
        "topic_probs", dist.Delta(document_topic_probs_vals)
    )


def make_batcher(data, batch_size=32):
    ds_count = data.shape[0] // batch_size
    num_max_elements = np.bincount(data.nonzero()[0]).max()

    def batch_fn(step):
        nonlocal data
        i = step % ds_count
        epoch = step // ds_count
        is_last = i == (ds_count - 1)
        batch_values = data[i * batch_size : (i + 1) * batch_size].todense()
        res = [[] for _ in range(batch_size)]
        for idx1, idx2 in zip(*np.nonzero(batch_values)):
            res[idx1].append(idx2)
        lengths = []
        padded_res = []
        for r in res:
            padded_res.append(r + [0] * (num_max_elements - len(r)))
            lengths.append(len(r))
        if is_last:
            data = shuffle(data)
        return (np.array(padded_res), np.array(lengths)), {}, epoch, is_last

    return batch_fn, num_max_elements


def main(_argv):
    newsgroups = fetch_20newsgroups()["data"]
    num_words = 100
    count_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=0.01,
        token_pattern=r"(?u)\b[^\d\W]\w+\b",
        max_features=num_words,
        stop_words="english",
    )
    newsgroups_docs = count_vectorizer.fit_transform(newsgroups)
    batch_fn, num_max_elements = make_batcher(newsgroups_docs, batch_size=128)
    rng_key = jax.random.PRNGKey(8938)
    stein = Stein(
        lda,
        WrappedGuide(lda_guide),
        Adam(0.001),
        Trace_ELBO(),
        RBFKernel(),
        num_particles=5,
        num_topics=20,
        num_words=num_words,
        num_max_elements=num_max_elements,
    )
    stein.run(rng_key, 1000, batch_fun=batch_fn, callbacks=[Progbar()])


if __name__ == "__main__":
    main(sys.argv)
