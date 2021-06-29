# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: ProdLDA
================
In this example, we will follow [1] to implement the ProdLDA topic model from
Autoencoding Variational Inference For Topic Models by Akash Srivastava and Charles
Sutton [2]. This model returns consistently better topics than vanilla LDA and trains
much more quickly. Furthermore, it does not require a custom inference algorithm that
relies on complex mathematical derivations. This example also serves as an
introduction to Flax and Haiku modules in NumPyro.

**References:**
    1. http://pyro.ai/examples/prodlda.html
    2. Akash Srivastava, & Charles Sutton. (2017). Autoencoding Variational Inference
       For Topic Models.

.. image:: ../_static/img/examples/prodlda.png
    :align: center
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

import haiku as hk
import jax
from jax import device_put, random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.module import haiku_module
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceMeanField_ELBO


class HaikuEncoder:
    def __init__(self, vocab_size, num_topics, hidden, dropout_rate):
        self._dropout_rate = dropout_rate
        self._vocab_size = vocab_size
        self._hidden = hidden
        self._num_topics = num_topics

    def __call__(self, inputs, is_training):
        dropout_rate = self._dropout_rate if is_training else 0.0

        h = jax.nn.softplus(hk.Linear(self._hidden)(inputs))
        h = jax.nn.softplus(hk.Linear(self._hidden)(h))
        h = hk.dropout(hk.next_rng_key(), dropout_rate, h)

        # NB: here we set `create_scale=False` and `create_offset=False` to reduce
        # the number of learning parameters
        h_mu = hk.Linear(self._num_topics)(h)
        logtheta_loc = hk.BatchNorm(
            create_scale=False, create_offset=False, decay_rate=0.1
        )(h_mu, is_training)

        h_sigma = hk.Linear(self._num_topics)(h)
        logtheta_logvar = hk.BatchNorm(
            create_scale=False, create_offset=False, decay_rate=0.1
        )(h_sigma, is_training)
        logtheta_scale = jnp.exp(0.5 * logtheta_logvar)
        return logtheta_loc, logtheta_scale


class HaikuDecoder:
    def __init__(self, vocab_size, dropout_rate):
        self._dropout_rate = dropout_rate
        self._vocab_size = vocab_size

    def __call__(self, inputs, is_training):
        dropout_rate = self._dropout_rate if is_training else 0.0
        h = hk.dropout(hk.next_rng_key(), dropout_rate, inputs)
        h = hk.Linear(self._vocab_size, with_bias=False)(h)
        h = hk.BatchNorm(create_scale=False, create_offset=False, decay_rate=0.1)(
            h, is_training
        )
        return jax.nn.softmax(h)


def model(docs, hyperparams, is_training=False):
    decoder = haiku_module(
        "decoder",
        # use `transform_with_state` for BatchNorm
        hk.transform_with_state(
            HaikuDecoder(hyperparams["vocab_size"], hyperparams["dropout_rate"])
        ),
        input_shape=(1, hyperparams["num_topics"]),
        apply_rng=True,
        # to ensure proper initialisation of BatchNorm we must
        # initialise with is_training=True
        is_training=True,
    )

    with numpyro.plate(
        "documents", docs.shape[0], subsample_size=hyperparams["batch_size"]
    ):
        batch_docs = numpyro.subsample(docs, event_dim=1)
        logtheta = numpyro.sample(
            "logtheta",
            dist.Normal(0, 1).expand((hyperparams["num_topics"],)).to_event(1),
        )
        theta = jax.nn.softmax(logtheta)

        count_param = decoder(numpyro.prng_key(), theta, is_training)

        total_count = batch_docs.sum(-1)
        numpyro.sample(
            "obs", dist.Multinomial(total_count, count_param), obs=batch_docs
        )


def guide(docs, hyperparams, is_training=False):
    encoder = haiku_module(
        "encoder",
        # use `transform_with_state` for BatchNorm
        hk.transform_with_state(
            HaikuEncoder(
                hyperparams["vocab_size"],
                hyperparams["num_topics"],
                hyperparams["hidden"],
                hyperparams["dropout_rate"],
            )
        ),
        input_shape=(1, hyperparams["vocab_size"]),
        apply_rng=True,
        # to ensure proper initialisation of BatchNorm we must
        # initialise with is_training=True
        is_training=True,
    )
    with numpyro.plate(
        "documents", docs.shape[0], subsample_size=hyperparams["batch_size"]
    ):
        batch_docs = numpyro.subsample(docs, event_dim=1)
        logtheta_loc, logtheta_scale = encoder(
            numpyro.prng_key(), batch_docs, is_training
        )
        numpyro.sample(
            "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
        )


def load_data():
    news = fetch_20newsgroups(subset="all")
    vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words="english")
    docs = jnp.array(vectorizer.fit_transform(news["data"]).toarray())

    vocab = pd.DataFrame(columns=["word", "index"])
    vocab["word"] = vectorizer.get_feature_names()
    vocab["index"] = vocab.index

    return docs, vocab


def run_inference(docs, args):
    rng_key = random.PRNGKey(0)
    docs = device_put(docs)

    hyperparams = dict(
        vocab_size=docs.shape[1],
        num_topics=args.num_topics,
        hidden=args.hidden,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
    )

    optimizer = numpyro.optim.Adam(args.learning_rate)
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

    return svi.run(
        rng_key,
        args.num_steps,
        docs,
        hyperparams,
        is_training=True,
        progress_bar=not args.disable_progbar,
    )


def plot_word_cloud(b, ax, vocab, n):
    indices = jnp.argsort(b)[::-1]
    df = pd.DataFrame(indices[:100], columns=["index"])
    words = pd.merge(df, vocab[["index", "word"]], how="left", on="index")[
        "word"
    ].values.tolist()
    sizes = (b[indices[:100]] * 1000).astype(int).tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title("Topic %d" % (n + 1))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")


def main(args):
    docs, vocab = load_data()
    print(f"Dictionary size: {len(vocab)}")
    print(f"Corpus size: {docs.shape}")

    svi_result = run_inference(docs, args)

    beta = svi_result.params["decoder$params"]["linear"]["w"]

    fig, axs = plt.subplots(7, 3, figsize=(14, 24))
    for n in range(beta.shape[0]):
        i, j = divmod(n, 3)
        plot_word_cloud(beta[n], axs[i, j], vocab, n)
    axs[-1, -1].axis("off")

    fig.savefig("wordclouds.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.6.0")
    parser = argparse.ArgumentParser(
        description="Probabilistic topic modelling with Flax and Haiku"
    )
    parser.add_argument("-n", "--num-steps", nargs="?", default=30_000, type=int)
    parser.add_argument("-t", "--num-topics", nargs="?", default=20, type=int)
    parser.add_argument("--batch-size", nargs="?", default=32, type=int)
    parser.add_argument("--learning-rate", nargs="?", default=1e-3, type=float)
    parser.add_argument("--hidden", nargs="?", default=100, type=int)
    parser.add_argument("--dropout-rate", nargs="?", default=0.2, type=float)
    parser.add_argument(
        "-dp",
        "--disable-progbar",
        action="store_true",
        default=False,
        help="Whether to disable progress bar",
    )
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
