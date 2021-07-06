# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: ProdLDA with Flax and Haiku
====================================

In this example, we will follow [1] to implement the ProdLDA topic model from
Autoencoding Variational Inference For Topic Models by Akash Srivastava and Charles
Sutton [2]. This model returns consistently better topics than vanilla LDA and trains
much more quickly. Furthermore, it does not require a custom inference algorithm that
relies on complex mathematical derivations. This example also serves as an
introduction to Flax and Haiku modules in NumPyro.

Note that unlike [1, 2], this implementation uses a Dirichlet prior directly rather
than approximating it with a softmax-normal distribution.

For the interested reader, a nice extension of this model is the CombinedTM model [3]
which utilizes a pre-trained sentence transformer (like https://www.sbert.net/) to
generate a better representation of the encoded latent vector.

**References:**
    1. http://pyro.ai/examples/prodlda.html
    2. Akash Srivastava, & Charles Sutton. (2017). Autoencoding Variational Inference
       For Topic Models.
    3. Federico Bianchi, Silvia Terragni, and Dirk Hovy (2021), "Pre-training is a Hot
       Topic: Contextualized Document Embeddings Improve Topic Coherence"
       (https://arxiv.org/abs/2004.03974)

.. image:: ../_static/img/examples/prodlda.png
    :align: center
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

import flax.linen as nn
import haiku as hk
import jax
from jax import device_put, random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.module import flax_module, haiku_module
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceMeanField_ELBO


class HaikuEncoder:
    def __init__(self, vocab_size, num_topics, hidden, dropout_rate):
        self._vocab_size = vocab_size
        self._num_topics = num_topics
        self._hidden = hidden
        self._dropout_rate = dropout_rate

    def __call__(self, inputs, is_training):
        dropout_rate = self._dropout_rate if is_training else 0.0

        h = jax.nn.softplus(hk.Linear(self._hidden)(inputs))
        h = jax.nn.softplus(hk.Linear(self._hidden)(h))
        h = hk.dropout(hk.next_rng_key(), dropout_rate, h)
        h = hk.Linear(self._num_topics)(h)

        # NB: here we set `create_scale=False` and `create_offset=False` to reduce
        # the number of learning parameters
        log_concentration = hk.BatchNorm(
            create_scale=False, create_offset=False, decay_rate=0.9
        )(h, is_training)
        return jnp.exp(log_concentration)


class HaikuDecoder:
    def __init__(self, vocab_size, dropout_rate):
        self._vocab_size = vocab_size
        self._dropout_rate = dropout_rate

    def __call__(self, inputs, is_training):
        dropout_rate = self._dropout_rate if is_training else 0.0
        h = hk.dropout(hk.next_rng_key(), dropout_rate, inputs)
        h = hk.Linear(self._vocab_size, with_bias=False)(h)
        return hk.BatchNorm(create_scale=False, create_offset=False, decay_rate=0.9)(
            h, is_training
        )


class FlaxEncoder(nn.Module):
    vocab_size: int
    num_topics: int
    hidden: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, is_training):
        h = nn.softplus(nn.Dense(self.hidden)(inputs))
        h = nn.softplus(nn.Dense(self.hidden)(h))
        h = nn.Dropout(self.dropout_rate, deterministic=not is_training)(h)
        h = nn.Dense(self.num_topics)(h)

        log_concentration = nn.BatchNorm(
            use_bias=False,
            use_scale=False,
            momentum=0.9,
            use_running_average=not is_training,
        )(h)
        return jnp.exp(log_concentration)


class FlaxDecoder(nn.Module):
    vocab_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, is_training):
        h = nn.Dropout(self.dropout_rate, deterministic=not is_training)(inputs)
        h = nn.Dense(self.vocab_size, use_bias=False)(h)
        return nn.BatchNorm(
            use_bias=False,
            use_scale=False,
            momentum=0.9,
            use_running_average=not is_training,
        )(h)


def model(docs, hyperparams, is_training=False, nn_framework="flax"):
    if nn_framework == "flax":
        decoder = flax_module(
            "decoder",
            FlaxDecoder(hyperparams["vocab_size"], hyperparams["dropout_rate"]),
            input_shape=(1, hyperparams["num_topics"]),
            # ensure PRNGKey is made available to dropout layers
            apply_rng=["dropout"],
            # indicate mutable state due to BatchNorm layers
            mutable=["batch_stats"],
            # to ensure proper initialisation of BatchNorm we must
            # initialise with is_training=True
            is_training=True,
        )
    elif nn_framework == "haiku":
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
    else:
        raise ValueError(f"Invalid choice {nn_framework} for argument nn_framework")

    with numpyro.plate(
        "documents", docs.shape[0], subsample_size=hyperparams["batch_size"]
    ):
        batch_docs = numpyro.subsample(docs, event_dim=1)
        theta = numpyro.sample(
            "theta", dist.Dirichlet(jnp.ones(hyperparams["num_topics"]))
        )

        if nn_framework == "flax":
            logits = decoder(theta, is_training, rngs={"dropout": numpyro.prng_key()})
        elif nn_framework == "haiku":
            logits = decoder(numpyro.prng_key(), theta, is_training)

        total_count = batch_docs.sum(-1)
        numpyro.sample(
            "obs", dist.Multinomial(total_count, logits=logits), obs=batch_docs
        )


def guide(docs, hyperparams, is_training=False, nn_framework="flax"):
    if nn_framework == "flax":
        encoder = flax_module(
            "encoder",
            FlaxEncoder(
                hyperparams["vocab_size"],
                hyperparams["num_topics"],
                hyperparams["hidden"],
                hyperparams["dropout_rate"],
            ),
            input_shape=(1, hyperparams["vocab_size"]),
            # ensure PRNGKey is made available to dropout layers
            apply_rng=["dropout"],
            # indicate mutable state due to BatchNorm layers
            mutable=["batch_stats"],
            # to ensure proper initialisation of BatchNorm we must
            # initialise with is_training=True
            is_training=True,
        )
    elif nn_framework == "haiku":
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
    else:
        raise ValueError(f"Invalid choice {nn_framework} for argument nn_framework")

    with numpyro.plate(
        "documents", docs.shape[0], subsample_size=hyperparams["batch_size"]
    ):
        batch_docs = numpyro.subsample(docs, event_dim=1)

        if nn_framework == "flax":
            concentration = encoder(
                batch_docs, is_training, rngs={"dropout": numpyro.prng_key()}
            )
        elif nn_framework == "haiku":
            concentration = encoder(numpyro.prng_key(), batch_docs, is_training)

        numpyro.sample("theta", dist.Dirichlet(concentration))


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
        nn_framework=args.nn_framework,
    )


def plot_word_cloud(b, ax, vocab, n):
    indices = jnp.argsort(b)[::-1]
    top20 = indices[:20]
    df = pd.DataFrame(top20, columns=["index"])
    words = pd.merge(df, vocab[["index", "word"]], how="left", on="index")[
        "word"
    ].values.tolist()
    sizes = b[top20].tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title(f"Topic {n + 1}")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")


def main(args):
    docs, vocab = load_data()
    print(f"Dictionary size: {len(vocab)}")
    print(f"Corpus size: {docs.shape}")

    svi_result = run_inference(docs, args)

    if args.nn_framework == "flax":
        beta = svi_result.params["decoder$params"]["Dense_0"]["kernel"]
    elif args.nn_framework == "haiku":
        beta = svi_result.params["decoder$params"]["linear"]["w"]

    beta = jax.nn.softmax(beta)

    # the number of plots depends on the chosen number of topics.
    # add 2 to num topics to ensure we create a row for any remainder after division
    nrows = (args.num_topics + 2) // 3
    fig, axs = plt.subplots(nrows, 3, figsize=(14, 3 + 3 * nrows))
    axs = axs.flatten()

    for n in range(beta.shape[0]):
        plot_word_cloud(beta[n], axs[n], vocab, n)

    # hide any unused axes
    for i in range(n, len(axs)):
        axs[i].axis("off")

    fig.savefig("wordclouds.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.7.0")
    parser = argparse.ArgumentParser(
        description="Probabilistic topic modelling with Flax and Haiku"
    )
    parser.add_argument("-n", "--num-steps", nargs="?", default=30_000, type=int)
    parser.add_argument("-t", "--num-topics", nargs="?", default=12, type=int)
    parser.add_argument("--batch-size", nargs="?", default=32, type=int)
    parser.add_argument("--learning-rate", nargs="?", default=1e-3, type=float)
    parser.add_argument("--hidden", nargs="?", default=200, type=int)
    parser.add_argument("--dropout-rate", nargs="?", default=0.2, type=float)
    parser.add_argument(
        "-dp",
        "--disable-progbar",
        action="store_true",
        default=False,
        help="Whether to disable progress bar",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help='use "cpu", "gpu" or "tpu".'
    )
    parser.add_argument(
        "--nn-framework",
        nargs="?",
        default="flax",
        help=(
            "The framework to use for constructing encoder / decoder. Options are "
            '"flax" or "haiku".'
        ),
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
