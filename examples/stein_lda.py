import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

import jax
from jax.experimental import stax
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.callbacks import Progbar
from numpyro.contrib.einstein import IMQKernel, Stein
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.handlers import replay
from numpyro.infer import Trace_ELBO, init_to_sample
from numpyro.infer.util import log_density
from numpyro.optim import Adam
from numpyro.util import ravel_pytree

numpyro.set_platform("cpu")


def lda(
    doc_words,
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
    numpyro.sample("topic_word_probs", dist.Delta(topic_word_probs_val).to_event(1))
    amortize_nn = numpyro.module(
        "amortize_nn",
        stax.serial(
            stax.Dense(num_hidden), stax.Relu, stax.Dense(num_topics), stax.Softmax
        ),
        (num_docs, num_max_elements),
    )
    document_topic_probs_vals = amortize_nn(doc_words)[..., None, :] + 1e-7
    numpyro.sample("topic_probs", dist.Delta(document_topic_probs_vals))


def make_batcher(data, batch_size=32, num_max_elements=-1):
    ds_count = data.shape[0] // batch_size
    num_max_elements = max(np.bincount(data.nonzero()[0]).max(), num_max_elements)

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
        return (np.array(padded_res),), {}, epoch, is_last

    return batch_fn, num_max_elements


def log2_perplexity(rng_key, stein, stein_state, batch_fn, **model_kwargs):
    particles, unravel_pytree = ravel_pytree(
        stein.get_params(stein_state), batch_dims=1
    )
    model = handlers.scale(stein._inference_model, stein.loss_temperature)

    keys = jax.random.split(rng_key, particles.shape[0])

    def particle_log_likelihood(particle, key, model_args):
        particle = unravel_pytree(particle)
        model_seed, guide_seed = jax.random.split(key)
        seeded_model = handlers.seed(model, model_seed)
        seeded_guide = handlers.seed(stein.guide, guide_seed)
        guide_trace = handlers.trace(
            handlers.substitute(seeded_guide, data=particle)
        ).get_trace(model_args, **model_kwargs)
        seeded_model = replay(seeded_model, guide_trace)
        model_log_density, _ = log_density(
            handlers.block(
                seeded_model,
                hide_fn=lambda site: site["type"] != "sample"
                or not site["is_observed"],
            ),
            (model_args,),
            model_kwargs,
            particle,
        )

        return model_log_density

    mean_likelihood = 0.0
    done = False
    i = -1
    while not done:
        (model_args,), _, _, done = batch_fn(i := i + 1)

        mean_likelihood -= (
            jax.vmap(particle_log_likelihood)(
                particles, keys, jnp.tile(model_args, (particles.shape[0], 1, 1))
            )
            / jnp.log(2)
        ).mean()

    return mean_likelihood / (i + 1)


def build_visual():
    max_num_topics = 20
    perplexities = []
    num_particles = (1, 4, 16, 64)
    for nparticles in num_particles:
        for num_topics in range(2, max_num_topics + 1, 2):
            perplexities.append(run_lda(num_topics, nparticles))
    perplexities = np.array(perplexities).reshape(4, max_num_topics // 2)

    for i in range(4):
        plt.plot(
            np.arange(2, max_num_topics + 1, 2),
            perplexities[i],
            "x--",
            label=f"{num_particles[i]} Particles",
        )
    plt.legend()
    plt.xticks(list(range(2, max_num_topics + 1, 2)))
    plt.ylabel("Log Perplexity")
    plt.xlabel("Number of Topics")
    plt.show()
    plt.clf()


def run_lda(num_topics=20, num_particles=5):
    newsgroups = fetch_20newsgroups(subset="train")
    num_words = 100
    count_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=0.01,
        token_pattern=r"(?u)\b[^\d\W]\w+\b",
        max_features=num_words,
        stop_words="english",
    )
    newsgroups_docs = count_vectorizer.fit_transform(newsgroups.data)
    batch_fn, num_max_elements = make_batcher(newsgroups_docs, batch_size=1024)
    rng_key = jax.random.PRNGKey(8938)
    inf_key, pred_key = jax.random.split(rng_key)
    stein = Stein(
        lda,
        lda_guide,
        Adam(1.0),
        Trace_ELBO(10),
        IMQKernel(),
        num_particles=num_particles,
        num_topics=num_topics,
        num_words=num_words,
        init_strategy=init_to_sample,
        num_max_elements=num_max_elements,
    )
    state, losses = stein.run(inf_key, 1000, batch_fun=batch_fn, callbacks=[Progbar()])
    plt.plot(losses)
    plt.show()
    plt.clf()

    batch_fn, _ = make_batcher(
        count_vectorizer.transform(fetch_20newsgroups(subset="test").data),
        batch_size=128,
        num_max_elements=89,
    )

    return log2_perplexity(
        jax.random.PRNGKey(0),
        stein,
        state,
        batch_fn,
        num_topics=num_topics,
        num_words=num_words,
        num_max_elements=num_max_elements,
    )


def main():
    pass


if __name__ == "__main__":
    build_visual()
