import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental import stax
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.callbacks import Progbar
from numpyro.contrib.einstein import Stein, RBFKernel
from numpyro.contrib.funsor import config_enumerate
from numpyro.infer import Trace_ELBO, init_to_sample, NUTS, MCMC, SVI
from numpyro.optim import Adam


@config_enumerate  # TODO: FIX enumerate with Funsor!
def lda(docs, num_words, num_topics, num_hidden=100, subsample_size=10):
    with numpyro.plate('topics', num_topics):
        topic_words = numpyro.sample('per_topic_word_probs', dist.Dirichlet(jnp.ones(num_words) ** (-1)))
        topic_weights = numpyro.sample('topic_weight', dist.Gamma(num_topics ** (-1), 1.))

    with numpyro.plate('documents', docs.shape[0], dim=-2):
        doc_topics = numpyro.sample('doc_topics', dist.Dirichlet(topic_weights))
        with numpyro.plate('words', num_words, dim=-1):
            word_topics = numpyro.sample('word_topic', dist.Categorical(probs=doc_topics),
                                         infer={"enumerate": "parallel"})

            numpyro.sample('word_count', dist.Categorical(probs=topic_words[word_topics]), obs=docs)


def lda_guide(docs, num_words, num_topics, num_hidden=100, subsample_size=10):
    words_probs_val = numpyro.param("word_probs_val", jnp.ones((num_topics, num_words)),
                                    constraint=dist.constraints.simplex)
    numpyro.sample('word_probs', dist.Delta(words_probs_val))
    nn = numpyro.module("amortize_nn",
                        stax.serial(stax.Dense(num_hidden), stax.Relu, stax.Dense(num_topics), stax.Softmax),
                        (docs.shape[0], num_words))
    topic_probs_val = nn(docs) + 1e-7
    numpyro.sample('topic_probs', dist.Delta(topic_probs_val))


def load_data(num_words):
    newsgroups = fetch_20newsgroups(subset="train")
    count_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=0.01,
        token_pattern=r"(?u)\b[^\d\W]\w+\b",
        max_features=num_words,
        stop_words="english",
    )
    newsgroups_docs = count_vectorizer.fit_transform(newsgroups.data)[:10]
    return jnp.array(newsgroups_docs.todense())


def run_stein(docs, rng_key, num_words=300, num_topics=20, num_particles=1):
    inf_key, pred_key = jax.random.split(rng_key)
    stein = Stein(
        lda,
        lda_guide,
        Adam(.001),
        Trace_ELBO(),
        RBFKernel(),
        num_particles=num_particles,
        num_topics=num_topics,
        num_words=num_words,
        init_strategy=init_to_sample,
    )
    state, losses = stein.run(inf_key, 100, docs, callbacks=[Progbar()])
    plt.plot(losses)
    plt.show()
    plt.clf()


def run_hmc(docs, rng_key, num_words, num_topics=20):
    kernel = NUTS(lda)
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10)
    mcmc.run(rng_key, docs, num_words, num_topics)
    mcmc.print_summary()


def main(args=None):
    num_words = 4
    rng_key = jax.random.PRNGKey(8938)
    docs = load_data(num_words)
    run_hmc(docs, rng_key, num_words)


if __name__ == '__main__':
    args = None
    main(args)
