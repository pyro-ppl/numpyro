import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import jax
from jax.experimental import stax
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.callbacks import Progbar
from numpyro.contrib.einstein import RBFKernel, Stein
from numpyro.contrib.funsor import config_enumerate
import numpyro.distributions as dist
from numpyro.handlers import replay
from numpyro.infer import MCMC, NUTS, Trace_ELBO, init_to_sample
from numpyro.infer.util import log_density
from numpyro.optim import Adam


@config_enumerate
def lda(docs, num_words, num_topics, num_hidden=100, subsample_size=10):
    with numpyro.plate("topics", num_topics):
        topic_weights = numpyro.sample(
            "topic_weight", dist.Gamma(num_topics ** (-1), 1.0)
        )
        topic_words = numpyro.sample(
            "topic_words", dist.Dirichlet(jnp.ones(num_words) ** (-1))
        )

    with numpyro.plate(
        "documents", docs.shape[1], subsample_size=subsample_size, dim=-1
    ):
        doc_topics = numpyro.sample("doc_topics", dist.Dirichlet(topic_weights))
        batch_docs = numpyro.subsample(docs, event_dim=0)
        with numpyro.plate("words", num_words, dim=-2):
            word_topics = numpyro.sample(
                "word_topic",
                dist.Categorical(probs=doc_topics),
                infer={"enumerate": "parallel"},
            )

            numpyro.sample(
                "word_count",
                dist.Categorical(probs=topic_words[word_topics]),
                obs=batch_docs,
            )


def lda_guide(docs, num_words, num_topics, num_hidden=100, subsample_size=10):
    topic_weights_posterior = numpyro.param(
        "topic_weights_posterior",
        jnp.ones(num_topics),
        constraint=dist.constraints.positive,
    )
    topic_words_posterior = numpyro.param(
        "topic_words_posterior",
        jnp.ones((num_topics, num_words)),
        constraint=dist.constraints.greater_than(0.5),
    )
    with numpyro.plate("topics", num_topics):
        numpyro.sample("topic_weight", dist.Gamma(topic_weights_posterior, 1.0))
        numpyro.sample("topic_words", dist.Dirichlet(topic_words_posterior ** (-1)))

    nn = numpyro.module(
        "amortize_nn",
        stax.serial(
            stax.Dense(num_hidden), stax.Relu, stax.Dense(num_topics), stax.Softmax
        ),
        (docs.shape[1], num_words),
    )
    with numpyro.plate(
        "documents", docs.shape[1], subsample_size=subsample_size, dim=-1
    ):
        batch_docs = numpyro.subsample(docs, event_dim=0)
        doc_topics = nn(batch_docs.T)
        doc_topics = doc_topics + 1e-7
        doc_topics = doc_topics / jnp.linalg.norm(doc_topics, axis=-1, keepdims=True)
        numpyro.sample("doc_topics", dist.Delta(doc_topics, event_dim=1))


def load_data(num_words, counter=None, subset="train"):
    newsgroups = fetch_20newsgroups(subset=subset)
    if counter is None:
        counter = CountVectorizer(
            max_df=0.95,
            max_features=num_words,
            stop_words="english",
        )
        newsgroups_docs = counter.fit_transform(newsgroups.data)
    else:
        newsgroups_docs = counter.transform(newsgroups)

    return jnp.array(newsgroups_docs.todense()), counter


def run_stein(docs, rng_key, num_words=300, num_topics=20, num_particles=1):
    inf_key, pred_key, perp_key = jax.random.split(rng_key, 3)
    stein = Stein(
        lda,
        lda_guide,
        Adam(0.001),
        Trace_ELBO(),
        RBFKernel(),
        num_particles=num_particles,
        num_topics=num_topics,
        num_words=num_words,
        init_strategy=init_to_sample,
        subsample_size=32,
    )
    state, losses = stein.run(inf_key, 100, docs, callbacks=[Progbar()])
    plt.plot(losses)
    plt.show()
    plt.clf()
    return stein, state


def run_hmc(docs, rng_key, num_words, num_topics=20):
    kernel = NUTS(lda)
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10)
    mcmc.run(rng_key, docs, num_words, num_topics)
    mcmc.print_summary()


def log2_perplexity(rng_key, stein, stein_state, *model_args, **model_kwargs):
    params = stein.get_params(stein_state)
    model = handlers.scale(stein._inference_model, stein.loss_temperature)

    model_seed, guide_seed = jax.random.split(rng_key)
    seeded_model = handlers.seed(model, model_seed)
    seeded_guide = handlers.seed(stein.guide, guide_seed)
    guide_trace = handlers.trace(
        handlers.substitute(seeded_guide, data=params)
    ).get_trace(*model_args, **model_kwargs)
    seeded_model = replay(seeded_model, guide_trace)
    model_log_density, _ = log_density(
        handlers.block(
            seeded_model,
            hide_fn=lambda site: site["type"] != "sample" or not site["is_observed"],
        ),
        model_args,
        model_kwargs,
        params,
    )

    mean_likelihood = (model_log_density / jnp.log(2)).mean()

    return 2 ** (-mean_likelihood)


def main(args=None):
    num_words = 100
    rng_key = jax.random.PRNGKey(8938)
    inf_key, perp_key = jax.random.split(rng_key)
    docs, transform = load_data(num_words)
    docs = docs[jnp.logical_not(docs.sum(-1) == 0), :]
    method, state = run_stein(docs.T, rng_key, num_words)
    test_docs, _ = load_data(num_words, transform, "test")
    print(log2_perplexity(perp_key, method, state, test_docs.T, num_words, 20))


if __name__ == "__main__":
    from jax import config

    config.update("jax_disable_jit", True)

    args = None
    main(args)
