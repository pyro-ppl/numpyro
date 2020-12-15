import sys

from jax.experimental import stax
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import jax
import jax.numpy as jnp
from sklearn.utils import shuffle

import numpyro
import numpyro.distributions as dist

import numpy as np

from numpyro.contrib.indexing import Vindex


def lda(doc_words, lengths, num_topics=20, num_words=100, num_max_elements=10,
        num_hidden=100):
    num_docs = doc_words.shape[0]
    topic_word_probs = numpyro.sample('topic_word_probs',
                                      dist.Dirichlet(jnp.ones((num_topics, num_words)) / num_words).to_event(1)) + 1e-7
    element_plate = numpyro.plate('words', num_max_elements, dim=-1)
    with numpyro.plate('documents', num_docs, dim=-2):
        document_topic_probs = numpyro.sample('topic_probs', dist.Dirichlet(jnp.ones(num_topics) / num_topics))
        with element_plate:
            word_topic = numpyro.sample('word_topic', dist.Categorical(document_topic_probs))
            numpyro.sample('word', dist.Categorical(Vindex(topic_word_probs)[word_topic]), obs=doc_words)


def lda_guide(doc_words, lengths, num_topics=20, num_words=100, num_max_elements=10,
              num_hidden=100):
    num_docs = doc_words.shape[0]
    topic_word_probs_val = numpyro.param('topic_word_probs_val', jnp.ones((num_topics, num_words)),
                                         constraint=dist.constraints.simplex)
    _topic_word_probs = numpyro.sample('topic_word_probs', dist.Delta(topic_word_probs_val).to_event(1))
    amortize_nn = numpyro.module('amortize_nn', stax.serial(
        stax.Dense(num_hidden),
        stax.Relu,
        stax.Dense(num_topics),
        stax.Softmax
    ), (num_docs, num_max_elements))
    document_topic_probs_vals = amortize_nn(doc_words)[..., None, :] + 1e-7
    _document_topic_probs = numpyro.sample('topic_probs', dist.Delta(document_topic_probs_vals))


def main(_argv):
    newsgroups = fetch_20newsgroups()['data']
    num_words = 300
    count_vectorizer = CountVectorizer(max_df=.95, min_df=.01,
                                       token_pattern=r'(?u)\b[^\d\W]\w+\b',
                                       max_features=num_words,
                                       stop_words='english')
    newsgroups_docs = count_vectorizer.fit_transform(newsgroups)
    rng_key = jax.random.PRNGKey(37)


if __name__ == '__main__':
    main(sys.argv)
