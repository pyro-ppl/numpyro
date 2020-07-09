from jax import nn, random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam


def get_data():
    """
    The data is taken from Table 1 of reference [2].

    :return: a tuple of annotator indices and class indices. The first term has shape
        `num_positions` whose entries take values from `0` to `num_annotators - 1`.
        The second term has shape `num_items x num_positions` whose entries take values
        from `0` to `num_classes - 1`.
    """
    # NB: the first annotator assessed each item 3 times
    annotators = jnp.array([1, 1, 1, 2, 3, 4, 5])
    annotations = jnp.array([
        [1,3,1,2,2,2,1,3,2,2,4,2,1,2,1,1,1,1,2,2,2,2,2,2,1,1,2,1,1,1,1,3,1,2,2,4,2,2,3,1,1,1,2,1,2],
        [1,3,1,2,2,2,2,3,2,3,4,2,1,2,2,1,1,1,2,2,2,2,2,2,1,1,3,1,1,1,1,3,1,2,2,3,2,3,3,1,1,2,3,2,2],
        [1,3,2,2,2,2,2,3,2,2,4,2,1,2,1,1,1,1,2,2,2,2,2,1,1,1,2,1,1,2,1,3,1,2,2,3,1,2,3,1,1,1,2,1,2],
        [1,4,2,3,3,3,2,3,2,2,4,3,1,3,1,2,1,1,2,1,2,2,3,2,1,1,2,1,1,1,1,3,1,2,3,4,2,3,3,1,1,2,2,1,2],
        [1,3,1,1,2,3,1,4,2,2,4,3,1,2,1,1,1,1,2,3,2,2,2,2,1,1,2,1,1,1,1,2,1,2,2,3,2,2,4,1,1,1,2,1,2],
        [1,3,2,2,2,2,1,3,2,2,4,4,1,1,1,1,1,1,2,2,2,2,2,2,1,1,2,1,1,2,1,3,1,2,3,4,3,3,3,1,1,1,2,1,2],
        [1,4,2,1,2,2,1,3,3,3,4,3,1,2,1,1,1,1,1,2,2,1,2,2,1,1,2,1,1,1,1,3,1,2,2,3,2,3,2,1,1,1,2,1,2],
    ]).T
    # we minus 1 because in Python, the first index is 0
    return annotators - 1, annotations - 1


def multinomial(annotations):
    """
    This model corresponses to the plate diagram in Figure 1 of reference [1].
    """
    num_classes = max(annotations) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample("zeta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with numpyro.plate("position", num_positions):
            numpyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)


def dawid_skene(annotators, annotations):
    """
    This model corresponses to the plate diagram in Figure 2 of reference [1].
    """
    num_annotators = max(annotators) + 1
    num_classes = max(annotations) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            beta = numpyro.sample("beta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with numpyro.plate("position", annotations.shape[-1]):
            numpyro.sample("y", dist.Categorical(Vindex(beta)[annotators, c, :]),
                           obs=annotations)


def mace(annotators, annotations):
    """
    This model corresponses to the plate diagram in Figure 3 of reference [1].
    """
    num_annotators = max(annotators) + 1
    num_classes = max(annotations) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators):
        epsilon = numpyro.sample("epsilon", dist.Dirichlet(jnp.full(num_classes, 10)))
        theta = numpyro.sample("theta", dist.Beta(0.5, 0.5))

    with numpyro.plate("item", num_items, dim=-2):
        # NB: using constant logits for discrete uniform prior
        # (NumPyro does not have DiscreteUniform distribution yet)
        c = numpyro.sample("c", dist.Categorical(logits=jnp.zeros(num_classes)))

        with numpyro.plate("position", annotations.shape[-1]):
            s = numpyro.sample("s", dist.Bernoulli(1 - theta[annotators]))
            probs = jnp.where(s == 0, nn.one_hot(c, num_classes), epsilon[annotators])
            numpyro.sample("y", dist.Categorical(probs), obs=annotations)


def hierarchical_dawid_skene(annotators, annotations):
    """
    This model corresponses to the plate diagram in Figure 4 of reference [1].
    """
    num_annotators = max(annotators) + 1
    num_classes = max(annotations) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        # NB: we define `beta` as the `logits` of `y` likelihood; but `logits` is
        # invariant up to a constant, so we'll follow [1]: fix the last term of `beta`
        # to 0 and only define hyperpriors for the first `num_classes - 1` terms.
        zeta = numpyro.sample("zeta",
                              dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        omega = numpyro.sample("Omega",
                               dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            # non-centered parameterization
            with handlers.reparam(config={"beta": TransformReparam()}):
                base_dist = dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
                beta = numpyro.sample("beta",
                                      dist.TransformedDistribution(
                                          base_dist, AffineTransform(zeta, omega)))
            beta = jnp.pad(beta, (0, 0, 0, 0, 0, 1))  # add the last term 0

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        with numpyro.plate("position", num_positions):
            numpyro.sample("y", dist.Categorical(logits=Vindex(beta)[annotators, c, :]),
                           obs=annotations)


def item_difficulty(annotations):
    """
    This model corresponses to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = max(annotations) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        eta = numpyro.sample("eta",
                             dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        chi = numpyro.sample("Chi",
                             dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))        

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        # non-centered parameterization
        with handlers.reparam(config={"theta": TransformReparam()}):
            base_dist = dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
            theta = numpyro.sample("theta",
                                   dist.TransformedDistribution(
                                       base_dist, AffineTransform(eta[c], chi[c])))
            theta = jnp.pad(theta, (0, 0, 0, 0, 0, 1))  # add the last term 0

        with numpyro.plate("position", annotations.shape[-1]):
            numpyro.sample("y", dist.Categorical(logits=theta), obs=annotations)


def logistic_random_effects(annotators, annotations):
    """
    This model corresponses to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = max(annotations) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample("zeta",
                              dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        omega = numpyro.sample("Omega",
                               dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))
        chi = numpyro.sample("Chi",
                             dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            # non-centered parameterization
            with handlers.reparam(config={"beta": TransformReparam()}):
                base_dist = dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
                beta = numpyro.sample("beta",
                                      dist.TransformedDistribution(
                                          base_dist, AffineTransform(zeta, omega)))
            beta = jnp.pad(beta, (0, 0, 0, 0, 0, 1))  # add the last term 0

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi))

        # non-centered parameterization
        with handlers.reparam(config={"theta": TransformReparam()}):
            base_dist = dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
            theta = numpyro.sample("theta",
                                   dist.TransformedDistribution(
                                       base_dist, AffineTransform(0, chi[c])))
            theta = jnp.pad(theta, (0, 0, 0, 0, 0, 1))  # add the last term 0

        with numpyro.plate("position", annotations.shape[-1]):
            numpyro.sample("y",
                           dist.Categorical(
                               logits=Vindex(beta)[annotators, c, :] - theta),
                           obs=annotations)


def main(args):
    pass