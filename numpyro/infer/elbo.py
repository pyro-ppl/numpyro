# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from functools import partial
from operator import itemgetter
import warnings

from jax import random, vmap
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpyro.distributions.kl import kl_divergence
from numpyro.distributions.util import scale_and_mask
from numpyro.handlers import Messenger, replay, seed, substitute, trace
from numpyro.infer.util import get_importance_trace, log_density
from numpyro.ops.provenance import eval_provenance, get_provenance
from numpyro.util import _validate_model, check_model_guide_match, find_stack_level


class ELBO:
    """
    Base class for all ELBO objectives.

    Subclasses should implement either :meth:`loss` or :meth:`loss_with_mutable_state`.

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    """

    """
    Determines whether the ELBO objective can support inference of discrete latent variables.

    Subclasses that are capable of inferring  discrete latent variables should override to `True`
    """
    can_infer_discrete = False

    def __init__(self, num_particles=1):
        self.num_particles = num_particles

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param dict param_map: dictionary of current parameter values keyed by site
            name.
        :param model: Python callable with NumPyro primitives for the model.
        :param guide: Python callable with NumPyro primitives for the guide.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: negative of the Evidence Lower Bound (ELBO) to be minimized.
        """
        return self.loss_with_mutable_state(
            rng_key, param_map, model, guide, *args, **kwargs
        )["loss"]

    def loss_with_mutable_state(
        self, rng_key, param_map, model, guide, *args, **kwargs
    ):
        """
        Like :meth:`loss` but also update and return the mutable state, which stores the
        values at :func:`~numpyro.mutable` sites.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param dict param_map: dictionary of current parameter values keyed by site
            name.
        :param model: Python callable with NumPyro primitives for the model.
        :param guide: Python callable with NumPyro primitives for the guide.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: dictionay containing ELBO loss and the mutable state
        """
        raise NotImplementedError("This ELBO objective does not support mutable state.")


class Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide.

    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variables with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    **References:**

    1. *Automated Variational Inference in Probabilistic Programming*,
       David Wingate, Theo Weber
    2. *Black Box Variational Inference*,
       Rajesh Ranganath, Sean Gerrish, David M. Blei

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    """

    def __init__(self, num_particles=1):
        self.num_particles = num_particles

    def loss_with_mutable_state(
        self, rng_key, param_map, model, guide, *args, **kwargs
    ):
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(
                seeded_guide, args, kwargs, param_map
            )
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, model_trace = log_density(
                seeded_model, args, kwargs, params
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            mutable_params.update(
                {
                    name: site["value"]
                    for name, site in model_trace.items()
                    if site["type"] == "mutable"
                }
            )

            # log p(z) - log q(z)
            elbo_particle = model_log_density - guide_log_density
            if mutable_params:
                if self.num_particles == 1:
                    return elbo_particle, mutable_params
                else:
                    raise ValueError(
                        "Currently, we only support mutable states with num_particles=1."
                    )
            else:
                return elbo_particle, None

        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        if self.num_particles == 1:
            elbo, mutable_state = single_particle_elbo(rng_key)
            return {"loss": -elbo, "mutable_state": mutable_state}
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            elbos, mutable_state = vmap(single_particle_elbo)(rng_keys)
            return {"loss": -jnp.mean(elbos), "mutable_state": mutable_state}


def _get_log_prob_sum(site):
    if site["intermediates"]:
        log_prob = site["fn"].log_prob(site["value"], site["intermediates"])
    else:
        log_prob = site["fn"].log_prob(site["value"])
    log_prob = scale_and_mask(log_prob, site["scale"])
    return jnp.sum(log_prob)


def _check_mean_field_requirement(model_trace, guide_trace):
    """
    Checks that the guide and model sample sites are ordered identically.
    This is sufficient but not necessary for correctness.
    """
    model_sites = [
        name
        for name, site in model_trace.items()
        if site["type"] == "sample" and name in guide_trace
    ]
    guide_sites = [
        name
        for name, site in guide_trace.items()
        if site["type"] == "sample" and name in model_trace
    ]
    assert set(model_sites) == set(guide_sites)
    if model_sites != guide_sites:
        warnings.warn(
            "Failed to verify mean field restriction on the guide. "
            "To eliminate this warning, ensure model and guide sites "
            "occur in the same order.\n"
            + "Model sites:\n  "
            + "\n  ".join(model_sites)
            + "Guide sites:\n  "
            + "\n  ".join(guide_sites),
            stacklevel=find_stack_level(),
        ),


class TraceMeanField_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI. This is currently the only
    ELBO estimator in NumPyro that uses analytic KL divergences when those
    are available.

    .. warning:: This estimator may give incorrect results if the mean-field
        condition is not satisfied.
        The mean field condition is a sufficient but not necessary condition for
        this estimator to be correct. The precise condition is that for every
        latent variable `z` in the guide, its parents in the model must not include
        any latent variables that are descendants of `z` in the guide. Here
        'parents in the model' and 'descendants in the guide' is with respect
        to the corresponding (statistical) dependency structure. For example, this
        condition is always satisfied if the model and guide have identical
        dependency structures.
    """

    def loss_with_mutable_state(
        self, rng_key, param_map, model, guide, *args, **kwargs
    ):
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            subs_guide = substitute(seeded_guide, data=param_map)
            guide_trace = trace(subs_guide).get_trace(*args, **kwargs)
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            subs_model = substitute(replay(seeded_model, guide_trace), data=params)
            model_trace = trace(subs_model).get_trace(*args, **kwargs)
            mutable_params.update(
                {
                    name: site["value"]
                    for name, site in model_trace.items()
                    if site["type"] == "mutable"
                }
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            _check_mean_field_requirement(model_trace, guide_trace)

            elbo_particle = 0
            for name, model_site in model_trace.items():
                if model_site["type"] == "sample":
                    if model_site["is_observed"]:
                        elbo_particle = elbo_particle + _get_log_prob_sum(model_site)
                    else:
                        guide_site = guide_trace[name]
                        try:
                            kl_qp = kl_divergence(guide_site["fn"], model_site["fn"])
                            kl_qp = scale_and_mask(kl_qp, scale=guide_site["scale"])
                            elbo_particle = elbo_particle - jnp.sum(kl_qp)
                        except NotImplementedError:
                            elbo_particle = (
                                elbo_particle
                                + _get_log_prob_sum(model_site)
                                - _get_log_prob_sum(guide_site)
                            )

            # handle auxiliary sites in the guide
            for name, site in guide_trace.items():
                if site["type"] == "sample" and name not in model_trace:
                    assert site["infer"].get("is_auxiliary") or site["is_observed"]
                    elbo_particle = elbo_particle - _get_log_prob_sum(site)

            if mutable_params:
                if self.num_particles == 1:
                    return elbo_particle, mutable_params
                else:
                    raise ValueError(
                        "Currently, we only support mutable states with num_particles=1."
                    )
            else:
                return elbo_particle, None

        if self.num_particles == 1:
            elbo, mutable_state = single_particle_elbo(rng_key)
            return {"loss": -elbo, "mutable_state": mutable_state}
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            elbos, mutable_state = vmap(single_particle_elbo)(rng_keys)
            return {"loss": -jnp.mean(elbos), "mutable_state": mutable_state}


class RenyiELBO(ELBO):
    r"""
    An implementation of Renyi's :math:`\alpha`-divergence
    variational inference following reference [1].
    In order for the objective to be a strict lower bound, we require
    :math:`\alpha \ge 0`. Note, however, that according to reference [1], depending
    on the dataset :math:`\alpha < 0` might give better results. In the special case
    :math:`\alpha = 0`, the objective function is that of the important weighted
    autoencoder derived in reference [2].

    .. note:: Setting :math:`\alpha < 1` gives a better bound than the usual ELBO.

    :param float alpha: The order of :math:`\alpha`-divergence.
        Here :math:`\alpha \neq 1`. Default is 0.
    :param num_particles: The number of particles/samples
        used to form the objective (gradient) estimator. Default is 2.

    **References:**

    1. *Renyi Divergence Variational Inference*, Yingzhen Li, Richard E. Turner
    2. *Importance Weighted Autoencoders*, Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
    """

    def __init__(self, alpha=0, num_particles=2):
        if alpha == 1:
            raise ValueError(
                "The order alpha should not be equal to 1. Please use ELBO class"
                "for the case alpha = 1."
            )
        self.alpha = alpha
        super().__init__(num_particles=num_particles)

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        def single_particle_elbo(rng_key):
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(
                seeded_guide, args, kwargs, param_map
            )
            # NB: we only want to substitute params not available in guide_trace
            model_param_map = {
                k: v for k, v in param_map.items() if k not in guide_trace
            }
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, model_trace = log_density(
                seeded_model, args, kwargs, model_param_map
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")

            # log p(z) - log q(z)
            elbo = model_log_density - guide_log_density
            return elbo

        rng_keys = random.split(rng_key, self.num_particles)
        elbos = vmap(single_particle_elbo)(rng_keys)
        scaled_elbos = (1.0 - self.alpha) * elbos
        avg_log_exp = logsumexp(scaled_elbos) - jnp.log(self.num_particles)
        weights = jnp.exp(scaled_elbos - avg_log_exp)
        renyi_elbo = avg_log_exp / (1.0 - self.alpha)
        weighted_elbo = jnp.dot(stop_gradient(weights), elbos) / self.num_particles
        return -(stop_gradient(renyi_elbo - weighted_elbo) + weighted_elbo)


def _get_plate_stacks(trace):
    """
    This builds a dict mapping site name to a set of plate stacks. Each
    plate stack is a list of :class:`CondIndepStackFrame`s corresponding to
    a :class:`plate`. This information is used by :class:`Trace_ELBO` and
    :class:`TraceGraph_ELBO`.
    """
    return {
        name: [f for f in node["cond_indep_stack"]]
        for name, node in trace.items()
        if node["type"] == "sample"
    }


class MultiFrameTensor(dict):
    """
    A container for sums of Tensors among different :class:`plate` contexts.
    Used in :class:`~numpyro.infer.elbo.TraceGraph_ELBO` to simplify
    downstream cost computation logic.

    Example::

        downstream_cost = MultiFrameTensor()
        for site in downstream_nodes:
            downstream_cost.add((site["cond_indep_stack"], site["log_prob"]))
        downstream_cost.add(*other_costs.items())  # add in bulk
        summed = downstream_cost.sum_to(target_site["cond_indep_stack"])
    """

    def __init__(self, *items):
        super().__init__()
        self.add(*items)

    def add(self, *items):
        """
        Add a collection of (cond_indep_stack, tensor) pairs. Keys are
        ``cond_indep_stack``s, i.e. tuples of :class:`CondIndepStackFrame`s.
        Values are :class:`numpy.ndarray`s.
        """
        for cond_indep_stack, value in items:
            frames = frozenset(f for f in cond_indep_stack)
            assert all(f.dim < 0 and -jnp.ndim(value) <= f.dim for f in frames)
            if frames in self:
                self[frames] = self[frames] + value
            else:
                self[frames] = value

    def sum_to(self, target_frames):
        total = None
        for frames, value in self.items():
            for f in frames:
                if f not in target_frames and jnp.shape(value)[f.dim] != 1:
                    value = value.sum(f.dim, keepdims=True)
            while jnp.shape(value) and jnp.shape(value)[0] == 1:
                value = value.squeeze(0)
            total = value if total is None else total + value
        return 0.0 if total is None else total

    def __repr__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ",\n\t".join(["({}, ...)".format(frames) for frames in self]),
        )


def _identify_dense_edges(trace):
    succ = {}
    for name, node in trace.items():
        if node["type"] == "sample":
            succ[name] = set()
    for name, node in trace.items():
        if node["type"] == "sample":
            for past_name, past_node in trace.items():
                if past_node["type"] == "sample":
                    if past_name == name:
                        break
                    # XXX: different from Pyro, we always add edge past_name -> name
                    succ[past_name].add(name)
    return succ


def _topological_sort(succ, reverse=False):
    """
    Return a list of nodes (site names) in topologically sorted order.
    """

    def dfs(site, visited):
        if site in visited:
            return
        for s in succ[site]:
            for node in dfs(s, visited):
                yield node
        visited.add(site)
        yield site

    visited = set()
    top_sorted = []
    for s in succ:
        for node in dfs(s, visited):
            top_sorted.append(node)
    return top_sorted if reverse else list(reversed(top_sorted))


def _compute_downstream_costs(model_trace, guide_trace, non_reparam_nodes):
    model_successors = _identify_dense_edges(model_trace)
    guide_successors = _identify_dense_edges(guide_trace)
    # recursively compute downstream cost nodes for all sample sites in model and guide
    # (even though ultimately just need for non-reparameterizable sample sites)
    # 1. downstream costs used for rao-blackwellization
    # 2. model observe sites (as well as terms that arise from the model and guide having different
    # dependency structures) are taken care of via 'children_in_model' below
    topo_sort_guide_nodes = _topological_sort(guide_successors, reverse=True)
    topo_sort_guide_nodes = [
        x for x in topo_sort_guide_nodes if guide_trace[x]["type"] == "sample"
    ]
    ordered_guide_nodes_dict = {n: i for i, n in enumerate(topo_sort_guide_nodes)}

    downstream_guide_cost_nodes = {}
    downstream_costs = {}
    stacks = _get_plate_stacks(model_trace)

    for node in topo_sort_guide_nodes:
        downstream_costs[node] = MultiFrameTensor(
            (
                stacks[node],
                model_trace[node]["log_prob"] - guide_trace[node]["log_prob"],
            )
        )
        nodes_included_in_sum = set([node])
        downstream_guide_cost_nodes[node] = set([node])
        # make more efficient by ordering children appropriately (higher children first)
        children = [(k, -ordered_guide_nodes_dict[k]) for k in guide_successors[node]]
        sorted_children = sorted(children, key=itemgetter(1))
        for child, _ in sorted_children:
            child_cost_nodes = downstream_guide_cost_nodes[child]
            downstream_guide_cost_nodes[node].update(child_cost_nodes)
            if nodes_included_in_sum.isdisjoint(child_cost_nodes):  # avoid duplicates
                downstream_costs[node].add(*downstream_costs[child].items())
                # XXX nodes_included_in_sum logic could be more fine-grained, possibly leading
                # to speed-ups in case there are many duplicates
                nodes_included_in_sum.update(child_cost_nodes)
        missing_downstream_costs = (
            downstream_guide_cost_nodes[node] - nodes_included_in_sum
        )
        # include terms we missed because we had to avoid duplicates
        for missing_node in missing_downstream_costs:
            downstream_costs[node].add(
                (
                    stacks[missing_node],
                    model_trace[missing_node]["log_prob"]
                    - guide_trace[missing_node]["log_prob"],
                )
            )

    # finish assembling complete downstream costs
    # (the above computation may be missing terms from model)
    for site in non_reparam_nodes:
        children_in_model = set()
        for node in downstream_guide_cost_nodes[site]:
            children_in_model.update(model_successors[node])
        # remove terms accounted for above
        children_in_model.difference_update(downstream_guide_cost_nodes[site])
        for child in children_in_model:
            assert model_trace[child]["type"] == "sample"
            downstream_costs[site].add((stacks[child], model_trace[child]["log_prob"]))
            downstream_guide_cost_nodes[site].update([child])

    for k in non_reparam_nodes:
        downstream_costs[k] = downstream_costs[k].sum_to(
            guide_trace[k]["cond_indep_stack"]
        )

    return downstream_costs, downstream_guide_cost_nodes


class track_nonreparam(Messenger):
    """
    Track non-reparameterizable sample sites. Intended to be used with ``eval_provenance``.

    **References:**

    1. *Nonstandard Interpretations of Probabilistic Programs for Efficient Inference*,
        David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind

    **Example:**

    .. doctest::

       >>> import jax.numpy as jnp
       >>> import numpyro
       >>> import numpyro.distributions as dist
       >>> from numpyro.infer.elbo import track_nonreparam
       >>> from numpyro.ops.provenance import eval_provenance, get_provenance
       >>> from numpyro.handlers import seed, trace

       >>> def model():
       ...     probs_a = jnp.array([0.3, 0.7])
       ...     probs_b = jnp.array([[0.1, 0.9], [0.8, 0.2]])
       ...     probs_c = jnp.array([[0.5, 0.5], [0.6, 0.4]])
       ...     a = numpyro.sample("a", dist.Categorical(probs_a))
       ...     b = numpyro.sample("b", dist.Categorical(probs_b[a]))
       ...     numpyro.sample("c", dist.Categorical(probs_c[b]), obs=jnp.array(0))

       >>> def get_log_probs():
       ...     seeded_model = seed(model, rng_seed=0)
       ...     model_tr = trace(seeded_model).get_trace()
       ...     return {
       ...         name: site["fn"].log_prob(site["value"])
       ...         for name, site in model_tr.items()
       ...         if site["type"] == "sample"
       ...     }

       >>> model_deps = get_provenance(eval_provenance(track_nonreparam(get_log_probs)))
       >>> print(model_deps)  # doctest: +SKIP
       {'a': frozenset({'a'}), 'b': frozenset({'a', 'b'}), 'c': frozenset({'a', 'b'})}
    """

    def postprocess_message(self, msg):
        if (
            msg["type"] == "sample"
            and (not msg["is_observed"])
            and (not msg["fn"].has_rsample)
        ):
            new_provenance = frozenset({msg["name"]})
            old_provenance = msg["value"].aval.named_shape.get(
                "_provenance", frozenset()
            )
            msg["value"].aval.named_shape["_provenance"] = (
                old_provenance | new_provenance
            )


def get_importance_log_probs(model, guide, args, kwargs, params):
    """
    Returns log probabilities at each site for the guide and the model that is run against it.
    """
    model_tr, guide_tr = get_importance_trace(model, guide, args, kwargs, params)
    model_log_probs = {
        name: site["log_prob"]
        for name, site in model_tr.items()
        if site["type"] == "sample"
    }
    guide_log_probs = {
        name: site["log_prob"]
        for name, site in guide_tr.items()
        if site["type"] == "sample"
    }
    return model_log_probs, guide_log_probs


class TraceGraph_ELBO(ELBO):
    """
    A TraceGraph implementation of ELBO-based SVI. The gradient estimator
    is constructed along the lines of reference [1] specialized to the case
    of the ELBO. It supports arbitrary dependency structure for the model
    and guide.
    Fine-grained conditional dependency information as recorded in the
    trace is used to reduce the variance of the gradient estimator.
    In particular provenance tracking [2] is used to find the ``cost`` terms
    that depend on each non-reparameterizable sample site.

    References

    [1] `Gradient Estimation Using Stochastic Computation Graphs`,
        John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    [2] `Nonstandard Interpretations of Probabilistic Programs for Efficient Inference`,
        David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind
    """

    can_infer_discrete = True

    def __init__(self, num_particles=1):
        super().__init__(num_particles=num_particles)

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param dict param_map: dictionary of current parameter values keyed by site
            name.
        :param model: Python callable with NumPyro primitives for the model.
        :param guide: Python callable with NumPyro primitives for the guide.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: negative of the Evidence Lower Bound (ELBO) to be minimized.
        """

        def single_particle_elbo(rng_key):
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            model_trace, guide_trace = get_importance_trace(
                seeded_model, seeded_guide, args, kwargs, param_map
            )
            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="strict")

            # Find dependencies on non-reparameterizable sample sites for
            # each cost term in the model and the guide.
            model_deps, guide_deps = get_provenance(
                eval_provenance(
                    partial(
                        track_nonreparam(get_importance_log_probs),
                        seeded_model,
                        seeded_guide,
                        args,
                        kwargs,
                        param_map,
                    )
                )
            )

            elbo = 0.0
            # mapping from non-reparameterizable sample sites to cost terms influenced by each of them
            downstream_costs = defaultdict(lambda: MultiFrameTensor())
            for name, site in model_trace.items():
                if site["type"] == "sample":
                    elbo = elbo + jnp.sum(site["log_prob"])
                    # add the log_prob to each non-reparam sample site upstream
                    for key in model_deps[name]:
                        downstream_costs[key].add(
                            (site["cond_indep_stack"], site["log_prob"])
                        )
            for name, site in guide_trace.items():
                if site["type"] == "sample":
                    log_prob_sum = jnp.sum(site["log_prob"])
                    if not site["fn"].has_rsample:
                        log_prob_sum = stop_gradient(log_prob_sum)
                    elbo = elbo - log_prob_sum
                    # add the -log_prob to each non-reparam sample site upstream
                    for key in guide_deps[name]:
                        downstream_costs[key].add(
                            (site["cond_indep_stack"], -site["log_prob"])
                        )

            for node, downstream_cost in downstream_costs.items():
                guide_site = guide_trace[node]
                downstream_cost = downstream_cost.sum_to(guide_site["cond_indep_stack"])
                surrogate = jnp.sum(
                    guide_site["log_prob"] * stop_gradient(downstream_cost)
                )
                elbo = elbo + surrogate - stop_gradient(surrogate)

            return elbo

        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        if self.num_particles == 1:
            return -single_particle_elbo(rng_key)
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            return -jnp.mean(vmap(single_particle_elbo)(rng_keys))
