# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial
import itertools
from pathlib import Path
from typing import Optional

import jax

from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_sample
from numpyro.ops.provenance import eval_provenance
from numpyro.ops.pytree import PytreeTrace


def is_sample_site(msg):
    if msg["type"] != "sample":
        return False

    # Exclude deterministic sites.
    if msg["fn_name"] == "Delta":
        return False

    return True


def _get_dist_name(fn):
    if isinstance(
        fn, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
    ):
        return _get_dist_name(fn.base_dist)
    return type(fn).__name__


def _get_abstract_trace(model, model_args, model_kwargs):
    def get_trace():
        # We use `init_to_sample` to get around ImproperUniform distribution,
        # which does not have `sample` method.
        subs_model = handlers.substitute(
            handlers.seed(model, 0),
            substitute_fn=init_to_sample,
        )
        trace = handlers.trace(subs_model).get_trace(*model_args, **model_kwargs)
        # Work around an issue where jax.eval_shape does not work
        # for distribution output (e.g. the function `lambda: dist.Normal(0, 1)`)
        # Here we will remove `fn` and store its name in the trace.
        for site in trace.values():
            if site["type"] == "sample":
                site["fn_name"] = _get_dist_name(site.pop("fn"))
        return PytreeTrace(trace)

    # We use eval_shape to avoid any array computation.
    return jax.eval_shape(get_trace).trace


def _get_log_probs(model, model_args, model_kwargs, **sample):
    # Note: We use seed 0 for parameter initialization.
    with handlers.trace() as tr, handlers.seed(rng_seed=0), handlers.substitute(
        data=sample
    ):
        model(*model_args, **model_kwargs)
    return {
        name: site["fn"].log_prob(site["value"])
        for name, site in tr.items()
        if site["type"] == "sample"
    }


def get_dependencies(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
) -> dict[str, object]:
    r"""
    Infers dependency structure about a conditioned model.

    This returns a nested dictionary with structure like::

        {
            "prior_dependencies": {
                "variable1": {"variable1": set()},
                "variable2": {"variable1": set(), "variable2": set()},
                ...
            },
            "posterior_dependencies": {
                "variable1": {"variable1": {"plate1"}, "variable2": set()},
                ...
            },
        }

    where

    -   `prior_dependencies` is a dict mapping downstream latent and observed
        variables to dictionaries mapping upstream latent variables on which
        they depend to sets of plates inducing full dependencies.
        That is, included plates introduce quadratically many dependencies as
        in complete-bipartite graphs, whereas excluded plates introduce only
        linearly many dependencies as in independent sets of parallel edges.
        Prior dependencies follow the original model order.
    -   `posterior_dependencies` is a similar dict, but mapping latent
        variables to the latent or observed sits on which they depend in the
        posterior. Posterior dependencies are reversed from the model order.

    Dependencies elide ``numpyro.deterministic`` sites and ``numpyro.sample(...,
    Delta(...))`` sites.

    **Examples**

    Here is a simple example with no plates. We see every node depends on
    itself, and only the latent variables appear in the posterior::

        def model_1():
            a = numpyro.sample("a", dist.Normal(0, 1))
            numpyro.sample("b", dist.Normal(a, 1), obs=0.0)

        assert get_dependencies(model_1) == {
            "prior_dependencies": {
                "a": {"a": set()},
                "b": {"a": set(), "b": set()},
            },
            "posterior_dependencies": {
                "a": {"a": set(), "b": set()},
            },
        }

    Here is an example where two variables ``a`` and ``b`` start out
    conditionally independent in the prior, but become conditionally dependent
    in the posterior do the so-called collider variable ``c`` on which they
    both depend. This is called "moralization" in the graphical model
    literature::

        def model_2():
            a = numpyro.sample("a", dist.Normal(0, 1))
            b = numpyro.sample("b", dist.LogNormal(0, 1))
            c = numpyro.sample("c", dist.Normal(a, b))
            numpyro.sample("d", dist.Normal(c, 1), obs=0.)

        assert get_dependencies(model_2) == {
            "prior_dependencies": {
                "a": {"a": set()},
                "b": {"b": set()},
                "c": {"a": set(), "b": set(), "c": set()},
                "d": {"c": set(), "d": set()},
            },
            "posterior_dependencies": {
                "a": {"a": set(), "b": set(), "c": set()},
                "b": {"b": set(), "c": set()},
                "c": {"c": set(), "d": set()},
            },
        }

    Dependencies can be more complex in the presence of plates. So far all the
    dict values have been empty sets of plates, but in the following posterior
    we see that ``c`` depends on itself across the plate ``p``. This means
    that, among the elements of ``c``, e.g. ``c[0]`` depends on ``c[1]`` (this
    is why we explicitly allow variables to depend on themselves)::

        def model_3():
            with numpyro.plate("p", 5):
                a = numpyro.sample("a", dist.Normal(0, 1))
            numpyro.sample("b", dist.Normal(a.sum(), 1), obs=0.0)

        assert get_dependencies(model_3) == {
            "prior_dependencies": {
                "a": {"a": set()},
                "b": {"a": set(), "b": set()},
            },
            "posterior_dependencies": {
                "a": {"a": {"p"}, "b": set()},
            },
        }

    [1] S.Webb, A.Goliński, R.Zinkov, N.Siddharth, T.Rainforth, Y.W.Teh, F.Wood (2018)
        "Faithful inversion of generative models for effective amortized inference"
        https://dl.acm.org/doi/10.5555/3327144.3327229

    :param callable model: A model.
    :param tuple model_args: Optional tuple of model args.
    :param dict model_kwargs: Optional dict of model kwargs.
    :returns: A dictionary of metadata (see above).
    :rtype: dict
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}

    # Collect sites with tracked provenance.
    trace = _get_abstract_trace(model, model_args, model_kwargs)
    sample_sites = [msg for msg in trace.values() if is_sample_site(msg)]

    # Collect observations.
    observed = {msg["name"] for msg in sample_sites if msg["is_observed"]}
    plates = {
        msg["name"]: {f.name for f in msg["cond_indep_stack"]} for msg in sample_sites
    }

    # Find direct prior dependencies among latent and observed sites.
    samples = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    sample_deps = eval_provenance(
        partial(_get_log_probs, model, model_args, model_kwargs), **samples
    )
    prior_dependencies = {n: {n: set()} for n in plates}  # no deps yet
    for i, downstream in enumerate(sample_sites):
        upstreams = [
            u
            for u in sample_sites[:i]
            if not u["is_observed"]
            if u["fn_name"] != "Unit"
        ]
        if not upstreams:
            continue
        provenance = sample_deps[downstream["name"]]
        for upstream in upstreams:
            u = upstream["name"]
            if u in provenance:
                d = downstream["name"]
                prior_dependencies[d][u] = set()

    # Next reverse dependencies and restrict downstream nodes to latent sites.
    posterior_dependencies = {n: {} for n in plates if n not in observed}
    for d, upstreams in prior_dependencies.items():
        for u, p in upstreams.items():
            if u not in observed:
                # Note the folowing reverses:
                # u is henceforth downstream and d is henceforth upstream.
                posterior_dependencies[u][d] = p.copy()

    # Moralize: add dependencies among latent variables in each Markov blanket.
    # This assumes all latents are eventually observed, at least indirectly.
    order = {msg["name"]: i for i, msg in enumerate(reversed(sample_sites))}
    for d, upstreams in prior_dependencies.items():
        upstreams = {u: p for u, p in upstreams.items() if u not in observed}
        for u1, p1 in upstreams.items():
            for u2, p2 in upstreams.items():
                if order[u1] <= order[u2]:
                    p12 = posterior_dependencies[u2].setdefault(u1, set())
                    p12 |= plates[u1] & plates[u2] - plates[d]
                    p12 |= plates[u2] & p1
                    p12 |= plates[u1] & p2

    return {
        "prior_dependencies": prior_dependencies,
        "posterior_dependencies": posterior_dependencies,
    }


def get_model_relations(model, model_args=None, model_kwargs=None):
    """
    Infer relations of RVs and plates from given model and optionally data.
    See https://github.com/pyro-ppl/numpyro/issues/949 for more details.

    This returns a dictionary with keys:

    -  "sample_sample" map each downstream sample site to a list of the upstream
       sample sites on which it depend;
    -  "sample_param" map each downstream sample site to a list of the upstream
       param sites on which it depend;
    -  "sample_dist" maps each sample site to the name of the distribution at
       that site;
    -  "param_constraint" maps each param site to the name of the constraints at
       that site;
    -  "plate_sample" maps each plate name to a lists of the sample sites
       within that plate; and
    -  "observe" is a list of observed sample sites.

    For example for the model::

        def model(data):
            m = numpyro.sample('m', dist.Normal(0, 1))
            sd = numpyro.sample('sd', dist.LogNormal(m, 1))
            with numpyro.plate('N', len(data)):
                numpyro.sample('obs', dist.Normal(m, sd), obs=data)

    the relation is::

        {'sample_sample': {'m': [], 'sd': ['m'], 'obs': ['m', 'sd']},
         'sample_dist': {'m': 'Normal', 'sd': 'LogNormal', 'obs': 'Normal'},
         'plate_sample': {'N': ['obs']},
         'observed': ['obs']}

    :param callable model: A model to inspect.
    :param model_args: Optional tuple of model args.
    :param model_kwargs: Optional dict of model kwargs.
    :rtype: dict
    """
    model_args = model_args or ()
    model_kwargs = model_kwargs or {}

    def _get_dist_name(fn):
        if isinstance(
            fn, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
        ):
            return _get_dist_name(fn.base_dist)
        return type(fn).__name__

    def get_trace():
        # We use `init_to_sample` to get around ImproperUniform distribution,
        # which does not have `sample` method.
        subs_model = handlers.substitute(
            handlers.seed(model, 0),
            substitute_fn=init_to_sample,
        )
        trace = handlers.trace(subs_model).get_trace(*model_args, **model_kwargs)
        # Work around an issue where jax.eval_shape does not work
        # for distribution output (e.g. the function `lambda: dist.Normal(0, 1)`)
        # Here we will remove `fn` and store its name in the trace.
        for name, site in trace.items():
            if site["type"] == "sample":
                site["fn_name"] = _get_dist_name(site.pop("fn"))
            elif site["type"] == "deterministic":
                site["fn_name"] = "Deterministic"
        return PytreeTrace(trace)

    # We use eval_shape to avoid any array computation.
    trace = jax.eval_shape(get_trace).trace
    obs_sites = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and site["is_observed"]
    ]
    sample_dist = {
        name: site["fn_name"]
        for name, site in trace.items()
        if site["type"] in ["sample", "deterministic"]
    }

    sample_plates = {
        name: [frame.name for frame in site["cond_indep_stack"]]
        for name, site in trace.items()
        if site["type"] in ["sample", "deterministic"]
    }
    plate_samples = {
        k: {name for name, plates in sample_plates.items() if k in plates}
        for k in trace
        if trace[k]["type"] == "plate"
    }

    def _resolve_plate_samples(plate_samples):
        for p, pv in plate_samples.items():
            for q, qv in plate_samples.items():
                if len(pv & qv) > 0 and len(pv - qv) > 0 and len(qv - pv) > 0:
                    plate_samples_ = plate_samples.copy()
                    plate_samples_[q] = pv & qv
                    plate_samples_[q + "__CLONE"] = qv - pv
                    return _resolve_plate_samples(plate_samples_)
        return plate_samples

    plate_samples = _resolve_plate_samples(plate_samples)
    # convert set to list to keep order of variables
    plate_samples = {
        k: [name for name in trace if name in v] for k, v in plate_samples.items()
    }

    def get_log_probs(**sample):
        class substitute_deterministic(handlers.substitute):
            def process_message(self, msg):
                if msg["type"] == "deterministic":
                    msg["args"] = (msg["value"],)
                    msg["kwargs"] = {}
                    msg["value"] = self.data.get(msg["name"])
                    msg["fn"] = lambda x: x

        # Note: We use seed 0 for parameter initialization.
        with handlers.trace() as tr, handlers.seed(rng_seed=0):
            with handlers.substitute(data=sample), substitute_deterministic(
                data=sample
            ):
                model(*model_args, **model_kwargs)
        provenance_arrays = {}
        for name, site in tr.items():
            if site["type"] == "sample":
                provenance_arrays[name] = site["fn"].log_prob(site["value"])
            elif site["type"] == "deterministic":
                provenance_arrays[name] = site["args"][0]
        return provenance_arrays

    samples = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" or site["type"] == "deterministic"
    }

    params = {
        name: site["value"] for name, site in trace.items() if site["type"] == "param"
    }

    sample_params_deps = eval_provenance(get_log_probs, **samples, **params)

    sample_sample = {}
    sample_param = {}
    for name in sample_dist:
        sample_sample[name] = [
            var
            for var in sample_dist
            if var in sample_params_deps[name] and var != name
        ]
        sample_param[name] = [var for var in sample_params_deps[name] if var in params]

    param_constraint = {}
    for param in params:
        if "constraint" in trace[param]["kwargs"]:
            param_constraint[param] = str(trace[param]["kwargs"]["constraint"])
        else:
            param_constraint[param] = ""

    return {
        "sample_sample": sample_sample,
        "sample_param": sample_param,
        "sample_dist": sample_dist,
        "param_constraint": param_constraint,
        "plate_sample": plate_samples,
        "observed": obs_sites,
    }


def generate_graph_specification(model_relations, render_params=False):
    """
    Convert model relations into data structure which can be readily
    converted into a network.

    :param bool render_params: Whether to add nodes of params.
    """
    # group nodes by plate
    plate_groups = dict(model_relations["plate_sample"])
    plate_rvs = {rv for rvs in plate_groups.values() for rv in rvs}
    plate_groups[None] = [
        rv for rv in model_relations["sample_sample"] if rv not in plate_rvs
    ]  # RVs which are in no plate

    # get set of params
    params = set()
    if render_params:
        for rv, params_list in model_relations["sample_param"].items():
            for param in params_list:
                params.add(param)
        plate_groups[None].extend(params)

    # retain node metadata
    node_data = {}
    for rv in model_relations["sample_sample"]:
        node_data[rv] = {
            "is_observed": rv in model_relations["observed"],
            "distribution": model_relations["sample_dist"][rv],
        }

    if render_params:
        for param, constraint in model_relations["param_constraint"].items():
            node_data[param] = {
                "is_observed": False,
                "constraint": constraint,
                "distribution": None,
            }

    # infer plate structure
    # (when the order of plates cannot be determined from subset relations,
    # it follows the order in which plates appear in trace)
    plate_data = {}
    for plate1, plate2 in list(itertools.combinations(plate_groups, 2)):
        if plate1 is None or plate2 is None:
            continue

        if set(plate_groups[plate1]) < set(plate_groups[plate2]):
            plate_data[plate1] = {"parent": plate2}
        elif set(plate_groups[plate1]) >= set(plate_groups[plate2]):
            plate_data[plate2] = {"parent": plate1}

    for plate in plate_groups:
        if plate is None:
            continue

        if plate not in plate_data:
            plate_data[plate] = {"parent": None}

    # infer RV edges
    edge_list = []
    for target, source_list in model_relations["sample_sample"].items():
        edge_list.extend([(source, target) for source in source_list])

    if render_params:
        for target, source_list in model_relations["sample_param"].items():
            edge_list.extend([(source, target) for source in source_list])

    return {
        "plate_groups": plate_groups,
        "plate_data": plate_data,
        "node_data": node_data,
        "edge_list": edge_list,
    }


def render_graph(graph_specification, render_distributions=False):
    """
    Create a graphviz object given a graph specification.

    :param bool render_distributions: Show distribution of each RV in plot.
    """
    try:
        import graphviz  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Looks like you want to use graphviz (https://graphviz.org/) "
            "to render your model. "
            "You need to install `graphviz` to be able to use this feature. "
            "It can be installed with `pip install graphviz`."
        ) from e

    plate_groups = graph_specification["plate_groups"]
    plate_data = graph_specification["plate_data"]
    node_data = graph_specification["node_data"]
    edge_list = graph_specification["edge_list"]

    graph = graphviz.Digraph()

    # add plates
    plate_graph_dict = {
        plate: graphviz.Digraph(name=f"cluster_{plate}")
        for plate in plate_groups
        if plate is not None
    }
    for plate, plate_graph in plate_graph_dict.items():
        plate_graph.attr(label=plate.split("__CLONE")[0], labeljust="r", labelloc="b")

    plate_graph_dict[None] = graph

    # add nodes
    for plate, rv_list in plate_groups.items():
        cur_graph = plate_graph_dict[plate]

        for rv in rv_list:
            color = "grey" if node_data[rv]["is_observed"] else "white"

            # For sample_nodes - ellipse
            if node_data[rv]["distribution"]:
                shape = "ellipse"
                rv_label = rv

            # For param_nodes - No shape
            else:
                shape = "plain"
                rv_label = rv.replace(
                    "$params", ""
                )  # incase of neural network parameters

            # use different symbol for Deterministic site
            node_style = (
                "filled,dashed"
                if node_data[rv]["distribution"] == "Deterministic"
                else "filled"
            )
            cur_graph.node(
                rv, label=rv_label, shape=shape, style=node_style, fillcolor=color
            )

    # add leaf nodes first
    while len(plate_data) >= 1:
        for plate, data in plate_data.items():
            parent_plate = data["parent"]
            is_leaf = True

            for plate2, data2 in plate_data.items():
                if plate == data2["parent"]:
                    is_leaf = False
                    break

            if is_leaf:
                plate_graph_dict[parent_plate].subgraph(plate_graph_dict[plate])
                plate_data.pop(plate)
                break

    # add edges
    for source, target in edge_list:
        graph.edge(source, target)

    # render distributions and constraints if requested
    if render_distributions:
        dist_label = ""
        for rv, data in node_data.items():
            rv_dist = data["distribution"]
            if rv_dist:
                dist_label += rf"{rv} ~ {rv_dist}\l"

            if "constraint" in data and data["constraint"]:
                dist_label += rf"{rv} ∈ {data['constraint']}\l"

        graph.node("distribution_description_node", label=dist_label, shape="plaintext")

    # return whole graph
    return graph


def render_model(
    model,
    model_args=None,
    model_kwargs=None,
    filename=None,
    render_distributions=False,
    render_params=False,
):
    """
    Wrap all functions needed to automatically render a model.

    .. warning:: This utility does not support the
        :func:`~numpyro.contrib.control_flow.scan` primitive.
        If you want to render a time-series model, you can try
        to rewrite the code using Python for loop.

    :param model: Model to render.
    :param model_args: Positional arguments to pass to the model.
    :param model_kwargs: Keyword arguments to pass to the model.
    :param str filename: File to save rendered model in.
    :param bool render_distributions: Whether to include RV distribution annotations in the plot.
    :param bool render_params: Whether to show params in the plot.
    """
    relations = get_model_relations(
        model,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    graph_spec = generate_graph_specification(relations, render_params=render_params)
    graph = render_graph(graph_spec, render_distributions=render_distributions)

    if filename is not None:
        filename = Path(filename)
        graph.render(
            filename.stem, view=False, cleanup=True, format=filename.suffix[1:]
        )  # remove leading period from suffix

    return graph


__all__ = [
    "get_dependencies",
    "get_model_relations",
    "render_model",
]
