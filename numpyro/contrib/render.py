# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
from pathlib import Path

import jax

from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_sample
from numpyro.ops.provenance import ProvenanceArray, eval_provenance, get_provenance
from numpyro.ops.pytree import PytreeTrace


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
        if site["type"] == "sample"
    }

    sample_plates = {
        name: [frame.name for frame in site["cond_indep_stack"]]
        for name, site in trace.items()
        if site["type"] == "sample"
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

    def get_log_probs(sample):
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

    samples = {
        name: ProvenanceArray(site["value"], frozenset({name}))
        for name, site in trace.items()
        if (site["type"] == "sample" and not site["is_observed"])
    }

    params = {
        name: jax.tree_util.tree_map(
            lambda x: ProvenanceArray(x, frozenset({name})), site["value"]
        )
        for name, site in trace.items()
        if site["type"] == "param"
    }

    sample_and_params = {**samples, **params}
    sample_params_deps = get_provenance(
        eval_provenance(get_log_probs, sample_and_params)
    )

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

            cur_graph.node(
                rv, label=rv_label, shape=shape, style="filled", fillcolor=color
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
