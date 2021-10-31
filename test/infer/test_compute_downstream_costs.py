# Copyright Contributors to the Pyro project.
# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

from numpy.testing import assert_allclose
import pytest

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer.elbo import (
    MultiFrameTensor,
    _compute_downstream_costs,
    _get_plate_stacks,
    _identify_dense_edges,
)


def _brute_force_compute_downstream_costs(
    model_trace, guide_trace, non_reparam_nodes  #
):

    model_successors = _identify_dense_edges(model_trace)
    guide_successors = _identify_dense_edges(guide_trace)
    guide_nodes = [x for x in guide_trace if guide_trace[x]["type"] == "sample"]
    downstream_costs, downstream_guide_cost_nodes = {}, {}
    stacks = _get_plate_stacks(model_trace)

    for node in guide_nodes:
        downstream_costs[node] = MultiFrameTensor(
            (
                stacks[node],
                model_trace[node]["log_prob"] - guide_trace[node]["log_prob"],
            )
        )
        downstream_guide_cost_nodes[node] = set([node])

        descendants = guide_successors[node]

        for desc in descendants:
            desc_mft = MultiFrameTensor(
                (
                    stacks[desc],
                    model_trace[desc]["log_prob"] - guide_trace[desc]["log_prob"],
                )
            )
            downstream_costs[node].add(*desc_mft.items())
            downstream_guide_cost_nodes[node].update([desc])

    for site in non_reparam_nodes:
        children_in_model = set()
        for node in downstream_guide_cost_nodes[site]:
            children_in_model.update(model_successors[node])
        children_in_model.difference_update(downstream_guide_cost_nodes[site])
        for child in children_in_model:
            assert model_trace[child]["type"] == "sample"
            child_mft = MultiFrameTensor(
                (stacks[child], model_trace[child]["log_prob"])
            )
            downstream_costs[site].add(*child_mft.items())
            downstream_guide_cost_nodes[site].update([child])

    for k in non_reparam_nodes:
        downstream_costs[k] = downstream_costs[k].sum_to(
            guide_trace[k]["cond_indep_stack"]
        )

    return downstream_costs, downstream_guide_cost_nodes


def big_model_guide(
    include_obs=True,
    include_single=False,
    include_inner_1=False,
    flip_c23=False,
    include_triple=False,
    include_z1=False,
):
    p0 = math.exp(-0.20)
    p1 = math.exp(-0.33)
    p2 = math.exp(-0.70)
    if include_triple:
        with numpyro.plate("plate_triple1", 6) as ind_triple1:
            with numpyro.plate("plate_triple2", 7) as ind_triple2:
                if include_z1:
                    numpyro.sample(
                        "z1",
                        dist.Bernoulli(p2).expand_by(
                            [len(ind_triple2), len(ind_triple1)]
                        ),
                    )
                with numpyro.plate("plate_triple3", 9) as ind_triple3:
                    numpyro.sample(
                        "z0",
                        dist.Bernoulli(p2).expand_by(
                            [len(ind_triple3), len(ind_triple2), len(ind_triple1)]
                        ),
                    )
    numpyro.sample("a1", dist.Bernoulli(p0))
    if include_single:
        with numpyro.plate("plate_single", 5) as ind_single:
            b0 = numpyro.sample("b0", dist.Bernoulli(p0).expand_by([len(ind_single)]))
            assert b0.shape == (5,)
    with numpyro.plate("plate_outer", 2) as ind_outer:
        numpyro.sample("b1", dist.Bernoulli(p0).expand_by([len(ind_outer)]))
        if include_inner_1:
            with numpyro.plate("plate_inner_1", 3) as ind_inner:
                numpyro.sample(
                    "c1", dist.Bernoulli(p1).expand_by([len(ind_inner), len(ind_outer)])
                )
                if flip_c23 and not include_obs:
                    numpyro.sample(
                        "c3",
                        dist.Bernoulli(p0).expand_by([len(ind_inner), len(ind_outer)]),
                    )
                    numpyro.sample(
                        "c2",
                        dist.Bernoulli(p1).expand_by([len(ind_inner), len(ind_outer)]),
                    )
                else:
                    numpyro.sample(
                        "c2",
                        dist.Bernoulli(p0).expand_by([len(ind_inner), len(ind_outer)]),
                    )
                    numpyro.sample(
                        "c3",
                        dist.Bernoulli(p2).expand_by([len(ind_inner), len(ind_outer)]),
                    )
        with numpyro.plate("plate_inner_2", 4) as ind_inner:
            numpyro.sample(
                "d1", dist.Bernoulli(p0).expand_by([len(ind_inner), len(ind_outer)])
            )
            d2 = numpyro.sample(
                "d2", dist.Bernoulli(p2).expand_by([len(ind_inner), len(ind_outer)])
            )
            assert d2.shape == (4, 2)
            if include_obs:
                numpyro.sample(
                    "obs",
                    dist.Bernoulli(p0).expand_by([len(ind_inner), len(ind_outer)]),
                    obs=jnp.ones(d2.shape),
                )


@pytest.mark.parametrize("include_inner_1", [True, False])
@pytest.mark.parametrize("include_single", [True, False])
@pytest.mark.parametrize("flip_c23", [True, False])
@pytest.mark.parametrize("include_triple", [True, False])
@pytest.mark.parametrize("include_z1", [True, False])
def test_compute_downstream_costs_big_model_guide_pair(
    include_inner_1, include_single, flip_c23, include_triple, include_z1
):
    seeded_guide = handlers.seed(big_model_guide, rng_seed=0)
    guide_trace = handlers.trace(seeded_guide).get_trace(
        include_obs=False,
        include_inner_1=include_inner_1,
        include_single=include_single,
        flip_c23=flip_c23,
        include_triple=include_triple,
        include_z1=include_z1,
    )
    model_trace = handlers.trace(handlers.replay(seeded_guide, guide_trace)).get_trace(
        include_obs=True,
        include_inner_1=include_inner_1,
        include_single=include_single,
        flip_c23=flip_c23,
        include_triple=include_triple,
        include_z1=include_z1,
    )

    for trace in (model_trace, guide_trace):
        for site in trace.values():
            if site["type"] == "sample":
                site["log_prob"] = site["fn"].log_prob(site["value"])
    non_reparam_nodes = set(
        name
        for name, site in guide_trace.items()
        if site["type"] == "sample"
        and (site["is_observed"] or not site["fn"].has_rsample)
    )

    dc, dc_nodes = _compute_downstream_costs(
        model_trace, guide_trace, non_reparam_nodes
    )

    dc_brute, dc_nodes_brute = _brute_force_compute_downstream_costs(
        model_trace, guide_trace, non_reparam_nodes
    )

    assert dc_nodes == dc_nodes_brute

    expected_nodes_full_model = {
        "a1": {"c2", "a1", "d1", "c1", "obs", "b1", "d2", "c3", "b0"},
        "d2": {"obs", "d2"},
        "d1": {"obs", "d1", "d2"},
        "c3": {"d2", "obs", "d1", "c3"},
        "b0": {"b0", "d1", "c1", "obs", "b1", "d2", "c3", "c2"},
        "b1": {"obs", "b1", "d1", "d2", "c3", "c1", "c2"},
        "c1": {"d1", "c1", "obs", "d2", "c3", "c2"},
        "c2": {"obs", "d1", "c3", "d2", "c2"},
    }
    if not include_triple and include_inner_1 and include_single and not flip_c23:
        assert dc_nodes == expected_nodes_full_model

    expected_b1 = model_trace["b1"]["log_prob"] - guide_trace["b1"]["log_prob"]
    expected_b1 += (model_trace["d2"]["log_prob"] - guide_trace["d2"]["log_prob"]).sum(
        0
    )
    expected_b1 += (model_trace["d1"]["log_prob"] - guide_trace["d1"]["log_prob"]).sum(
        0
    )
    expected_b1 += model_trace["obs"]["log_prob"].sum(0, keepdims=False)
    if include_inner_1:
        expected_b1 += (
            model_trace["c1"]["log_prob"] - guide_trace["c1"]["log_prob"]
        ).sum(0)
        expected_b1 += (
            model_trace["c2"]["log_prob"] - guide_trace["c2"]["log_prob"]
        ).sum(0)
        expected_b1 += (
            model_trace["c3"]["log_prob"] - guide_trace["c3"]["log_prob"]
        ).sum(0)
    assert_allclose(expected_b1, dc["b1"], atol=1.0e-6)

    if include_single:
        expected_b0 = model_trace["b0"]["log_prob"] - guide_trace["b0"]["log_prob"]
        expected_b0 += (
            model_trace["b1"]["log_prob"] - guide_trace["b1"]["log_prob"]
        ).sum()
        expected_b0 += (
            model_trace["d2"]["log_prob"] - guide_trace["d2"]["log_prob"]
        ).sum()
        expected_b0 += (
            model_trace["d1"]["log_prob"] - guide_trace["d1"]["log_prob"]
        ).sum()
        expected_b0 += model_trace["obs"]["log_prob"].sum()
        if include_inner_1:
            expected_b0 += (
                model_trace["c1"]["log_prob"] - guide_trace["c1"]["log_prob"]
            ).sum()
            expected_b0 += (
                model_trace["c2"]["log_prob"] - guide_trace["c2"]["log_prob"]
            ).sum()
            expected_b0 += (
                model_trace["c3"]["log_prob"] - guide_trace["c3"]["log_prob"]
            ).sum()
        assert_allclose(expected_b0, dc["b0"], atol=1.0e-6)
        assert dc["b0"].shape == (5,)

    if include_inner_1:
        expected_c3 = model_trace["c3"]["log_prob"] - guide_trace["c3"]["log_prob"]
        expected_c3 += (
            model_trace["d1"]["log_prob"] - guide_trace["d1"]["log_prob"]
        ).sum(0)
        expected_c3 += (
            model_trace["d2"]["log_prob"] - guide_trace["d2"]["log_prob"]
        ).sum(0)
        expected_c3 += model_trace["obs"]["log_prob"].sum(0)

        expected_c2 = model_trace["c2"]["log_prob"] - guide_trace["c2"]["log_prob"]
        expected_c2 += (
            model_trace["d1"]["log_prob"] - guide_trace["d1"]["log_prob"]
        ).sum(0)
        expected_c2 += (
            model_trace["d2"]["log_prob"] - guide_trace["d2"]["log_prob"]
        ).sum(0)
        expected_c2 += model_trace["obs"]["log_prob"].sum(0)

        expected_c1 = model_trace["c1"]["log_prob"] - guide_trace["c1"]["log_prob"]

        if flip_c23:
            expected_c3 += model_trace["c2"]["log_prob"] - guide_trace["c2"]["log_prob"]
            expected_c2 += model_trace["c3"]["log_prob"]
        else:
            expected_c2 += model_trace["c3"]["log_prob"] - guide_trace["c3"]["log_prob"]
            expected_c2 += model_trace["c2"]["log_prob"] - guide_trace["c2"]["log_prob"]
        expected_c1 += expected_c3

        assert_allclose(expected_c1, dc["c1"], atol=1.0e-6)
        assert_allclose(expected_c2, dc["c2"], atol=1.0e-6)
        assert_allclose(expected_c3, dc["c3"], atol=1.0e-6)

    expected_d1 = model_trace["d1"]["log_prob"] - guide_trace["d1"]["log_prob"]
    expected_d1 += model_trace["d2"]["log_prob"] - guide_trace["d2"]["log_prob"]
    expected_d1 += model_trace["obs"]["log_prob"]

    expected_d2 = model_trace["d2"]["log_prob"] - guide_trace["d2"]["log_prob"]
    expected_d2 += model_trace["obs"]["log_prob"]

    if include_triple:
        expected_z0 = (
            dc["a1"] + model_trace["z0"]["log_prob"] - guide_trace["z0"]["log_prob"]
        )
        assert_allclose(expected_z0, dc["z0"], atol=1.0e-6)
    assert_allclose(expected_d2, dc["d2"], atol=1.0e-6)
    assert_allclose(expected_d1, dc["d1"], atol=1.0e-6)

    assert dc["b1"].shape == (2,)
    assert dc["d2"].shape == (4, 2)

    for k in dc:
        assert guide_trace[k]["log_prob"].shape == dc[k].shape
        assert_allclose(dc[k], dc_brute[k], rtol=2e-7)


def plate_reuse_model_guide(include_obs=True, dim1=3, dim2=2):
    p0 = math.exp(-0.40 - include_obs * 0.2)
    p1 = math.exp(-0.33 - include_obs * 0.1)
    numpyro.sample("a1", dist.Bernoulli(p0 * p1))
    my_plate1 = numpyro.plate("plate1", dim1, dim=-1)
    my_plate2 = numpyro.plate("plate2", dim2, dim=-2)
    with my_plate1 as ind1:
        with my_plate2 as ind2:
            numpyro.sample("c1", dist.Bernoulli(p1).expand_by([len(ind2), len(ind1)]))
    numpyro.sample("b1", dist.Bernoulli(p0 * p1))
    with my_plate2 as ind2:
        with my_plate1 as ind1:
            c2 = numpyro.sample(
                "c2", dist.Bernoulli(p1).expand_by([len(ind2), len(ind1)])
            )
            if include_obs:
                numpyro.sample("obs", dist.Bernoulli(c2), obs=jnp.ones(c2.shape))


@pytest.mark.parametrize("dim1", [2, 5])
@pytest.mark.parametrize("dim2", [3, 4])
def test_compute_downstream_costs_plate_reuse(dim1, dim2):
    seeded_guide = handlers.seed(plate_reuse_model_guide, rng_seed=0)
    guide_trace = handlers.trace(seeded_guide).get_trace(
        include_obs=False, dim1=dim1, dim2=dim2
    )
    model_trace = handlers.trace(handlers.replay(seeded_guide, guide_trace)).get_trace(
        include_obs=True, dim1=dim1, dim2=dim2
    )

    for trace in (model_trace, guide_trace):
        for site in trace.values():
            if site["type"] == "sample":
                site["log_prob"] = site["fn"].log_prob(site["value"])
    non_reparam_nodes = set(
        name
        for name, site in guide_trace.items()
        if site["type"] == "sample"
        and (site["is_observed"] or not site["fn"].has_rsample)
    )

    dc, dc_nodes = _compute_downstream_costs(
        model_trace, guide_trace, non_reparam_nodes
    )

    dc_brute, dc_nodes_brute = _brute_force_compute_downstream_costs(
        model_trace, guide_trace, non_reparam_nodes
    )

    assert dc_nodes == dc_nodes_brute

    for k in dc:
        assert guide_trace[k]["log_prob"].shape == dc[k].shape
        assert_allclose(dc[k], dc_brute[k], rtol=1e-6)

    expected_c1 = model_trace["c1"]["log_prob"] - guide_trace["c1"]["log_prob"]
    expected_c1 += (model_trace["b1"]["log_prob"] - guide_trace["b1"]["log_prob"]).sum()
    expected_c1 += model_trace["c2"]["log_prob"] - guide_trace["c2"]["log_prob"]
    expected_c1 += model_trace["obs"]["log_prob"]
    assert_allclose(expected_c1, dc["c1"], rtol=1e-6)
