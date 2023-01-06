# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor
from funsor.adjoint import AdjointTape
from funsor.sum_product import _partition

from numpyro.contrib.funsor import to_data, to_funsor, plate_to_enum_plate
from numpyro.contrib.funsor import enum, plate, trace
from numpyro.handlers import replay as OrigReplayMessenger
from numpyro.handlers import substitute


# Work around a bug in unfold_contraction_generic_tuple interacting with
# Approximate introduced in https://github.com/pyro-ppl/funsor/pull/488 .
# Once fixed, this can be replaced by funsor.optimizer.apply_optimizer().
def apply_optimizer(x):
    with funsor.interpretations.normalize:
        expr = funsor.interpreter.reinterpret(x)

    with funsor.optimizer.optimize_base:
        return funsor.interpreter.reinterpret(expr)


def terms_from_trace(tr, is_guide=False):
    """Helper function to extract elbo components from execution traces."""
    # data structure containing densities, measures, scales, and identification
    # of free variables as either product (plate) variables or sum (measure) variables
    terms = {
        "log_factors": [],
        "log_measures": [],
        "scale": to_funsor(1.0),
        "plate_vars": frozenset(),
        "measure_vars": frozenset(),
        "plate_to_step": dict(),
    }

    for name, node in tr.items():
        if node["type"] != "sample":
            continue
        # TODO fix this
        value = node["value"]
        intermediates = node["intermediates"]
        scale = node["scale"]
        if intermediates:
            log_prob = node["fn"].log_prob(value, intermediates)
        else:
            log_prob = node["fn"].log_prob(value)

        #  if (scale is not None) and (not is_identically_one(scale)):
        #      log_prob = scale * log_prob

        #  if msg["scale"] is not None and "scale" not in msg["funsor"]:
        #      msg["funsor"]["scale"] = to_funsor(msg["scale"], output=funsor.Real)

        dim_to_name = node["infer"]["dim_to_name"]
        # check this
        if "funsor" not in node:
            node["funsor"] = {}
        # if "fn" not in node["funsor"]:
        #     node["funsor"]["fn"] = to_funsor(node["fn"], funsor.Real)(value=node["name"])
        # if "value" not in node["funsor"]:
        #     node["funsor"]["value"] = to_funsor(value, node["funsor"]["fn"].inputs[node["name"]])
        if "log_prob" not in node["funsor"]:
            node["funsor"]["log_prob"] = to_funsor(log_prob, output=funsor.Real, dim_to_name=dim_to_name)

        # grab plate dimensions from the cond_indep_stack
        # TODO: consider if we need `f.dim is not None`
        terms["plate_vars"] |= frozenset(
            f.name for f in node["cond_indep_stack"] if f.dim is not None
        )
        # grab the log-measure, found only at sites that are not observed or replayed
        # if node["funsor"].get("log_measure", None) is not None:
        if not (node["is_observed"] or node.get("is_replayed", False)):
            terms["log_measures"].append(node["funsor"]["log_prob"])
            # terms["log_measures"].append(log_prob_factor)
            # sum (measure) variables: the fresh non-plate variables at a site
            # terms["measure_vars"] |= (
            #     frozenset(node["funsor"]["value"].inputs) | {name}
            # ) - terms["plate_vars"]
            # TODO: reconsider the logic
            terms["measure_vars"] |= frozenset({node["name"]})
        # print("DEBUG log measures", terms["log_measures"], node)
        # grab the scale, assuming a common subsampling scale
        # if (
        #     node.get("replay_active", False)
        #     and set(node["funsor"]["log_prob"].inputs) & terms["measure_vars"]
        #     and float(to_data(node["funsor"]["scale"])) != 1.0
        # ):
        #     # model site that depends on enumerated variable: common scale
        #     terms["scale"] = node["funsor"]["scale"]
        # else:  # otherwise: default scale behavior
        #     node["funsor"]["log_prob"] = (
        #         node["funsor"]["log_prob"] * node["funsor"]["scale"]
        #     )
        # grab the log-density, found at all sites except those that are not replayed
        if node["is_observed"] or node.get("is_replayed", False):
            terms["log_factors"].append(node["funsor"]["log_prob"])
    # add plate dimensions to the plate_to_step dictionary
    terms["plate_to_step"].update(
        {plate: terms["plate_to_step"].get(plate, {}) for plate in terms["plate_vars"]}
    )
    return terms


def traceenum_elbo(params, model, guide, max_plate_nesting, *args, **kwargs):
    # get batched, enumerated, to_funsor-ed traces from the guide and model
    with plate_to_enum_plate(), enum(
        first_available_dim=(-max_plate_nesting - 1)
        if max_plate_nesting
        else None
    ):
        guide = substitute(guide, data=params)
        guide_tr = trace(guide).get_trace(*args, **kwargs)
        model = substitute(replay(model, guide_tr), data=params)
        model_tr = trace(model).get_trace(*args, **kwargs)

    # extract from traces all metadata that we will need to compute the elbo
    guide_terms = terms_from_trace(guide_tr)
    model_terms = terms_from_trace(model_tr)
    print("DEBUG")
    print(f"{guide_tr}")
    print(f"{model_tr}")
    print(f"{guide_terms}")
    print(f"{model_terms}")

    # build up a lazy expression for the elbo
    with funsor.terms.lazy:
        # identify and contract out auxiliary variables in the model with partial_sum_product
        contracted_factors, uncontracted_factors = [], []
        for f in model_terms["log_factors"]:
            if model_terms["measure_vars"].intersection(f.inputs):
                contracted_factors.append(f)
            else:
                uncontracted_factors.append(f)
        contracted_costs = []
        # incorporate the effects of subsampling and handlers.scale through a common scale factor
        for group_factors, group_vars in _partition(
            model_terms["log_measures"] + contracted_factors,
            model_terms["measure_vars"],
        ):
            group_factor_vars = frozenset().union(
                *[f.inputs for f in group_factors]
            )
            group_plates = model_terms["plate_vars"] & group_factor_vars
            outermost_plates = frozenset.intersection(
                *(frozenset(f.inputs) & group_plates for f in group_factors)
            )
            elim_plates = group_plates - outermost_plates
            for f in funsor.sum_product.partial_sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                group_factors,
                plates=group_plates,
                eliminate=group_vars | elim_plates,
            ):
                contracted_costs.append(model_terms["scale"] * f)

        # accumulate costs from model (logp) and guide (-logq)
        costs = contracted_costs + uncontracted_factors  # model costs: logp
        costs += [-f for f in guide_terms["log_factors"]]  # guide costs: -logq

        # compute expected cost
        # Cf. pyro.infer.util.Dice.compute_expectation()
        # https://github.com/pyro-ppl/pyro/blob/0.3.0/pyro/infer/util.py#L212
        # TODO Replace this with funsor.Expectation
        plate_vars = guide_terms["plate_vars"] | model_terms["plate_vars"]
        # compute the marginal logq in the guide corresponding to each cost term
        targets = dict()
        for cost in costs:
            input_vars = frozenset(cost.inputs)
            if input_vars not in targets:
                targets[input_vars] = funsor.Tensor(
                    funsor.ops.new_zeros(
                        funsor.tensor.get_default_prototype(),
                        tuple(v.size for v in cost.inputs.values()),
                    ),
                    cost.inputs,
                    cost.dtype,
                )
        with AdjointTape() as tape:
            logzq = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                guide_terms["log_measures"] + list(targets.values()),
                plates=plate_vars,
                eliminate=(plate_vars | guide_terms["measure_vars"]),
            )
        marginals = tape.adjoint(
            funsor.ops.logaddexp, funsor.ops.add, logzq, tuple(targets.values())
        )
        # finally, integrate out guide variables in the elbo and all plates
        elbo = to_funsor(0, output=funsor.Real)
        for cost in costs:
            target = targets[frozenset(cost.inputs)]
            logzq_local = marginals[target].reduce(
                funsor.ops.logaddexp, frozenset(cost.inputs) - plate_vars
            )
            log_prob = marginals[target] - logzq_local
            elbo_term = funsor.Integrate(
                log_prob,
                cost,
                guide_terms["measure_vars"] & frozenset(log_prob.inputs),
            )
            elbo += elbo_term.reduce(
                funsor.ops.add, plate_vars & frozenset(cost.inputs)
            )

    # evaluate the elbo, using memoize to share tensor computation where possible
    with funsor.interpretations.memoize():
        return to_data(apply_optimizer(elbo))


class replay(OrigReplayMessenger):
    def process_message(self, msg):
        super().process_message(msg)
        if msg["type"] == "sample" and msg["name"] in self.trace:
            msg["is_replayed"] = True
