# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class PytreeTrace:
    def __init__(self, trace):
        self.trace = trace

    def tree_flatten(self):
        trace, aux_trace = {}, {}
        for name, site in self.trace.items():
            if site["type"] in ["sample", "deterministic", "plate", "param"]:
                trace[name], aux_trace[name] = {}, {"_control_flow_done": True}
                for key in site:
                    if key == "fn":
                        if site["type"] == "sample":
                            trace[name][key] = site[key]
                        elif site["type"] == "plate":
                            aux_trace[name][key] = site[key]
                    elif key in ["args", "value", "intermediates"]:
                        trace[name][key] = site[key]
                    # scanned sites have stop field because we trace them inside a block handler
                    elif key != "stop":
                        if key == "kwargs":
                            kwargs = site["kwargs"].copy()
                            if "rng_key" in kwargs:
                                # rng_key is not traced else it is collected by the
                                # scan primitive which doesn't make sense
                                # set to None to avoid leaks during tracing by JAX
                                kwargs["rng_key"] = None
                            aux_trace[name][key] = kwargs
                        elif key == "infer":
                            kwargs = site["infer"].copy()
                            if "_scan_current_index" in kwargs:
                                # set to None to avoid leaks during tracing by JAX
                                kwargs["_scan_current_index"] = None
                            aux_trace[name][key] = kwargs
                        else:
                            aux_trace[name][key] = site[key]
        # keep the site order information because in JAX, flatten and unflatten do not preserve
        # the order of keys in a dict
        site_names = list(trace.keys())
        return (trace,), (aux_trace, site_names)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        aux_trace, site_names = aux_data
        (trace,) = children
        trace_with_aux = {}
        for name in site_names:
            trace[name].update(aux_trace[name])
            trace_with_aux[name] = trace[name]
        return cls(trace_with_aux)
