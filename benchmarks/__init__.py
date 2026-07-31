# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Speed benchmarks used to compare a feature branch against master.

This package is deliberately kept out of the installed ``numpyro`` distribution
and depends on nothing beyond ``numpyro[cpu]`` itself, so that the exact same
benchmark definitions can be executed against two different checkouts of
NumPyro installed into two different virtual environments.
"""

from benchmarks.harness import Benchmark, benchmark, registry, run_benchmark

__all__ = ["Benchmark", "benchmark", "registry", "run_benchmark"]
