# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark suites.

Importing this package imports every module below it, which is what populates
the registry in :mod:`benchmarks.harness`.
"""

from benchmarks.suites import distributions, handlers, mcmc, svi

__all__ = ["distributions", "handlers", "mcmc", "svi"]

#: Suite names in the order they should be reported.
SUITE_ORDER = ["mcmc", "svi", "distributions", "handlers"]
