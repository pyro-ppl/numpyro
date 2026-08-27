# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Timing harness that separates JIT compilation cost from warm execution cost.

Every benchmark is written as a *setup* function that returns a zero-argument
*run* callable::

    @benchmark(suite="mcmc", warm_repeats=3)
    def nuts_logistic_regression():
        "NUTS on a 1000x10 logistic regression."
        mcmc, kwargs = ...          # not timed

        def run():
            mcmc.run(random.key(0), **kwargs)
            return mcmc.get_samples()

        return run

The setup body is never timed, so data generation and object construction do
not pollute the measurement. The harness then calls ``run`` once with the JAX
caches cleared -- that first call pays for tracing, lowering and XLA
compilation -- and a further ``warm_repeats`` times with the compiled
executable already cached.

``compile_s`` is therefore ``cold_s - warm_min_s``: the part of the first call
that the following calls did not have to pay for. It is an estimate rather than
an exact reading of XLA's own timer, but it is measured identically on both
sides of a comparison, which is what matters for a regression report.
"""

from collections.abc import Callable
from dataclasses import dataclass
import gc
import os
import platform
import statistics
import sys
import time
import traceback
from typing import Any, Optional

import jax

__all__ = ["Benchmark", "benchmark", "collect_metadata", "registry", "run_benchmark"]

#: Set ``NUMPYRO_BENCH_QUICK=1`` to shrink every benchmark to a single warm
#: repeat. Intended for smoke-testing the harness itself, not for reporting.
QUICK = os.environ.get("NUMPYRO_BENCH_QUICK", "") not in ("", "0", "false")

_REGISTRY: dict[str, "Benchmark"] = {}


@dataclass(frozen=True)
class Benchmark:
    """A single named measurement."""

    name: str
    suite: str
    description: str
    setup: Callable[[], Callable[[], Any]]
    warm_repeats: int

    @property
    def key(self) -> str:
        """Fully qualified identifier, e.g. ``mcmc.nuts_logistic_regression``."""
        return f"{self.suite}.{self.name}"


def benchmark(
    *,
    suite: str,
    name: Optional[str] = None,
    warm_repeats: int = 5,
) -> Callable[[Callable[[], Callable[[], Any]]], Callable[[], Callable[[], Any]]]:
    """Register a setup function as a benchmark.

    :param suite: Group the benchmark belongs to, e.g. ``"mcmc"``.
    :param name: Benchmark name; defaults to the decorated function's name.
    :param warm_repeats: Number of post-compilation timed repeats.
    """

    def decorator(
        fn: Callable[[], Callable[[], Any]],
    ) -> Callable[[], Callable[[], Any]]:
        bench = Benchmark(
            name=name or fn.__name__,
            suite=suite,
            description=(fn.__doc__ or "").strip().splitlines()[0]
            if fn.__doc__
            else "",
            setup=fn,
            warm_repeats=1 if QUICK else warm_repeats,
        )
        if bench.key in _REGISTRY:
            raise ValueError(f"duplicate benchmark {bench.key!r}")
        _REGISTRY[bench.key] = bench
        return fn

    return decorator


def registry() -> dict[str, Benchmark]:
    """Return every registered benchmark, keyed by ``suite.name``."""
    return dict(_REGISTRY)


def _timed(run: Callable[[], Any]) -> float:
    """Time one call to ``run``, waiting for all device work to land."""
    start = time.perf_counter()
    jax.block_until_ready(run())
    return time.perf_counter() - start


def run_benchmark(bench: Benchmark) -> dict[str, Any]:
    """Measure one benchmark and return a JSON-serialisable record.

    A benchmark that raises is reported with ``status="error"`` rather than
    aborting the run: a newly added benchmark is expected to fail against an
    older base checkout that lacks the feature under test.
    """
    record: dict[str, Any] = {
        "name": bench.name,
        "suite": bench.suite,
        "description": bench.description,
        "status": "ok",
    }
    try:
        # Clear before setup so the cold call below genuinely recompiles, while
        # any compilation triggered by data generation stays untimed.
        jax.clear_caches()
        gc.collect()
        run = bench.setup()

        cold_s = _timed(run)
        warm = [_timed(run) for _ in range(bench.warm_repeats)]
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        return record

    warm_min = min(warm)
    record.update(
        cold_s=cold_s,
        warm_min_s=warm_min,
        warm_median_s=statistics.median(warm),
        warm_stdev_s=statistics.stdev(warm) if len(warm) > 1 else 0.0,
        warm_repeats=len(warm),
        # The first call also runs the workload once, so the excess over the
        # cheapest warm call is what compilation cost.
        compile_s=max(cold_s - warm_min, 0.0),
    )
    return record


def collect_metadata() -> dict[str, Any]:
    """Describe the environment a set of measurements was taken in."""
    import numpyro

    return {
        "numpyro_version": numpyro.__version__,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_device_count": jax.device_count(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "quick_mode": QUICK,
    }
