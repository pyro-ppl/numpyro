# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Run the benchmark suites and write the measurements to JSON.

Usage::

    python -m benchmarks.runner --output head.json --label head
    python -m benchmarks.runner --list
    python -m benchmarks.runner --suite distributions --output dists.json

The module must be importable while ``numpyro`` resolves to whichever checkout
is installed in the active environment, so run it from a directory that does
*not* contain a ``numpyro/`` source tree (the CI workflow copies ``benchmarks/``
into a scratch directory for exactly this reason).
"""

import argparse
import json
import re
import sys
import time
from typing import Optional

from benchmarks.harness import collect_metadata, registry, run_benchmark
from benchmarks.suites import SUITE_ORDER


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", help="path to write the JSON report to")
    parser.add_argument(
        "--label",
        default="results",
        help="name for this set of measurements, e.g. 'head' or 'base'",
    )
    parser.add_argument(
        "--commit", default="", help="commit sha these results describe"
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITE_ORDER,
        help="restrict to a suite; repeatable",
    )
    parser.add_argument(
        "--filter", help="only run benchmarks whose 'suite.name' matches this regex"
    )
    parser.add_argument("--list", action="store_true", help="list benchmarks and exit")
    return parser.parse_args(argv)


def select(suites: Optional[list[str]] = None, pattern: Optional[str] = None) -> list:
    """Return the registered benchmarks matching ``suites`` and ``pattern``."""
    benches = list(registry().values())
    if suites:
        benches = [b for b in benches if b.suite in suites]
    if pattern:
        regex = re.compile(pattern)
        benches = [b for b in benches if regex.search(b.key)]
    order = {name: i for i, name in enumerate(SUITE_ORDER)}
    return sorted(benches, key=lambda b: (order.get(b.suite, len(order)), b.name))


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    benches = select(args.suite, args.filter)

    if args.list:
        for bench in benches:
            print(f"{bench.key}\t{bench.description}")
        return 0

    if not benches:
        print("no benchmarks selected", file=sys.stderr)
        return 1

    metadata = collect_metadata()
    print(
        f"running {len(benches)} benchmarks against numpyro "
        f"{metadata['numpyro_version']} on jax {metadata['jax_version']} "
        f"({metadata['jax_backend']})",
        file=sys.stderr,
    )

    results = []
    started = time.perf_counter()
    for i, bench in enumerate(benches, start=1):
        print(
            f"[{i}/{len(benches)}] {bench.key} ... ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        record = run_benchmark(bench)
        results.append(record)
        if record["status"] == "ok":
            print(
                f"run {record['warm_min_s'] * 1e3:.1f} ms, "
                f"compile {record['compile_s'] * 1e3:.1f} ms",
                file=sys.stderr,
            )
        else:
            print(f"FAILED ({record['error']})", file=sys.stderr)

    report = {
        "label": args.label,
        "commit": args.commit,
        "metadata": metadata,
        "wall_time_s": time.perf_counter() - started,
        "results": results,
    }

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(payload)

    failed = [r["name"] for r in results if r["status"] == "error"]
    if failed:
        print(
            f"{len(failed)} benchmark(s) failed: {', '.join(failed)}", file=sys.stderr
        )
    # Failures are reported in the JSON and rendered as 'n/a' in the comparison;
    # they must not fail the run, because a benchmark for a brand new feature
    # will legitimately fail against the base checkout.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
