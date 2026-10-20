# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Compare two sets of benchmark measurements and render a Markdown report.

Usage::

    python -m benchmarks.compare \\
        --base base-1.json base-2.json \\
        --head head-1.json head-2.json \\
        --output comment.md

Several JSON files may be given per side. Each benchmark is reduced to the
*best* observation across those rounds, which is the standard defence against
shared-CI-runner noise: a run can be slowed down by a noisy neighbour, but it
cannot be sped up below the machine's real capability.
"""

import argparse
import json
from typing import Any, Optional

#: HTML marker used to find and update the bot's own comment in place.
COMMENT_MARKER = "<!-- numpyro-benchmark-report -->"

FASTER, SLOWER, NEUTRAL, UNAVAILABLE = "faster", "slower", "neutral", "n/a"
#: Moved beyond the threshold, but the measurement is too small or too noisy to
#: stand behind. Reported in parentheses rather than called a change.
UNRESOLVED = "unresolved"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", nargs="+", required=True, help="base JSON report(s)")
    parser.add_argument("--head", nargs="+", required=True, help="head JSON report(s)")
    parser.add_argument("-o", "--output", help="path to write the Markdown report to")
    parser.add_argument(
        "--json-output", help="path to write a machine-readable summary"
    )
    parser.add_argument(
        "--base-label",
        default="baseline",
        help="what to call the base side in column headers",
    )
    parser.add_argument(
        "--head-label",
        default="this PR",
        help="what to call the head side in column headers",
    )
    parser.add_argument(
        "--base-ref", default="", help="branch or ref name the base side came from"
    )
    parser.add_argument(
        "--head-ref", default="", help="branch or ref name the head side came from"
    )
    parser.add_argument(
        "--repo-url",
        default="",
        help="repository URL, e.g. https://github.com/pyro-ppl/numpyro, used to "
        "turn commit shas into links",
    )
    parser.add_argument(
        "--run-threshold",
        type=float,
        default=5.0,
        help="percent change in run time below which a result is called neutral",
    )
    parser.add_argument(
        "--compile-threshold",
        type=float,
        default=25.0,
        help=(
            "percent change in compile time below which a result is called neutral; "
            "looser than the run threshold because an A/A control run shows compile "
            "time swinging ~20 percent on identical code"
        ),
    )
    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=1.0,
        help="ignore run-time changes below this, where relative noise dominates",
    )
    parser.add_argument(
        "--min-compile-ms",
        type=float,
        default=50.0,
        help=(
            "ignore compile-time changes below this. A benchmark with no jitted work "
            "still shows a small cold/warm difference from dispatch warm-up, which is "
            "jitter rather than compilation"
        ),
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero when any run-time regression is detected",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Loading and reduction
# --------------------------------------------------------------------------- #


def load_side(paths: list[str]) -> dict[str, Any]:
    """Load one side's rounds and reduce each benchmark to its best observation."""
    best: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    commit = ""
    rounds = 0

    for path in paths:
        with open(path) as f:
            report = json.load(f)
        rounds += 1
        metadata = metadata or report.get("metadata", {})
        commit = commit or report.get("commit", "")
        for record in report["results"]:
            key = f"{record['suite']}.{record['name']}"
            if record["status"] != "ok":
                # Keep the first failure so the reason can be reported, but let
                # any successful round from another attempt win.
                best.setdefault(key, record)
                continue
            current = best.get(key)
            if (
                current is None
                or current["status"] != "ok"
                or record["warm_min_s"] < current["warm_min_s"]
            ):
                best[key] = record

    return {"records": best, "metadata": metadata, "commit": commit, "rounds": rounds}


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def _pct(base: float, head: float) -> Optional[float]:
    if base <= 0:
        return None
    return (head - base) / base * 100.0


def _classify(
    base: Optional[float],
    head: Optional[float],
    threshold: float,
    min_duration_s: float,
) -> tuple[str, Optional[float]]:
    if base is None or head is None:
        return UNAVAILABLE, None
    pct = _pct(base, head)
    if pct is None:
        return UNAVAILABLE, None
    if max(base, head) < min_duration_s:
        # Too fast to measure reliably on a shared runner. Still worth showing
        # the number when it moved a lot, but not worth calling a regression.
        return (UNRESOLVED if abs(pct) > threshold else NEUTRAL), pct
    if pct > threshold:
        return SLOWER, pct
    if pct < -threshold:
        return FASTER, pct
    return NEUTRAL, pct


def compare(
    base: dict[str, Any],
    head: dict[str, Any],
    run_threshold: float,
    compile_threshold: float,
    min_duration_ms: float,
    min_compile_ms: float = 50.0,
) -> list[dict[str, Any]]:
    """Build one comparison row per benchmark present on either side."""
    min_duration_s = min_duration_ms / 1e3
    min_compile_s = min_compile_ms / 1e3
    rows = []

    for key in sorted(set(base["records"]) | set(head["records"])):
        base_rec = base["records"].get(key)
        head_rec = head["records"].get(key)
        suite, _, name = key.partition(".")

        def metric(rec: Optional[dict[str, Any]], field: str) -> Optional[float]:
            if rec is None or rec["status"] != "ok":
                return None
            return rec[field]

        base_run = metric(base_rec, "warm_min_s")
        head_run = metric(head_rec, "warm_min_s")
        base_compile = metric(base_rec, "compile_s")
        head_compile = metric(head_rec, "compile_s")

        run_verdict, run_pct = _classify(
            base_run, head_run, run_threshold, min_duration_s
        )
        compile_verdict, compile_pct = _classify(
            base_compile, head_compile, compile_threshold, min_compile_s
        )

        note = ""
        if base_rec is None:
            note = "new benchmark, not present on the base ref"
        elif head_rec is None:
            note = "benchmark missing from this PR"
        elif base_rec["status"] != "ok" and head_rec["status"] != "ok":
            note = f"failed on both sides: {head_rec.get('error', '')}"
        elif base_rec["status"] != "ok":
            note = f"failed on the base ref: {base_rec.get('error', '')}"
        elif head_rec["status"] != "ok":
            note = f"failed on this PR: {head_rec.get('error', '')}"

        rows.append(
            {
                "key": key,
                "suite": suite,
                "name": name,
                "description": (head_rec or base_rec or {}).get("description", ""),
                "base_run_s": base_run,
                "head_run_s": head_run,
                "run_pct": run_pct,
                "run_verdict": run_verdict,
                "base_compile_s": base_compile,
                "head_compile_s": head_compile,
                "compile_pct": compile_pct,
                "compile_verdict": compile_verdict,
                "note": note,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def fmt_duration(seconds: Optional[float]) -> str:
    """Render a duration with a unit that keeps three significant figures."""
    if seconds is None:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def fmt_delta(pct: Optional[float], verdict: str) -> str:
    """Render a percentage change as plain text."""
    if pct is None:
        return "n/a"
    if verdict == UNRESOLVED:
        # Parenthesised: it cleared the threshold, but on a measurement too
        # small to stand behind.
        return f"({pct:+.1f}%)"
    return f"{pct:+.1f}%"


def commit_link(sha: str, repo_url: str) -> str:
    """Render a commit sha, linked to the repository when the URL is known."""
    if not sha:
        return "—"
    short = sha[:8]
    if not repo_url:
        return f"`{short}`"
    return f"[`{short}`]({repo_url.rstrip('/')}/commit/{sha})"


def _prefix(row: dict[str, Any]) -> str:
    """Pick the character that colours a row inside a ``diff`` fence.

    A regression outranks an improvement, so a benchmark that got faster to run
    but slower to compile still shows up red. Only the line as a whole can be
    coloured; the per-column direction is carried by the signed percentages.
    """
    verdicts = (row["run_verdict"], row["compile_verdict"])
    if SLOWER in verdicts:
        return "-"
    if FASTER in verdicts:
        return "+"
    return " "


def _table(rows: list[dict[str, Any]], base_label: str, head_label: str) -> list[str]:
    """Render rows as an aligned monospace table inside a ``diff`` fence.

    GitHub highlights a ``diff`` block by line: one starting with ``-`` is red
    and one starting with ``+`` is green. That gives real colour without the
    LaTeX workaround, and a fixed-width block keeps the numbers in columns
    instead of letting a Markdown table squeeze them.
    """
    if not rows:
        return []

    body = [
        (
            _prefix(row),
            row["name"] + (" †" if row["note"] else ""),
            fmt_duration(row["base_run_s"]),
            fmt_duration(row["head_run_s"]),
            fmt_delta(row["run_pct"], row["run_verdict"]),
            fmt_duration(row["base_compile_s"]),
            fmt_duration(row["head_compile_s"]),
            fmt_delta(row["compile_pct"], row["compile_verdict"]),
        )
        for row in rows
    ]
    headers = (
        "benchmark",
        base_label,
        head_label,
        "Δ",
        base_label,
        head_label,
        "Δ",
    )
    widths = [
        max(len(header), max(len(cell[i + 1]) for cell in body))
        for i, header in enumerate(headers)
    ]

    def line(cells: tuple[str, ...]) -> str:
        name = cells[1].ljust(widths[0])
        run = "  ".join(c.rjust(w) for c, w in zip(cells[2:5], widths[1:4]))
        compile_ = "  ".join(c.rjust(w) for c, w in zip(cells[5:8], widths[4:7]))
        return f"{cells[0]} {name}   {run}     {compile_}".rstrip()

    run_span = sum(widths[1:4]) + 4
    compile_span = sum(widths[4:7]) + 4
    group = (
        " " * (2 + widths[0] + 3)
        + " run time ".center(run_span, "─")
        + "     "
        + " compile time ".center(compile_span, "─")
    )
    header = line((" ",) + headers)
    # A rule drawn with ASCII hyphens would read as a diff file header, so use
    # box drawing.
    rule = "─" * len(header)

    return [
        "```diff",
        group,
        header,
        rule,
        *(line(cells) for cells in body),
        "```",
    ]


def _legend() -> list[str]:
    return [
        "<sub>Red is slower, green is faster; a row is coloured by the worse of "
        "its two columns. A delta in parentheses cleared the threshold on a "
        "measurement below the resolution floor, so it is shown without being "
        "called a change. † marks a benchmark that could not be compared — see "
        "below.</sub>",
    ]


def _headline(rows: list[dict[str, Any]]) -> list[str]:
    """Render the counts as a small coloured ``diff`` block."""
    metrics = (("run_verdict", "run time"), ("compile_verdict", "compile time"))
    width = max(len(name) for _, name in metrics)

    lines = []
    for field, name in metrics:
        slower = sum(1 for r in rows if r[field] == SLOWER)
        faster = sum(1 for r in rows if r[field] == FASTER)
        label = f"{name}:".ljust(width + 1)
        if slower:
            lines.append(f"- {label} {slower} slower, {faster} faster")
        elif faster:
            lines.append(f"+ {label} {faster} faster")
        else:
            lines.append(f"  {label} unchanged across {len(rows)} benchmarks")
    return ["```diff", *lines, "```"]


def render(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    head: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Render the full Markdown comment body."""
    significant = [
        r
        for r in rows
        if r["run_verdict"] in (FASTER, SLOWER)
        or r["compile_verdict"] in (FASTER, SLOWER)
    ]
    problems = [r for r in rows if r["note"]]

    def side(label: str, ref: str, meta: dict[str, Any]) -> str:
        parts = [f"**{label}**"]
        if ref:
            parts.append(f"`{ref}`")
        if meta["commit"]:
            parts.append(f"at {commit_link(meta['commit'], args.repo_url)}")
        return " ".join(parts)

    out = [
        COMMENT_MARKER,
        "## Benchmark report",
        "",
        "{} vs {}".format(
            side(args.head_label, args.head_ref, head),
            side(args.base_label, args.base_ref, base),
        ),
        "",
        *_headline(rows),
        "",
    ]

    if base["metadata"].get("quick_mode") or head["metadata"].get("quick_mode"):
        out += [
            "> [!CAUTION]",
            "> Measured with `NUMPYRO_BENCH_QUICK=1`, which takes a single warm "
            "repeat per benchmark. These numbers are a smoke test of the harness, "
            "not a result: expect tens of percent of noise on unchanged code.",
            "",
        ]

    # A difference in the surrounding stack means the numbers describe that
    # difference rather than the change under review, so say so up front.
    for field, label in (("jax_version", "JAX"), ("python_version", "Python")):
        base_value = base["metadata"].get(field)
        head_value = head["metadata"].get(field)
        if base_value and head_value and base_value != head_value:
            out += [
                "> [!WARNING]",
                f"> The two sides ran on different {label} versions "
                f"(`{base_value}` vs `{head_value}`). This comparison reflects that "
                "difference as much as it reflects the change under review.",
                "",
            ]

    if significant:
        out += [
            f"### Significant changes ({len(significant)})",
            "",
            *_table(significant, args.base_label, args.head_label),
            "",
        ]
    else:
        out += [
            "### No significant changes",
            "",
            f"Every benchmark stayed within ±{args.run_threshold:g}% run time and "
            f"±{args.compile_threshold:g}% compile time.",
            "",
        ]
    # The full results below carry the same notation either way.
    out += [*_legend(), ""]

    suites = sorted({r["suite"] for r in rows})
    out += ["<details>", "<summary>Full results</summary>", ""]
    for suite in suites:
        suite_rows = [r for r in rows if r["suite"] == suite]
        out += [
            f"#### `{suite}`",
            "",
            *_table(suite_rows, args.base_label, args.head_label),
            "",
        ]
    out += ["</details>", ""]

    if problems:
        out += [
            "<details>",
            "<summary>† Benchmarks that did not compare cleanly</summary>",
            "",
        ]
        for row in problems:
            out.append(f"- `{row['key']}` — {row['note']}")
        out += ["", "</details>", ""]

    out += [
        "<details>",
        "<summary>Methodology and environment</summary>",
        "",
        "Each benchmark is set up untimed, then called once with the JAX caches "
        "cleared and several more times warm. **Run** is the fastest warm call; "
        "**compile** is the first call minus that, i.e. the tracing, lowering and "
        "XLA compilation the warm calls did not have to pay for.",
        "",
        f"Both refs were measured on the same runner over "
        f"{min(base['rounds'], head['rounds'])} interleaved round(s), taking the best "
        "observation per benchmark. A result is called neutral when it moves less than "
        f"±{args.run_threshold:g}% (run) or ±{args.compile_threshold:g}% (compile), or when "
        f"the measurement itself is under {args.min_duration_ms:g} ms (run) / "
        f"{args.min_compile_ms:g} ms (compile) — a shared CI runner cannot resolve changes "
        "below that. Compile time gets the looser band because it is measured once per "
        "round rather than best-of-N, and swings by roughly 20% even between two runs of "
        "identical code. A delta shown in parentheses did clear its threshold, but "
        "on a measurement below the resolution floor, so it is reported without "
        "being called a change.",
        "",
        f"| | {args.base_label} | {args.head_label} |",
        "| --- | --- | --- |",
        f"| ref | {f'`{args.base_ref}`' if args.base_ref else '—'} "
        f"| {f'`{args.head_ref}`' if args.head_ref else '—'} |",
        f"| commit | {commit_link(base['commit'], args.repo_url)} "
        f"| {commit_link(head['commit'], args.repo_url)} |",
        f"| numpyro | {base['metadata'].get('numpyro_version', '—')} "
        f"| {head['metadata'].get('numpyro_version', '—')} |",
        f"| jax | {base['metadata'].get('jax_version', '—')} "
        f"| {head['metadata'].get('jax_version', '—')} |",
        f"| backend | {base['metadata'].get('jax_backend', '—')} "
        f"| {head['metadata'].get('jax_backend', '—')} |",
        f"| python | {base['metadata'].get('python_version', '—')} "
        f"| {head['metadata'].get('python_version', '—')} |",
        "",
        f"Runner: `{head['metadata'].get('platform', 'unknown')}`, "
        f"{head['metadata'].get('cpu_count', '?')} CPUs.",
        "",
        "</details>",
    ]
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    base = load_side(args.base)
    head = load_side(args.head)
    rows = compare(
        base,
        head,
        args.run_threshold,
        args.compile_threshold,
        args.min_duration_ms,
        args.min_compile_ms,
    )
    # `Δ compile` is the widest header, so keep the label pair from pushing the
    # table into a horizontal scroll on a narrow screen.
    for name in ("base_label", "head_label"):
        if len(getattr(args, name)) > 24:
            setattr(args, name, getattr(args, name)[:23] + "…")
    body = render(rows, base, head, args)

    if args.output:
        with open(args.output, "w") as f:
            f.write(body)
    else:
        print(body)

    regressions = [r for r in rows if r["run_verdict"] == SLOWER]
    if args.json_output:
        summary = {
            "regressions": len(regressions),
            "improvements": sum(1 for r in rows if r["run_verdict"] == FASTER),
            "compile_regressions": sum(
                1 for r in rows if r["compile_verdict"] == SLOWER
            ),
            "problems": [r["key"] for r in rows if r["note"]],
            "rows": rows,
        }
        with open(args.json_output, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    if args.fail_on_regression and regressions:
        keys = ", ".join(r["key"] for r in regressions)
        print(f"run-time regressions detected: {keys}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
