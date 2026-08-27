# Benchmarks

Speed benchmarks used to compare a feature branch against its merge base, and
to report the difference as a pull request comment.

The suites cover MCMC (`HMC`/`NUTS`), SVI with the common autoguides, the
distributions layer, and effect handlers. Each measurement is reported twice:
as **run time** and as **compile time**.

## Running locally

The package depends on nothing beyond `numpyro[cpu]` itself.

```sh
# list what is registered
python -m benchmarks.runner --list

# measure everything, or just one suite
python -m benchmarks.runner --output head.json
python -m benchmarks.runner --suite distributions --output dists.json

# a fast, statistically meaningless pass, for checking the harness works
NUMPYRO_BENCH_QUICK=1 python -m benchmarks.runner --suite handlers -o smoke.json
```

To compare two revisions by hand, install each into its own environment and run
the *same* copy of `benchmarks/` against both, from a directory that has no
`numpyro/` source tree in it — otherwise `import numpyro` picks up the working
copy instead of the installed one:

```sh
mkdir -p /tmp/bench && cp -r benchmarks /tmp/bench/
cd /tmp/bench
"$BASE_VENV/bin/python" -m benchmarks.runner --label base -o base-1.json
"$HEAD_VENV/bin/python" -m benchmarks.runner --label head -o head-1.json
python -m benchmarks.compare --base base-*.json --head head-*.json
```

## What the two columns mean

Every benchmark is written as a *setup* function that returns a zero-argument
*run* callable. Setup is never timed, so data generation and object
construction stay out of the measurement:

```python
@benchmark(suite="mcmc", warm_repeats=3)
def nuts_eight_schools():
    """NUTS on non-centred eight schools."""
    data = eight_schools_data()
    mcmc = MCMC(
        NUTS(eight_schools), num_warmup=500, num_samples=500, progress_bar=False
    )

    def run():
        mcmc.run(random.PRNGKey(0), **data)
        return mcmc.get_samples()

    return run
```

The harness then clears the JAX caches, calls `run` once, and calls it
`warm_repeats` more times.

- **Run** is the fastest warm call — the compiled executable, cache already hot.
- **Compile** is the cold call minus that — the tracing, lowering and XLA
  compilation the warm calls did not have to pay for.

Compile time is therefore an estimate rather than a reading off XLA's own
timer, but it is derived identically on both sides of a comparison, which is
what a regression report needs. Benchmarks with no jitted work (most of the
`handlers` suite) report a near-zero compile column by construction.

## Noise

These run on shared GitHub runners, so `benchmarks.compare` calls a result
neutral unless it clears a threshold:

| | default | why |
| --- | --- | --- |
| `--run-threshold` | 5% | An A/A control run — identical code on both sides — stays within ±2%. |
| `--compile-threshold` | 25% | The same control run swings ~20%: compile time is measured once per round, not best-of-N. |
| `--min-duration-ms` | 1 ms | Below this, dispatch jitter dominates the measurement. |
| `--min-compile-ms` | 50 ms | An uncompiled benchmark still shows a small cold/warm gap from warm-up, which is not compilation. |

Results are rendered as fixed-width tables inside ` ```diff ` fences rather
than as Markdown tables: the columns stay aligned, and GitHub's diff
highlighting colours a line red when it starts with `-` and green when it
starts with `+`. A regression outranks an improvement, so a benchmark that got
faster to run but slower to compile still shows red — the per-column direction
is carried by the signed percentages. Colouring is per line, not per cell,
which is the trade for having it work at all.

A delta **in parentheses** cleared its threshold but sits below the resolution
floor, so it is shown without being called a change — usually a sign the
benchmark itself is too small and should be given more work to do.

The environment table stays Markdown, because commit links do not survive
inside a code block.

Two further defences are applied by the workflow rather than the comparison
itself: both refs are measured on the same runner, in rounds that alternate
which side goes first, and each benchmark is reduced to its best observation
across rounds. A run can be slowed down by a noisy neighbour; it cannot be sped
up past what the machine can do.

**Before believing a regression, re-run it.** A single flagged row on a busy
runner is weak evidence.

## CI

`.github/workflows/benchmark.yml` runs on every pull request against `master`,
and again on each push to one, so the comment always describes the current
head. A full comparison takes tens of minutes; runs for superseded pushes are
cancelled as new ones start.

It can also be started from the Actions tab, where `base_ref`, `rounds` and
`benchmark_args` can be set.

It builds two virtual environments — the PR head and the merge base — pinning
the base to the head's JAX version so the report measures NumPyro rather than
JAX. Both are then measured with the head revision of `benchmarks/`, so a
benchmark added by the PR still runs against the base. Where it cannot run
there, the row is reported as `n/a` with the reason rather than being dropped.

The result is posted as a sticky comment, editing the bot's own previous
comment in place rather than piling up a new one per run. Which of the two
paths does the posting depends on where the pull request came from:

- **From a branch in this repository**, the run has a write token, so
  `benchmark.yml` comments directly at the end of the job.
- **From a fork**, GitHub caps the run's token at read-only whatever the
  workflow asks for. `benchmark.yml` can then only upload an artifact, and
  `.github/workflows/benchmark-comment.yml` posts on its behalf: it is
  triggered by `workflow_run`, holds the write token itself, and never
  executes anything out of the pull request — it reads the artifact and
  validates the PR number and head SHA before commenting.

Both call the same `.github/scripts/post-sticky-comment.js`.

> **`workflow_run` only ever runs the copy of the workflow that is on the
> default branch.** `benchmark-comment.yml` therefore does nothing at all until
> it has been merged to `master` — which is fine for same-repo pull requests,
> since those never reach it, but means fork PRs stay uncommented until then.

The report is also written to the job summary, so it is readable from the
Actions tab even when no comment was posted.

## Adding a benchmark

Add a setup function to the relevant module under `benchmarks/suites/`. Keep it
on public API that has been stable for a while — the same code has to run
against the base branch, and anything freshly added will simply fail there.

Aim for a warm time between roughly 1 ms and a few hundred ms: faster than that
disappears into the noise floor, slower than that is paid four times over in
every comparison.
