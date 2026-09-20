# AGENTS.md

This file provides guidance to AI coding agents working with code in this repository.

## Environment

Dependencies are managed with `uv` (`uv.lock` is committed). Dev tooling lives in PEP 735 dependency groups, not extras, so the `pip install -e '.[dev,...]'` line in `Makefile`/`CONTRIBUTING.md` is out of date. Mirror CI instead:

```sh
uv sync --extra cpu --group dev --group test            # core development
uv sync --extra cpu --group dev --group test --group docs --group examples   # + docs / examples
```

Prefix commands with `uv run` (or activate `.venv`). `funsor` is pulled from git (`[tool.uv.sources]`).

## Commands

```sh
make lint      # ruff check, ruff format --check, license-header check, ty check
make format    # adds license headers, ruff format, ruff check --fix
make test      # lint, then pytest -v test  (slow: the full suite)
make docs      # sphinx html -> docs/build/html (needs pandoc)
make doctest   # docstring tests via sphinx, forced to CPU
python -m doctest -v README.md   # README snippets are tested in CI
```

Single test / subset:

```sh
pytest -vs test/test_distributions.py::test_log_prob -k Gompertz
pytest -vs -n auto test/infer/test_svi.py        # needs pytest-xdist
```

Variants CI runs, which matter when touching the corresponding code:

```sh
JAX_ENABLE_X64=1 pytest -vs test/infer/test_mcmc.py -k x64                      # double precision
XLA_FLAGS="--xla_force_host_platform_device_count=2" pytest -vs test/infer/test_mcmc.py -k "chain or pmap or vmap"   # multi-chain
JAX_ENABLE_CUSTOM_PRNG=1 pytest -vs test/infer/test_mcmc.py                     # typed PRNG keys
JAX_CHECK_TRACER_LEAKS=1 pytest -vs test/infer/test_mcmc.py::test_chain_inside_jit
CI=1 XLA_FLAGS="--xla_force_host_platform_device_count=2" pytest -vs -k test_example   # runs examples/ scripts
```

CI shards the suite as: everything except `test/infer` and `test/contrib` with `-k "not test_example"` (modeling); `test/contrib` + `test/infer` (inference); `-k test_example` (examples). `test/contrib/test_nested_sampling.py` requires `JAX_ENABLE_X64=1`.

Benchmarks (compared against the merge base on PRs): `python -m benchmarks.runner --list`, `NUMPYRO_BENCH_QUICK=1 python -m benchmarks.runner --suite handlers -o smoke.json`. See `benchmarks/README.md`.

## Things that will fail CI

- **Warnings are errors.** `pyproject.toml` sets pytest `filterwarnings = ["error", ...]`; a new `DeprecationWarning`/`UserWarning` on a tested path fails the test. Use `pytest.warns` or fix the source.
- **License headers.** Every non-empty `.py` file must begin with the two-line Pyro copyright / `SPDX-License-Identifier: Apache-2.0` header. `make license` (or `make format`) adds it.
- **Type annotations are enforced selectively.** Ruff's `ANN` rules apply only to the typed modules listed in `[tool.ruff.lint.per-file-ignores]` (`diagnostics.py`, `handlers.py`, `optim.py`, `patch.py`, `primitives.py`, `infer/elbo.py`, `distributions/distribution.py`), and `ty check` covers the include list in `[tool.ty.src]` (all of `numpyro/distributions`, most of `numpyro/infer`, several contrib packages). Code in those paths must be annotated and type-check; shared aliases are in `numpyro/_typing.py`.
- **Import order** is ruff-isort with a custom `known-jax` section (`flax`, `jax`, `optax`, `tensorflow_probability`) placed between third-party and first-party, `force-sort-within-sections`. Let `make format` do it.
- pre-commit additionally runs `codespell`, `yamlfmt`, and `tombi` (TOML formatting) — `pyproject.toml` edits must stay tombi-formatted.
- `test/conftest.py` forces the CPU platform, reseeds with `set_rng_seed(0)` before every test, enables x64 when `JAX_ENABLE_X64` is set, and asserts `jax.live_arrays()` is empty before the first test — so test modules must not create JAX arrays at import/collection time (build parametrize data with NumPy).

## Architecture

NumPyro is Pyro's modeling API re-implemented on JAX. Three layers, each usable independently:

### 1. Primitives + effect handlers (`primitives.py`, `handlers.py`)

A model is a plain Python function calling `numpyro.sample`, `param`, `deterministic`, `plate`, `factor`, etc. Each primitive builds a **message dict** (`type`, `name`, `fn`, `args`, `kwargs`, `value`, `is_observed`, `scale`, `mask`, `cond_indep_stack`, `infer`, ...) and passes it through `apply_stack` over the global `_PYRO_STACK` of `Messenger`s: `process_message` runs from the top of the stack down, then the default sampler runs if no handler set `value`, then `postprocess_message` runs back up. With an empty stack, `sample` just calls the distribution (which is why a `rng_key` is then required).

Handlers (`trace`, `seed`, `substitute`, `condition`, `uncondition`, `block`, `replay`, `scale`, `mask`, `reparam`, `collapse`, `do`, `lift`, `scope`, `infer_config`) are `Messenger` subclasses used as context managers or function wrappers. All inference is built by composing them, e.g. `trace(substitute(seed(model, key), params)).get_trace(...)`. Because JAX has no global RNG, randomness only enters through the `seed` handler, which splits its key per sample site. `plate` is itself a Messenger: it records a `CondIndepStackFrame`, broadcasts `fn` batch shapes via `expand`, and sets `scale` for subsampling.

Handlers execute Python side effects, so they run at **trace time**: inference code traces the model once under `jit` to turn it into a pure function. Python control flow over traced values inside models must go through `contrib/control_flow` (`scan`, `cond`), which are handler-aware.

### 2. Distributions (`numpyro/distributions/`)

Mirrors `torch.distributions`: `Distribution` (`distribution.py`) with `batch_shape`/`event_shape`, `arg_constraints`, `support`, `reparametrized_params`, `sample(key, sample_shape)`, `log_prob`, plus wrappers `Independent` (`.to_event`), `MaskedDistribution` (`.mask`), `ExpandedDistribution` (`.expand`), `TransformedDistribution`, `Delta`, `Unit`, `ImproperUniform`.

- **Every distribution, transform, and constraint is a JAX pytree**, registered automatically in `__init_subclass__`. Array parameters are leaves (derived from `arg_constraints` plus `pytree_data_fields`); anything static must be listed in `pytree_aux_fields`. Putting array-valued data in aux fields breaks `jit`/`vmap` (hash/equality on tracers) — see the recent refactor keeping truncation bounds out of the static treedef. `batch_util.py` (`vmap_over`, `promote_batch_shape`) has per-class registrations needed for distributions to pass through `vmap`.
- `constraints.py` → `transforms.py`: `biject_to(constraint)` is a registry mapping each support to a bijection from unconstrained space. HMC, autoguides and `param(constraint=...)` all rely on it, so a new constraint needs a `biject_to` registration.
- Argument validation is "omnistaging": `validate_args` checks work both eagerly and under `jit` (where they can only warn/NaN rather than raise).
- `kl.py` uses `multipledispatch` for `kl_divergence(p, q)`; `conjugate.py`, `truncated.py`, `censored.py`, `mixtures.py`, `directional.py`, `copula.py`, `flows.py` hold the families their names suggest.

**Adding a distribution** touches several places: the class (in `continuous.py`/`discrete.py`/...), export in `distributions/__init__.py`, an entry in `docs/source/distributions.rst`, and test cases in `test/test_distributions.py` — add `T(dist_cls, *params)` rows to the `CONTINUOUS`/`DISCRETE`/`DIRECTIONAL` lists and, if SciPy has an equivalent, a mapping in `_DIST_MAP`. The generic parametrized tests (shapes, `log_prob` vs SciPy, gradients, `cdf`/`icdf`, mean/var, constraints, pytree/vmap, `expand`, sample goodness-of-fit) then run automatically. `vmap`-related tests may need a `vmap_over` registration in `batch_util.py`. Recent distribution docstrings follow a "mathematical details" format with LaTeX pdf/support/moments and linked citations (issue gh-2187).

### 3. Inference (`numpyro/infer/`)

`infer/util.py` is the bridge between models and algorithms: `log_density` (trace + substitute → joint log prob honoring `scale`/`mask`), `initialize_model` (finds valid initial params via the strategies in `initialization.py`, and returns `potential_fn`, `postprocess_fn`, the unconstraining transforms), `transform_fn`/`constrain_fn`/`unconstrain_fn`, `potential_energy`, and the `Predictive` / `log_likelihood` utilities. Algorithms work in **unconstrained space** and map back with `biject_to` transforms.

- **MCMC** (`mcmc.py`): `MCMC` drives any `MCMCKernel` (`init` / `sample` / `postprocess_fn`, state is a namedtuple pytree). It owns warmup/sample collection via `fori_collect`, progress bars, and `chain_method` = `parallel` (pmap; needs multiple devices, hence the `XLA_FLAGS` tests), `vectorized` (vmap), or `sequential`. Kernels: `HMC`/`NUTS` (`hmc.py`, with the functional core — leapfrog, tree building, step-size and mass-matrix adaptation — in `hmc_util.py`), `SA`, `BarkerMH`, ensemble samplers `AIES`/`ESS` (`ensemble.py`), and the Gibbs family — `Gibbs`/`CustomGibbs`/`DiscreteGibbs` in `gibbs.py`, with `HMCGibbs`/`DiscreteHMCGibbs` (subclasses of `Gibbs`) and `HMCECS` in `hmc_gibbs.py`, plus `MixedHMC` in `mixed_hmc.py` — composable kernels that wrap inner kernels.
- **SVI** (`svi.py`, `elbo.py`, `autoguide.py`): `SVI(model, guide, optim, loss)` with functional `init`/`update`/`run` over an `SVIState`. ELBOs (`Trace_ELBO`, `TraceMeanField_ELBO`, `TraceGraph_ELBO`, `TraceEnum_ELBO`, `RenyiELBO`) replay the model against a guide trace. Autoguides build guides from the model's latent sites using `initialize_model`-style setup. `numpyro/optim.py` wraps `jax.example_libraries.optimizers` and adapts `optax` transforms to the same `init/update/get_params` interface.
- **Reparameterization** (`reparam.py`) is applied through `handlers.reparam`, not inside distributions.
- **Discrete latents**: enumeration is delegated to `funsor` via `numpyro/contrib/funsor` (`enum`, `config_enumerate`, `infer_discrete`, enum-aware `log_density`). `funsor` is an optional dependency — keep its imports lazy/inside contrib.

### Other packages

- `numpyro/contrib/`: optional-dependency integrations, each imported lazily — `module.py` (flax linen/nnx and equinox modules via `flax_module`/`nnx_module`/`eqx_module` and their `random_*_module` Bayesian variants), `tfp/` (wrap TFP distributions and MCMC kernels), `einstein/` (SteinVI), `hsgp/` (Hilbert-space GP approximations), `stochastic_support/` (DCC), `nested_sampling.py` (jaxns), `control_flow/`, `render.py` (`render_model` via graphviz), `ecs_proxies.py`.
- `numpyro/compat/`: a Pyro-style API shim, validated against `pyro-api` in `test/pyroapi`.
- `numpyro/ops/`: `indexing` (`Vindex`), `provenance` (dependency tracking used by `infer/inspect.py` and `render_model`), `pytree`.
- `numpyro/util.py`: `fori_loop`/`fori_collect`/`cond`/`while_loop` wrappers that fall back to plain Python control flow inside the `control_flow_prims_disabled()` context (used for debugging and by tests, usually together with `jax.disable_jit()`), `soft_vmap`, `set_host_device_count`, `enable_x64`, `set_platform`, and model-checking helpers (`format_shapes`, `check_model_guide_match`).
- `examples/` and `notebooks/source/` are rendered into the docs (sphinx-gallery / nbsphinx); `test/test_examples.py` executes each script listed in its `EXAMPLES` list with small arguments, so a new example must be added there.
- New public API needs an entry in the matching `docs/source/*.rst` file to appear in the docs and be doctested.

## Conventions

- Version lives in `numpyro/version.py` (`scripts/update_version.py` bumps it everywhere).
- Tests compare against SciPy / analytic results with tolerances chosen for float32 (the default); use `JAX_ENABLE_X64=1` only where a test is explicitly x64.
- PRs target `master` on `pyro-ppl/numpyro`. Recent commit subjects use a conventional-style prefix, e.g. `fix(distributions): ...`, `doc(gh-2187): ...`, `refactor: ...`.
