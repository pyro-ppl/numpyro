# Development

Please follow our established coding style including variable names, module imports, and function definitions.
The NumPyro codebase follows the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
(which you can check with `uv run make lint`) and follows
[`isort`](https://github.com/timothycrosley/isort) import order (which you can enforce with `uv run make format`).

# Setup

To set up a local development environment, install NumPyro from source with
[`uv`](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/pyro-ppl/numpyro.git
cd numpyro
uv sync --extra cpu --group dev --group test
```

The first command is sufficient for linting and unit tests. To also install
the documentation and examples dependencies needed for
`uv run make doctest`, run:

```sh
uv sync --extra cpu --group dev --group test --group docs --group examples
```

For CUDA support, replace `--extra cpu` with `--extra cuda12` or
`--extra cuda13`, as appropriate for your CUDA version.

For running `uv run make doctest`, [install pandoc](https://pandoc.org/installing.html).

# Testing

Before submitting a pull request, please autoformat the code and ensure that unit tests pass locally:
```sh
uv run make lint
uv run make format
uv run make test
uv run make doctest
```

To run all tests locally in parallel, use the `pytest-xdist` package:
```sh
uv run --with pytest-xdist pytest -vs -n auto
```

To run a single test from the command line:
```sh
uv run pytest -vs {path_to_test}::{test_name}
JAX_PLATFORM_NAME=gpu JAX_ENABLE_X64=1 uv run pytest -vs {path_to_test}::{test_name}
```

## Pre-Commit Hooks

For local development we recommend using [pre-commit](https://pre-commit.com/) hooks to automatically format your code before committing.

To install pre-commit hooks and use the local development tools for subsequent commits, run
```sh
uv run pre-commit install
source .venv/bin/activate
```

After each commit, pre-commit will run and verify that your code is formatted correctly. The pre-commit hooks can be skipped by adding the `--no-verify` flag to your `git commit` command.


# Profiling

TensorBoard can be used to profile NumPyro following the instructions following [JAX documentation](https://jax.readthedocs.io/en/latest/profiling.html).

# Submitting

For relevant design questions to consider, see past [design documents](https://github.com/pyro-ppl/pyro/wiki/Design-Docs).

For larger changes, please open an issue for discussion before submitting a pull request.

In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies

If you add new files, please run `uv run make license` to automatically add copyright headers.

For speculative changes meant for early-stage review, include `[WIP]` in the PR's title.
(One of the maintainers will add the `WIP` tag.)
