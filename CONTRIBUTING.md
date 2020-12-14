# Development

Please follow our established coding style including variable names, module imports, and function definitions.
The NumPyro codebase follows the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
(which you can check with `make lint`) and follows
[`isort`](https://github.com/timothycrosley/isort) import order (which you can enforce with `make format`).

# Setup

To set up local development environment, install NumPyro from source:

```
git clone https://github.com/pyro-ppl/numpyro.git
# install jax/jaxlib first for CUDA support
pip install -e .[dev]  # contains additional dependencies for NumPyro development
```

# Testing

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally
```sh
make format            # runs isort
make test              # linting and unit tests
```

To run all tests locally in parallel, use the `pytest-xdist` package
```sh
pip install pytest-xdist
pytest -vs -n auto
```

To run a single test from the command line
```sh
pytest -vs {path_to_test}::{test_name}
# or in cuda mode
CUDA_TEST=1 PYRO_TENSOR_TYPE=torch.cuda.DoubleTensor pytest -vs {path_to_test}::{test_name}
```

# Submitting

For larger changes, please open an issue for discussion before submitting a pull request.

In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies

If you add new files, please run `make license` to automatically add copyright headers.

For speculative changes meant for early-stage review, include `[WIP]` in the PR's title. 
(One of the maintainers will add the `WIP` tag.)
