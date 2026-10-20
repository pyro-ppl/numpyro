# Documentation

NumPyro documentation is built with [Sphinx](https://www.sphinx-doc.org/).
To build the docs, run from the top-level directory:

```sh
make docs
```

The generated HTML lands in `docs/build/html`.

## Installation

The documentation dependencies live in the `docs` dependency group of
`pyproject.toml`. Install them with:

```sh
uv sync --extra cpu --group docs
```

Building the docs also needs [pandoc](https://pandoc.org/installing.html),
which is not a Python package:

```sh
sudo apt install -y pandoc
```

## Workflow

To change the documentation, update the `*.rst` files in `source`.

To build the docstrings, `sphinx-apidoc [options] -o <output_path> <module_path> [exclude_pattern, ...]`

To build the HTML pages, run `make html` from this directory.

To check the examples embedded in docstrings, run `make doctest` from this
directory, or `make doctest` from the top-level directory.
