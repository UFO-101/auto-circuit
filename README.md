# AutoCircuit
A library for testing automatic circuit discovery algorithms.

## Getting Started
- Install [poetry](https://python-poetry.org/docs/#installation)
- Run `poetry install --dev` to install dependencies
    - Poetry is configured to use system packages. This can be helpful when working on a cluster with PyTorch already available. You can change this in [poetry.toml](poetry.toml)
- Install [graphviz](https://graphviz.org/download/)
    - Linux:
        ```
        sudo apt-get update && sudo apt-get install libgl1-mesa-glx graphviz build-essential graphviz-dev
        ```
    - Mac:
        ```
        brew install graphviz
        export CFLAGS="-I$(brew --prefix graphviz)/include"
        export LDFLAGS="-L$(brew --prefix graphviz)/lib"
        ```

## Contributing
- [Pyright](https://github.com/microsoft/pyright) is used for type checking. Type hints are required for all functions.
- Tests are written with [Pytest](https://docs.pytest.org/en/stable/)
- [Black](https://github.com/psf/black) is used for formatting.
- Linting with [ruff](https://github.com/astral-sh/ruff).

To check / fix your code run:
```
pre-commit run --all-files
```
Install the git hook with:
```
pre-commit install
```
