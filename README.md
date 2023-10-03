# AutoCircuit
A library for testing automatic circuit discovery algorithms.

## Getting Started
- Install [poetry](https://python-poetry.org/docs/#installation)
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
        Make sure to set these environment variables in your shell config.
- Run `poetry install --with dev` to install dependencies

Poetry is configured to use system packages. This can be helpful when working on a cluster with PyTorch already available. To change this set `options.system-site-packages` to `false` in [poetry.toml](poetry.toml).

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

The code is written in a functional style as far as possible. This means that there should be no global state and no side effects. This means not writing classes except frozen dataclasses (which are essentially just structs) and not using variables outside of functions. Functions should just take in data and return data. The major exception to this is the patching code which injects modules into the main models and patches based on patch_mask instance variables. We use context managers to ensure that state remains local to each function.
