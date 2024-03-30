# AutoCircuit
A library for testing automatic circuit discovery algorithms.

## Read the draft paper
[Transformer Circuit Metrics are not Robust](Transformer%20Circuit%20Metrics%20are%20not%20Robust.pdf)

## Getting Started
- Install [poetry](https://python-poetry.org/docs/#installation)
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
To run the full test suite:
```
pytest --runslow
```

The code is written in a functional style as far as possible. This means that there should be no global state and no side effects. This means not writing classes except frozen dataclasses (which are essentially just structs) and not using variables outside of functions. Functions should just take in data and return data. The major exception to this is the patching code which injects modules into the main models and patches based on patch_mask instance variables. We use context managers to ensure that state remains local to each function.

## Running Experiments

An experiment is defined by a `Task`, `PruneAlgo` and `Metric`. A `Task` defines a behavior that a model can perform. A `PruneAlgo` (pruning algorithm) finds edges that perform the behavior. A `Metric` evaluates the performance of the model on the task after pruning the unimportant edges. Experiments are setup and performed in `experiments.py`.

### Tasks

Tasks are defined in `tasks.py`. They require a model and a dataset.
 - If `_model_def` is set to a string, then the `Task` object will try to load a TransformerLens model with that name, otherwise `_model_def` should just be the actual model object.
 - Datasets are defined by a `_dataset_name`, which should be the name of a JSON file in `/datasets` (excluding the `.json` extension). The JSON file should contain a list of `prompts` with `clean` and `corrupt` inputs and `correct` and `incorrect` outputs.

### PruneAlgos

Pruning Algorithms are defined in `prune_algos/prune_algos.py`. They require a function that takes a `Task` object and returns a `PruneScores` object, which is a dictionary mapping from `nn.Module` names to tensors. Each element of the tensor represents an edge from some SrcNode to some DestNode.

### Metrics

Metrics are defined in `metrics/`. These are usually functions that map a `Task` object along with a `PruneScores` or `CircuitOutputs` object to a list of `x,y` Measurements. (In prune_metrics/ `x` is the number of edges and `y` is some metric of faithfulness).

## Pruning

The core of the codebase implements edge patching in a flexible and efficient manner. The `Nodes` and `Edges` of a model are computed in `/model_utils.py`. `PatchWrapper` modules are injected at the `Node` positions (see `graph_utils.py`) and a `PatchableModel` is returned. When a `PatchableModel` is run in `patch_mode` the `PatchWrappers` at `SrcNodes` store their outputs in a shared object. And `DestNodes` compute their patched inputs by multiplying their `patch_masks` by the difference between the outputs of the incoming `SrcNodes` on this run, and on the input which is being patched in. This means that activation patching requires two passes. One forward pass computes the output of each `SrcNode` on the input to be patched in. The second pass adjusts the inputs to each patched `DestNode` to be the same as the first pass.
