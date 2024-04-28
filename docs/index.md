# AutoCircuit
A library for efficient patching and automatic circuit discovery.

![](../../assets/Edge_Patching_Rounded.png)

### Installation
```
pip install auto-circuit
```

## Easy and Efficient Edge Patching
```python
--8<-- "experiments/demos/zero_ablate_an_edge.py:20:27"
```

## Different Ablation Methods
```python
--8<-- "experiments/demos/patch_an_edge.py:36:37"
```

## Automatic Circuit Discovery
```python
--8<-- "experiments/demos/patch_an_edge.py:49:56"
```

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
