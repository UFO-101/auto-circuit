## Prune Scores

The most important data structure to understand in AutoCircuit is the
[`PruneScores`][auto_circuit.types.PruneScores] object. This type is a map:
```
Dict[str, Tensor]
```
Where the keys are the `module_name`s of the [`DestNode`][auto_circuit.types.DestNode]s
in the model and the values are tensors with entries corresponding to the attribution
scores for each edge that points to that node.

You can access the score for a particular [`Edge`][auto_circuit.types.Edge] by indexing
into the tensor at index given by the [patch_idx][auto_circuit.types.Edge.patch_idx] of
the edge.

```python
score = prune_scores[edge.dest.module_name][edge.patch_idx]
```

## Patch Masks

Each [`DestNode`][auto_circuit.types.DestNode] is wrapped by a
[`PatchWrapper`][auto_circuit.utils.patch_wrapper.PatchWrapperImpl] that contains a
`patch_mask` Pytorch `Parameter`. This tensor corresponds exactly to the tensor in the
[`PruneScores`][auto_circuit.types.PruneScores] object that is indexed by the
[`DestNode`][auto_circuit.types.DestNode] `module_name`.

The value of the `patch_mask` for each edge interpolates between the default value of
the edge in the current forward pass and the value of the edge in `patch_src_outs` when
the [`patch_mode`][auto_circuit.utils.graph_utils.patch_mode] context manager is active.

There are helper functions to access the current mask value for a particular edge:
```python
score = edge.patch_mask(model).data[edge.patch_idx]
```

For a more thorough explanation of how patching works, see the
[announcement post](https://www.lesswrong.com/posts/caZ3yR5GnzbZe2yJ3/how-to-do-patching-fast)
for this library.
