## Ablation types
- Resample (aka. patching)
- Zero
- Mean (calculated over a batch or [PromptDataset][auto_circuit.data.PromptDataset])

See [AblationType][auto_circuit.types.AblationType] for more details.

```python
--8<-- "experiments/demos/other_features.py:38:38"
```

## Automatic KV caching
When `tail_divergence` is `True`,
[`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] automatically
computes the KV Cache for the common prefix of all of the prompts in the dataset and
removes the prefix from the prompts.

```python
--8<-- "experiments/demos/other_features.py:19:28"
```

The KV Caches are stored in the `kv_cache` attribute of the
[`PromptDataloader`][auto_circuit.data.PromptDataLoader]s. Pass the caches to the
[`patchable_model`][auto_circuit.utils.graph_utils.patchable_model] function to use them
automatically.

```python
--8<-- "experiments/demos/other_features.py:29:37"
```

## Automatically patch multiple circuits
To patch multiple circuits of increasing size (decreasing
[`PruneScores`][auto_circuit.types.PruneScores]), use the
[`run_circuits`][auto_circuit.prune.run_circuits] function.

```python
--8<-- "experiments/demos/other_features.py:50:58"
```

## Measure circuit metrics
For a full list of metrics, see the
[reference documentation][auto_circuit.metrics.prune_metrics.kl_div].
```python
--8<-- "experiments/demos/other_features.py:59:59"
```
