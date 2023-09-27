#%%
from typing import Any, Dict, List

import pytest
import torch as t
from ordered_set import OrderedSet
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune_functions.ACDC import acdc_edge_counts, acdc_prune_scores
from auto_circuit.types import ActType, Edge, ExperimentType, TensorIndex
from auto_circuit.utils.graph_utils import graph_edges


@pytest.mark.parametrize(
    "model, dataloader, output_slice",
    [
        ("micro_model", "micro_dataloader", slice(None)),
        ("mini_tl_transformer", "mini_tl_dataloader", (slice(None), -1)),
    ],
)
def test_acdc(
    model: t.nn.Module,
    dataloader: DataLoader[PromptPairBatch],
    output_slice: TensorIndex,
    request: Any,
    show_graphs: bool = False,  # Useful for debugging
):
    fixture_model = request.getfixturevalue(model)
    fixture_dataloader = request.getfixturevalue(dataloader)
    factorized = True
    acdc_prune_scores(
        fixture_model,
        factorized,
        fixture_dataloader,
        (1e-4, 1e-4),
        1e-4,
        output_slice,
        test_mode=True,
        show_graphs=show_graphs,
    )


import ipytest

ipytest.run("-q", "-s")
#%%


@pytest.mark.parametrize("decrease_prune_scores", [True, False])
def test_acdc_edge_counts(micro_model: MicroModel, decrease_prune_scores: bool):
    print("HI")
    model = micro_model
    edges: OrderedSet[Edge] = graph_edges(model, True)
    experiment_type = ExperimentType(
        ActType.CLEAN, ActType.CORRUPT, decrease_prune_scores
    )

    counts: List[int] = acdc_edge_counts(model, True, experiment_type, {})
    assert counts == []

    prune_scores: Dict[Edge, float] = {edges[0]: 0.0}
    counts: List[int] = acdc_edge_counts(model, True, experiment_type, prune_scores)
    assert counts == [1]

    prune_scores: Dict[Edge, float] = {edges[0]: 1.0, edges[1]: 1.0}
    counts: List[int] = acdc_edge_counts(model, True, experiment_type, prune_scores)
    assert counts == [2]

    prune_scores: Dict[Edge, float] = {edges[0]: 1.0, edges[1]: 1.0, edges[2]: 2.0}
    counts: List[int] = acdc_edge_counts(model, True, experiment_type, prune_scores)
    assert counts == [1, 3] if decrease_prune_scores else [2, 3]

    prune_scores: Dict[Edge, float] = {edges[0]: 1.0, edges[1]: 2.0, edges[2]: 2.0}
    counts: List[int] = acdc_edge_counts(model, True, experiment_type, prune_scores)
    assert counts == [2, 3] if decrease_prune_scores else [1, 3]
