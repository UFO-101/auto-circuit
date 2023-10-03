#%%
from typing import Any, Dict, List

import pytest
import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune_functions.ACDC import acdc_edge_counts, acdc_prune_scores
from auto_circuit.types import ActType, Edge, ExperimentType
from auto_circuit.utils.graph_utils import prepare_model


@pytest.mark.parametrize(
    "model, dataloader, output_dim",
    [
        ("micro_model", "micro_dataloader", 0),
        ("mini_tl_transformer", "mini_tl_dataloader", 1),
    ],
)
def test_acdc(
    model: t.nn.Module,
    dataloader: DataLoader[PromptPairBatch],
    output_dim: int,
    request: Any,
    show_graphs: bool = False,  # Useful for debugging
):
    fixture_model = request.getfixturevalue(model) if request else model
    prepare_model(fixture_model, factorized=True, device="cpu")
    fixture_dataloader = request.getfixturevalue(dataloader) if request else dataloader
    acdc_prune_scores(
        model=fixture_model,
        train_data=fixture_dataloader,
        tao_exps=[-3],
        output_dim=output_dim,
        test_mode=True,  # The actual test logic is embedded in the function
        show_graphs=show_graphs,
    )


# model = micro_model()
# model = mini_tl_transformer()
# dataloader = micro_dataloader()
# dataloader = mini_tl_dataloader()
# output_dim = 0
# output_dim = 1
# request = None
# show_graphs = False
# test_acdc(model, dataloader, output_dim, request, show_graphs)
#%%


@pytest.mark.parametrize("decrease_prune_scores", [True, False])
def test_acdc_edge_counts(micro_model: MicroModel, decrease_prune_scores: bool):
    model = micro_model
    prepare_model(model, factorized=True, device="cpu")
    edges: List[Edge] = list(model.edges)  # type: ignore
    experiment_type = ExperimentType(
        ActType.CLEAN, ActType.CORRUPT, decrease_prune_scores
    )

    counts: List[int] = acdc_edge_counts(model, experiment_type, {})
    assert counts == []

    prune_scores: Dict[Edge, float] = {edges[0]: 0.0}
    counts: List[int] = acdc_edge_counts(model, experiment_type, prune_scores)
    assert counts == [1]

    prune_scores: Dict[Edge, float] = {edges[0]: 1.0, edges[1]: 1.0}
    counts: List[int] = acdc_edge_counts(model, experiment_type, prune_scores)
    assert counts == [2]

    prune_scores: Dict[Edge, float] = {edges[0]: 1.0, edges[1]: 1.0, edges[2]: 2.0}
    counts: List[int] = acdc_edge_counts(model, experiment_type, prune_scores)
    assert counts == [1, 3] if decrease_prune_scores else [2, 3]

    prune_scores: Dict[Edge, float] = {edges[0]: 1.0, edges[1]: 2.0, edges[2]: 2.0}
    counts: List[int] = acdc_edge_counts(model, experiment_type, prune_scores)
    assert counts == [2, 3] if decrease_prune_scores else [1, 3]
