import os

import torch as t
import transformer_lens as tl
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.types import ActType, ExperimentType
from auto_circuit.utils.graph_utils import edge_counts_util, graph_edges

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def test_kl_vs_edges(
    mini_tl_transformer: tl.HookedTransformer,
    mini_tl_dataloader: DataLoader[PromptPairBatch],
):
    """Test that experiments satisfy basic requirements with a real model."""
    model = mini_tl_transformer
    test_loader = mini_tl_dataloader
    factorized = True

    experiment_type = ExperimentType(
        input_type=ActType.CLEAN, patch_type=ActType.CORRUPT
    )

    test_input = next(iter(test_loader))
    with t.inference_mode():
        clean_out = model(test_input.clean)[:, -1]
        corrupt_out = model(test_input.corrupt)[:, -1]

    prune_scores = random_prune_scores(model, factorized, test_loader)
    test_edge_counts = edge_counts_util(model, factorized, [0.0, 5, 1.0])
    pruned_outs = run_pruned(
        model, factorized, test_loader, experiment_type, test_edge_counts, prune_scores
    )
    assert t.allclose(clean_out, pruned_outs[0][0], atol=1e-3)
    assert not t.allclose(corrupt_out, pruned_outs[5][0], atol=1e-3)
    edges = graph_edges(model, factorized)
    assert t.allclose(corrupt_out, pruned_outs[len(edges)][0], atol=1e-3)
