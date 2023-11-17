import os
from typing import Set

import torch as t
import transformer_lens as tl
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.types import Edge, PatchType
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def test_kl_vs_edges(
    mini_tl_transformer: tl.HookedTransformer,
    mini_tl_dataloader: DataLoader[PromptPairBatch],
):
    """Test that experiments satisfy basic requirements with a real model."""
    model = mini_tl_transformer
    prepare_model(model, factorized=True, device="cpu", slice_output=True)
    edges: Set[Edge] = model.edges  # type: ignore
    test_loader = mini_tl_dataloader

    test_input = next(iter(test_loader))
    with t.inference_mode():
        clean_out = model(test_input.clean)[:, -1]
        corrupt_out = model(test_input.corrupt)[:, -1]

    prune_scores = random_prune_scores(model, test_loader)
    test_edge_counts = edge_counts_util(edges, [0.0, 5, 1.0])
    pruned_outs = run_pruned(
        model=model,
        data_loader=test_loader,
        test_edge_counts=test_edge_counts,
        prune_scores=prune_scores,
        patch_type=PatchType.PATH_PATCH,
        render_graph=False,
    )
    assert t.allclose(corrupt_out, pruned_outs[0][0], atol=1e-3)
    assert not t.allclose(clean_out, pruned_outs[5][0], atol=1e-3)
    assert t.allclose(clean_out, pruned_outs[len(edges)][0], atol=1e-3)
