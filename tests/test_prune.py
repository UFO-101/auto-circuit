#%%
import os
from typing import Dict, Optional

import pytest
import torch as t

from auto_circuit.data import (
    PromptDataLoader,
)
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune import run_circuits
from auto_circuit.types import Edge, PatchType, PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.patchable_model import PatchableModel
from tests.conftest import DEVICE

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.mark.parametrize("seq_len", [None, 3])
def test_pruning(
    micro_model: MicroModel,
    micro_dataloader: PromptDataLoader,
    seq_len: Optional[int],
    show_graphs: bool = False,
):
    """Check that pruning works by pruning a "MicroModel" where the correct output can
    be worked out by hand.

    To visualize, set render_graph=True in run_pruned.
    """
    model: PatchableModel = patchable_model(
        model=micro_model,
        factorized=True,
        slice_output="last_seq",
        seq_len=seq_len,
        separate_qkv=True,
        device=DEVICE,
    )
    test_loader = micro_dataloader

    test_batch = next(iter(test_loader))
    with t.inference_mode():
        clean_out = model(test_batch.clean)
        corrupt_out = model(test_batch.corrupt)

    assert t.allclose(clean_out[:, -1].cpu(), t.tensor([[25.0, 49.0]]))
    assert t.allclose(corrupt_out[:, -1].cpu(), t.tensor([[-25.0, -49.0]]))

    edge_dict = dict([((edge.seq_idx, edge.name), edge) for edge in model.edges])

    seq_idx = None if seq_len is None else seq_len - 1
    prune_scores: PruneScores = model.new_prune_scores()
    prune_edges: Dict[Edge, float] = {
        edge_dict[(seq_idx, "B0.1->Resid End")]: 3.0,
        edge_dict[(seq_idx, "B0.0->B1.1")]: 2.0,
        edge_dict[(seq_idx, "Resid Start->B0.0")]: 1.0,
    }
    for edge, score in prune_edges.items():
        prune_scores[edge.dest.module_name][edge.patch_idx] = score

    circ_outs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=[0, 1, 2, 3],
        prune_scores=prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        render_graph=show_graphs,
    )
    key = test_batch.key
    assert t.allclose(circ_outs[0][key], corrupt_out[:, -1], atol=1e-3)
    assert t.allclose(circ_outs[1][key].cpu(), t.tensor([[-19.0, -41.0]]), atol=1e-3)
    assert t.allclose(circ_outs[2][key].cpu(), t.tensor([[-13.0, -25.0]]), atol=1e-3)
    assert t.allclose(circ_outs[3][key].cpu(), t.tensor([[-9.0, -13.0]]), atol=1e-3)


# model = micro_model()
# dataloader = micro_dataloader()
# test_pruning(model, dataloader, seq_len=None, show_graphs=True)


def test_prune_sequence(
    micro_model: MicroModel,
    micro_dataloader: PromptDataLoader,
    show_graphs: bool = False,
):
    """Test pruning different positions in the sequence."""
    model: PatchableModel = patchable_model(
        model=micro_model, factorized=True, slice_output=None, seq_len=3, device=DEVICE
    )
    test_loader = micro_dataloader

    test_batch = next(iter(test_loader))
    with t.inference_mode():
        corrupt_out = model(test_batch.corrupt)[model.out_slice]
    # assert t.allclose(clean_out[:, -1], t.tensor([[25.0, 49.0]]))

    edge_dict = dict([((edge.seq_idx, edge.name), edge) for edge in model.edges])

    prune_scores: PruneScores = model.new_prune_scores()
    prune_edges: Dict[Edge, float] = {
        edge_dict[(2, "B0.1->Resid End")]: 3.0,
        edge_dict[(2, "B0.0->B1.1")]: 2.0,
        edge_dict[(0, "Resid Start->Resid End")]: 1.0,
    }
    for edge, score in prune_edges.items():
        prune_scores[edge.dest.module_name][edge.patch_idx] = score

    outs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=[0, 1, 2, 3],
        prune_scores=prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        render_graph=show_graphs,
    )
    key = test_batch.key
    assert t.allclose(outs[0][key], corrupt_out, atol=1e-3)
    assert t.allclose(outs[1][key][:, 2].cpu(), t.tensor([[-19.0, -41.0]]), atol=1e-3)
    assert t.allclose(outs[2][key][:, 2].cpu(), t.tensor([[-13.0, -25.0]]), atol=1e-3)
    assert t.allclose(outs[3][key][:, 0].cpu(), t.tensor([[-69.0, -141.0]]), atol=1e-3)


# test_prune_sequence(model, dataloader, show_graphs=True)
