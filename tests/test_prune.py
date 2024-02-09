#%%
import os
from typing import Optional

import pytest
import torch as t

from auto_circuit.data import (
    PromptDataLoader,
)
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune import run_circuits
from auto_circuit.types import PatchType
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.patchable_model import PatchableModel

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.mark.parametrize("seq_len", [None, 3])
def test_pruning(
    micro_model: MicroModel,
    micro_dataloader: PromptDataLoader,
    seq_len: Optional[int],
    show_graphs: bool = False,
):
    """Check that pruning works by pruning a "MicroModel"
    where the correct output can be worked out by hand.

    To visualize, set render_graph=True in run_pruned."""
    model: PatchableModel = patchable_model(
        micro_model, True, True, "last_seq", seq_len=seq_len
    )
    test_loader = micro_dataloader

    test_batch = next(iter(test_loader))
    with t.inference_mode():
        clean_out = model(test_batch.clean)
        corrupt_out = model(test_batch.corrupt)

    assert t.allclose(clean_out[:, -1], t.tensor([[25.0, 49.0]]))
    assert t.allclose(corrupt_out[:, -1], t.tensor([[-25.0, -49.0]]))

    edge_dict = dict([((edge.seq_idx, edge.name), edge) for edge in model.edges])

    seq_idx = None if seq_len is None else seq_len - 1
    prune_scores = {
        edge_dict[(seq_idx, "Block 0 Head 1->Output")]: 3.0,
        edge_dict[(seq_idx, "Block 0 Head 0->Block 1 Head 1")]: 2.0,
        edge_dict[(seq_idx, "Input->Block 0 Head 0")]: 1.0,
    }

    pruned_outs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=[0, 1, 2, 3],
        prune_scores=prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        render_graph=show_graphs,
        render_patched_edge_only=False,
    )
    key = test_batch.key
    assert t.allclose(pruned_outs[0][key], corrupt_out[:, -1], atol=1e-3)
    assert t.allclose(pruned_outs[1][key], t.tensor([[-19.0, -41.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[2][key], t.tensor([[-13.0, -25.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[3][key], t.tensor([[-9.0, -13.0]]), atol=1e-3)


# micro_model = micro_model()
# micro_dataloader = micro_dataloader()
# test_pruning(micro_model, micro_dataloader, seq_len=None, show_graphs=True)


def test_prune_sequence(
    micro_model: MicroModel,
    micro_dataloader: PromptDataLoader,
    show_graphs: bool = False,
):
    """Test pruning different positions in the sequence."""
    model: PatchableModel = patchable_model(micro_model, True, False, None, seq_len=3)
    test_loader = micro_dataloader

    test_batch = next(iter(test_loader))
    with t.inference_mode():
        corrupt_out = model(test_batch.corrupt)[model.out_slice]
    # assert t.allclose(clean_out[:, -1], t.tensor([[25.0, 49.0]]))

    edge_dict = dict([((edge.seq_idx, edge.name), edge) for edge in model.edges])

    prune_scores = {
        edge_dict[(2, "Block 0 Head 1->Output")]: 3.0,
        edge_dict[(2, "Block 0 Head 0->Block 1 Head 1")]: 2.0,
        edge_dict[(0, "Input->Output")]: 1.0,
        # edge_dict[(2, "A0.0->A1.0.Q")]: 7.0,
        # edge_dict[(2, "A0.1->A1.0.Q")]: 6.0,
        # edge_dict[(2, "A1.0->Resid End")]: 4.0,
        # edge_dict[(2, "Resid Start->A1.1.Q")]: 4.0,
        # edge_dict[(0, "Resid Start->A0.0.Q")]: 3.0,
        # edge_dict[(2, "A1.1->Resid End")]: 2.0,
        # edge_dict[(2, "Resid Start->A0.1.Q")]: 1.0,
    }

    pruned_outs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=[0, 1, 2, 3],
        prune_scores=prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        render_graph=show_graphs,
        render_patched_edge_only=True,
        render_file_path="tree_patching.png",
    )
    key = test_batch.key
    assert t.allclose(pruned_outs[0][key], corrupt_out, atol=1e-3)
    assert t.allclose(pruned_outs[1][key][:, 2], t.tensor([[-19.0, -41.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[2][key][:, 2], t.tensor([[-13.0, -25.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[3][key][:, 0], t.tensor([[-69.0, -141.0]]), atol=1e-3)


# test_prune_sequence(micro_model, micro_dataloader, show_graphs=True)
