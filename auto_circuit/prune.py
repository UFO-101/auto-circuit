from collections import defaultdict
from typing import Dict, List, Optional

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import (
    Edge,
    PatchType,
    PrunedOutputs,
    SrcNode,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.visualize import draw_seq_graph


def run_pruned(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    test_edge_counts: List[int],
    prune_scores: Dict[Edge, float],
    patch_type: PatchType = PatchType.EDGE_PATCH,
    reverse_clean_corrupt: bool = False,
    render_graph: bool = False,
    render_prune_scores: bool = False,
    render_patched_edge_only: bool = True,
    render_top_n: int = 50,
    render_file_path: Optional[str] = None,
) -> PrunedOutputs:
    """Run the model with the given pruned edges.
    Tree Patching runs the clean input and patches corrupt activations into every edge
    _not_ in the circuit.
    Edge Patching runs the corrupt input and patches clean activations into every edge
    _in_ the circuit.
    Unless reverse_clean_corrupt is True, in which case clean and corrupt are swapped.
    """
    out_slice = model.out_slice
    pruned_outs: Dict[int, List[t.Tensor]] = defaultdict(list)
    edges: List[Edge] = list(
        sorted(prune_scores.keys(), key=lambda x: abs(prune_scores[x]), reverse=True)
    )

    for batch_idx, batch in enumerate(batch_pbar := tqdm(dataloader)):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}", refresh=True)
        if (patch_type == PatchType.TREE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.EDGE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.clean
            patch_outs = get_sorted_src_outs(model, batch.corrupt)
        elif (patch_type == PatchType.EDGE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.TREE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.corrupt
            patch_outs = get_sorted_src_outs(model, batch.clean)
        else:
            raise NotImplementedError

        # patch_outs: Dict[SrcNode, t.Tensor]
        # patch_src_outs: t.Tensor = t.stack(list(patch_outs.values())).detach()
        # patch_src_outs = t.stack(list(patch_outs.values()))
        # repeats = [patch_src_outs.shape[0]] + [1] * (patch_src_outs.ndim - 1)
        # patch_src_outs: t.Tensor = (
        #     patch_src_outs.mean(dim=0, keepdim=True).repeat(repeats).detach()
        # )

        patch_outs: Dict[SrcNode, t.Tensor]
        patch_src_outs: t.Tensor = t.stack(list(patch_outs.values())).detach()

        curr_src_outs: t.Tensor = t.zeros_like(patch_src_outs)

        patched_edge_val = 0.0 if patch_type == PatchType.TREE_PATCH else 1.0
        set_all_masks(model, val=1.0 if patch_type == PatchType.TREE_PATCH else 0.0)
        with patch_mode(model, curr_src_outs, patch_src_outs):
            if render_graph:
                draw_seq_graph(
                    model,
                    batch_input,
                    # Get the top 50 prune scores
                    prune_scores=dict(
                        sorted(
                            prune_scores.items(), key=lambda x: abs(x[1]), reverse=True
                        )[:render_top_n]
                    ),
                    show_prune_scores=render_prune_scores,
                    show_all_edges=not render_patched_edge_only,
                    seq_labels=dataloader.seq_labels,
                    file_path=render_file_path,
                )
            for edge_idx in (edge_pbar := tqdm(range(len(edges) + 1))):
                if edge_idx in test_edge_counts:
                    edge_pbar.set_description_str(f"Pruning {edge_idx} Edges")
                    with t.inference_mode():
                        model_output = model(batch_input)[out_slice]
                    pruned_outs[edge_idx].append(model_output.detach().clone())
                if edge_idx < len(edges):
                    edge = edges[edge_idx]
                    edge.patch_mask(model).data[edge.patch_idx] = patched_edge_val
        del patch_outs, patch_src_outs, curr_src_outs  # Free up memory
    return pruned_outs
