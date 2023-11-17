from collections import defaultdict
from typing import Dict, List, Optional

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import (
    Edge,
    PatchType,
    SrcNode,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
)
from auto_circuit.visualize import draw_graph, draw_seq_graph


def run_pruned(
    model: t.nn.Module,
    data_loader: DataLoader[PromptPairBatch],
    test_edge_counts: List[int],
    prune_scores: Dict[Edge, float],
    patch_type: PatchType = PatchType.PATH_PATCH,
    render_graph: bool = False,
    render_patched_edge_only: bool = False,
    seq_labels: Optional[List[str]] = None,
    render_file_path: Optional[str] = None,
) -> Dict[int, List[t.Tensor]]:
    out_slice = model.out_slice
    pruned_outs: Dict[int, List[t.Tensor]] = defaultdict(list)
    prune_scores = dict(
        sorted(prune_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    for batch_idx, batch in enumerate(batch_pbar := tqdm(data_loader)):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}", refresh=True)
        if patch_type == PatchType.TREE_PATCH:
            batch_input = batch.clean
            patch_outs = get_sorted_src_outs(model, batch.corrupt)
        elif patch_type == PatchType.PATH_PATCH:
            batch_input = batch.corrupt
            patch_outs = get_sorted_src_outs(model, batch.clean)
        else:
            raise NotImplementedError

        if 0 in test_edge_counts:
            with t.inference_mode():
                pruned_outs[0].append(model(batch_input)[out_slice])

        patch_outs: Dict[SrcNode, t.Tensor]
        patch_src_outs: t.Tensor = t.stack(list(patch_outs.values())).detach()
        curr_src_outs: t.Tensor = t.zeros_like(patch_src_outs)

        patched_edge_val = 0.0 if patch_type == PatchType.TREE_PATCH else 1.0
        set_all_masks(model, val=1.0 if patch_type == PatchType.TREE_PATCH else 0.0)
        with patch_mode(model, curr_src_outs, patch_src_outs):
            for edge_idx, edge in enumerate(edge_pbar := tqdm(prune_scores.keys())):
                edge_pbar.set_description(f"Prune Edge {edge}", refresh=False)
                n_edge = edge_idx + 1
                edge.patch_mask(model).data[edge.patch_idx] = patched_edge_val
                if n_edge in test_edge_counts:
                    with t.inference_mode():
                        model_output = model(batch_input)
                    pruned_outs[n_edge].append(model_output[out_slice].detach().clone())
            if render_graph:
                draw_graph(model, batch_input)
                draw_seq_graph(
                    model,
                    batch_input,
                    prune_scores,
                    render_patched_edge_only,
                    seq_labels=seq_labels,
                    file_path=render_file_path,
                )
        del patch_outs, patch_src_outs, curr_src_outs  # Free up memory
    return pruned_outs
