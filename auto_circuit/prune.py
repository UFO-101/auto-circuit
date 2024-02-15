from collections import defaultdict
from typing import Dict, List, Optional

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    PatchType,
    PatchWrapper,
    PruneScores,
    SrcNode,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
)
from auto_circuit.utils.misc import module_by_name
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold
from auto_circuit.visualize import draw_seq_graph


def run_circuits(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    test_edge_counts: List[int],
    prune_scores: PruneScores,
    patch_type: PatchType = PatchType.EDGE_PATCH,
    ablation_type: AblationType = AblationType.RESAMPLE,
    reverse_clean_corrupt: bool = False,
    render_graph: bool = False,
    render_all_edges: bool = True,
    render_file_path: Optional[str] = None,
) -> CircuitOutputs:
    """Run the model with the given pruned edges.
    Tree Patching runs the clean input and patches corrupt activations into every edge
    _not_ in the circuit.
    Edge Patching runs the corrupt input and patches clean activations into every edge
    _in_ the circuit.
    Unless reverse_clean_corrupt is True, in which case clean and corrupt are swapped.
    """
    circ_outs: CircuitOutputs = defaultdict(dict)
    desc_ps: t.Tensor = desc_prune_scores(prune_scores)

    for batch_idx, batch in enumerate(batch_pbar := tqdm(dataloader)):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}", refresh=True)
        if (patch_type == PatchType.TREE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.EDGE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.clean
            patch_outs = get_sorted_src_outs(model, batch.corrupt, ablation_type)
        elif (patch_type == PatchType.EDGE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.TREE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.corrupt
            patch_outs = get_sorted_src_outs(model, batch.clean, ablation_type)
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

        set_all_masks(model, val=1.0 if patch_type == PatchType.TREE_PATCH else 0.0)
        with patch_mode(model, curr_src_outs, patch_src_outs):
            for edge_count in (edge_pbar := tqdm(test_edge_counts)):
                edge_pbar.set_description_str(f"Running Circuit: {edge_count} Edges")
                threshold = prune_scores_threshold(desc_ps, edge_count)
                # When prune_scores are tied we can't prune exactly edge_count edges
                patch_edge_count = 0
                for mod_name, patch_mask in prune_scores.items():
                    dest = module_by_name(model, mod_name)
                    assert isinstance(dest, PatchWrapper)
                    assert dest.is_dest and dest.patch_mask is not None
                    if patch_type == PatchType.EDGE_PATCH:
                        dest.patch_mask.data = (patch_mask.abs() >= threshold).float()
                        patch_edge_count += dest.patch_mask.int().sum().item()
                    else:
                        assert patch_type == PatchType.TREE_PATCH
                        dest.patch_mask.data = (patch_mask.abs() < threshold).float()
                        patch_edge_count += (1 - dest.patch_mask.int()).sum().item()
                with t.inference_mode():
                    model_output = model(batch_input)[model.out_slice]
                circ_outs[patch_edge_count][batch.key] = model_output.detach().clone()
            if render_graph:
                draw_seq_graph(
                    model=model,
                    input=batch_input,
                    show_all_edges=render_all_edges,
                    show_all_seq_pos=True,
                    seq_labels=dataloader.seq_labels,
                    file_path=render_file_path,
                )
        del patch_outs, patch_src_outs, curr_src_outs  # Free up memory
    return circ_outs
