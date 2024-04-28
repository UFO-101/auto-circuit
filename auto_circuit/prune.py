from collections import defaultdict
from typing import List, Optional

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    PatchType,
    PatchWrapper,
    PruneScores,
)
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    patch_mode,
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
    render_all_edges: bool = False,
    render_file_path: Optional[str] = None,
) -> CircuitOutputs:
    """Run the model, pruning edges based on the given `prune_scores`. Runs the model
    over the given `dataloader` for each `test_edge_count`.

    Args:
        model: The model to run
        dataloader: The dataloader to use for input and patches
        test_edge_counts: The numbers of edges to prune.
        prune_scores: The scores that determine the ordering of edges for pruning
        patch_type: Whether to patch the circuit or the complement.
        ablation_type: The type of ablation to use.
        reverse_clean_corrupt: Reverse clean and corrupt (for input and patches).
        render_graph: Whether to render the graph using `draw_seq_graph`.
        render_all_edges: Whether to render all edges, if `render_graph` is `True`.
        render_file_path: Path to save the rendered graph, if `render_graph` is `True`.

    Returns:
        A dictionary mapping from the number of pruned edges to a
            [`BatchOutputs`][auto_circuit.types.BatchOutputs] object, which is a
            dictionary mapping from [`BatchKey`s][auto_circuit.types.BatchKey] to output
            tensors.
    """
    circ_outs: CircuitOutputs = defaultdict(dict)
    desc_ps: t.Tensor = desc_prune_scores(prune_scores)

    patch_src_outs: Optional[t.Tensor] = None
    if ablation_type.mean_over_dataset:
        patch_src_outs = src_ablations(model, dataloader, ablation_type)

    for batch_idx, batch in enumerate(batch_pbar := tqdm(dataloader)):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}", refresh=True)
        if (patch_type == PatchType.TREE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.EDGE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.clean
            if not ablation_type.mean_over_dataset:
                patch_src_outs = src_ablations(model, batch.corrupt, ablation_type)
        elif (patch_type == PatchType.EDGE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.TREE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.corrupt
            if not ablation_type.mean_over_dataset:
                patch_src_outs = src_ablations(model, batch.clean, ablation_type)
        else:
            raise NotImplementedError

        assert patch_src_outs is not None
        with patch_mode(model, patch_src_outs):
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
                    show_all_edges=render_all_edges,
                    show_all_seq_pos=False,
                    seq_labels=dataloader.seq_labels,
                    file_path=render_file_path,
                )
    del patch_src_outs
    return circ_outs
