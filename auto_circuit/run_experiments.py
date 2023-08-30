from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch as t
from einops import repeat
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import (
    ActType,
    Edge,
    EdgeSrc,
    ExperimentType,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_acts, graph_edges
from auto_circuit.utils.misc import percent_gpu_mem_used


def update_current_acts_hook(
    model: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    edge_src: EdgeSrc,
    src_outs: Dict[EdgeSrc, t.Tensor],
):
    src_outs[edge_src] = output[edge_src.t_idx]


def path_patch_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    edge: Edge,
    src_outs: Dict[EdgeSrc, t.Tensor],  # Dictionary is updated by other hook
    patch_src_out: Optional[t.Tensor],  # None represents zero ablation
) -> t.Tensor:
    src_out = src_outs[edge.src]
    patch_src_out = t.zeros_like(src_out) if patch_src_out is None else patch_src_out
    assert len(input) == 1
    current_in = input[0]
    if current_in.ndim != src_out.ndim:  # Split by head
        head_dim = current_in.shape[-2]
        src_out = repeat(src_out, "b p r -> b p h r", h=head_dim)
        patch_src_out = repeat(patch_src_out, "b p r -> b p h r", h=head_dim)
    return current_in + (patch_src_out - src_out)


def run_pruned(
    model: t.nn.Module,
    factorized: bool,
    test_loader: DataLoader[PromptPairBatch],
    experiment_type: ExperimentType,
    test_edge_counts: List[int],
    prune_scores: Dict[Edge, float],
    include_zero_edges: bool = True,
) -> Dict[int, List[t.Tensor]]:
    edges = graph_edges(model, factorized)
    if experiment_type.patch_type == ActType.CLEAN:
        patch_acts = [edge_acts(model, edges, batch.clean) for batch in test_loader]
    elif experiment_type.patch_type == ActType.CORRUPT:
        patch_acts = [edge_acts(model, edges, batch.corrupt) for batch in test_loader]
    else:
        assert experiment_type.patch_type == ActType.ZERO
        patch_acts = [None for _ in test_loader]

    # Sort edges by prune score
    if experiment_type.sort_prune_scores_high_to_low:
        prune_scores = dict(sorted(prune_scores.items(), key=lambda item: -item[1]))
    else:
        prune_scores = dict(sorted(prune_scores.items(), key=lambda item: item[1]))

    pruned_outs: Dict[int, List[t.Tensor]] = defaultdict(list)
    for batch_idx, batch in (
        batch_pbar := tqdm(enumerate(test_loader), total=len(test_loader))
    ):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}")

        if experiment_type.input_type == ActType.CLEAN:
            batch_input = batch.clean
        elif experiment_type.input_type == ActType.CORRUPT:
            batch_input = batch.corrupt
        else:
            raise NotImplementedError

        if include_zero_edges:
            with t.inference_mode():
                pruned_outs[0].append(model(batch_input)[:, -1])

        handles = []
        src_outs: Dict[EdgeSrc, t.Tensor] = {}
        patch_src_outs: Optional[Dict[EdgeSrc, t.Tensor]] = patch_acts[batch_idx]

        try:
            edge_pbar = enumerate(list(prune_scores.items()))
            # edge_pbar = tqdm(enumerate(list(prune_scores.items())))
            for edge_idx, (edge, _) in edge_pbar:
                # edge_pbar.set_description_str(f"Pruning {edge}")
                n_edges = edge_idx + 1
                prev_src_out_hook = partial(
                    update_current_acts_hook,
                    edge_src=edge.src,
                    src_outs=src_outs,
                )
                hndl_1 = edge.src.module(model).register_forward_hook(prev_src_out_hook)
                patch_hook = partial(
                    path_patch_hook,
                    edge=edge,
                    src_outs=src_outs,
                    patch_src_out=None
                    if patch_src_outs is None
                    else patch_src_outs[edge.src],
                )
                hndl_2 = edge.dest.module(model).register_forward_pre_hook(patch_hook)
                handles.extend([hndl_1, hndl_2])
                if n_edges in test_edge_counts:
                    with t.inference_mode():
                        model_output = model(batch_input)
                    pruned_outs[n_edges].append(model_output[:, -1])
        finally:
            [handle.remove() for handle in handles]
    del patch_acts
    return pruned_outs


def measure_kl_div(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
) -> Tuple[Dict[int, float], ...]:
    # ) -> Dict[int, float]:
    kl_divs_clean, kl_divs_corrupt = {}, {}
    # Measure KL divergence
    with t.inference_mode():
        clean_outs = t.cat([model(batch.clean)[:, -1] for batch in test_loader])
        corrupt_outs = t.cat([model(batch.corrupt)[:, -1] for batch in test_loader])
    clean_logprobs = t.nn.functional.log_softmax(clean_outs, dim=-1)
    corrupt_logprobs = t.nn.functional.log_softmax(corrupt_outs, dim=-1)

    for edge_count, pruned_out in pruned_outs.items():
        pruned_out = t.cat(pruned_out)
        pruned_logprobs = t.nn.functional.log_softmax(pruned_out, dim=-1)
        kl_clean = t.nn.functional.kl_div(
            pruned_logprobs,
            clean_logprobs,
            reduction="batchmean",
            log_target=True,
        )
        kl_corrupt = t.nn.functional.kl_div(
            pruned_logprobs,
            corrupt_logprobs,
            reduction="batchmean",
            log_target=True,
        )
        # Numerical errors can cause tiny negative values in KL divergence
        kl_divs_clean[edge_count] = max(kl_clean.mean().item(), 0)
        kl_divs_corrupt[edge_count] = max(kl_corrupt.mean().item(), 0)
    return kl_divs_clean, kl_divs_corrupt
    # return kl_divs_clean, None
