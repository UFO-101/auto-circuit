from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import (
    ActType,
    Edge,
    ExperimentType,
    SrcNode,
    TensorIndex,
)
from auto_circuit.utils.graph_utils import (
    get_src_outs,
    graph_edges,
    graph_src_nodes,
    src_out_hook,
)
from auto_circuit.utils.misc import remove_hooks
from auto_circuit.visualize import draw_graph


def path_patch_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    edge: Edge,
    src_outs: Dict[SrcNode, t.Tensor],  # Dictionary is updated by other hook
    patch_src_out: t.Tensor,
) -> t.Tensor:
    src_out = src_outs[edge.src]
    assert len(input) == 1
    current_in = input[0].clone()
    current_in[edge.dest.in_idx] += patch_src_out - src_out
    return current_in


def run_pruned(
    model: t.nn.Module,
    factorized: bool,
    data_loader: DataLoader[PromptPairBatch],
    experiment_type: ExperimentType,
    test_edge_counts: List[int],
    prune_scores: Dict[Edge, float],
    include_zero_edges: bool = True,
    output_idx: TensorIndex = (slice(None), -1),
    render_graph: bool = False,
) -> Dict[int, List[t.Tensor]]:
    graph_edges(model, factorized)
    src_nodes = graph_src_nodes(model, factorized)
    pruned_outs: Dict[int, List[t.Tensor]] = defaultdict(list)

    def sort_fn(v: Tuple[Edge, float]) -> float:
        return -v[1] if experiment_type.sort_prune_scores_high_to_low else v[1]

    prune_scores = dict(sorted(prune_scores.items(), key=sort_fn))

    for batch_idx, batch in (
        # batch_pbar := tqdm(enumerate(data_loader), total=len(data_loader))
        batch_pbar := enumerate(data_loader)
    ):
        # batch_pbar.set_description_str(f"Pruning Batch {batch_idx}")
        if experiment_type.input_type == ActType.CLEAN:
            batch_input = batch.clean
        elif experiment_type.input_type == ActType.CORRUPT:
            batch_input = batch.corrupt
        else:
            raise NotImplementedError

        if include_zero_edges:
            with t.inference_mode():
                pruned_outs[0].append(model(batch_input)[output_idx])

        src_outs: Dict[SrcNode, t.Tensor] = {}
        patch_outs: Dict[SrcNode, t.Tensor]
        if experiment_type.patch_type == ActType.CLEAN:
            patch_outs = get_src_outs(model, src_nodes, batch.clean)
        elif experiment_type.patch_type == ActType.CORRUPT:
            patch_outs = get_src_outs(model, src_nodes, batch.corrupt)
        else:
            assert experiment_type.patch_type == ActType.ZERO
            patch_outs = get_src_outs(model, src_nodes, batch.clean)
            patch_outs = dict([(n, t.zeros_like(out)) for n, out in patch_outs.items()])

        with remove_hooks() as handles:
            edge_pbar = list(prune_scores.items())
            for edge_idx, (edge, _) in enumerate(edge_pbar):
                n_edges = edge_idx + 1
                # TODO: Registering duplicate hooks!!!!!
                src_hook = partial(src_out_hook, edge_src=edge.src, src_outs=src_outs)
                hndl_1 = edge.src.module(model).register_forward_hook(src_hook)
                patch_hook = partial(
                    path_patch_hook,
                    edge=edge,
                    src_outs=src_outs,
                    patch_src_out=patch_outs[edge.src],
                )
                hndl_2 = edge.dest.module(model).register_forward_pre_hook(patch_hook)
                handles.extend([hndl_1, hndl_2])
                if n_edges in test_edge_counts:
                    with t.inference_mode():
                        model_output = model(batch_input)
                    pruned_outs[n_edges].append(model_output[output_idx])
            if render_graph:
                d = dict([(e, patch_outs[e.src]) for e, _ in prune_scores.items()])
                draw_graph(model, factorized, batch_input, edge_label_override=d)
        del patch_outs, src_outs  # Free up memory
    return pruned_outs


def measure_kl_div(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
    output_idx: TensorIndex = (slice(None), -1),
) -> Tuple[Dict[int, float], ...]:
    # ) -> Dict[int, float]:
    kl_divs_clean, kl_divs_corrupt = {}, {}
    # Measure KL divergence
    with t.inference_mode():
        clean_outs = t.cat([model(batch.clean)[output_idx] for batch in test_loader])
        corrupt_outs = t.cat(
            [model(batch.corrupt)[output_idx] for batch in test_loader]
        )
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
