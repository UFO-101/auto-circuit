from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple

import torch as t
from ordered_set import OrderedSet
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.prune import path_patch_hook
from auto_circuit.types import Edge, SrcNode, TensorIndex
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    get_src_outs,
    graph_edges,
    graph_src_nodes,
    src_out_hook,
)
from auto_circuit.utils.misc import remove_hooks


def acdc_prune_scores(
    model: t.nn.Module,
    factorized: bool,
    train_data: DataLoader[PromptPairBatch],
    tao_range: Tuple[float, float] = (0.1, 0.9),
    tao_step: float = 0.1,
    output_idx: TensorIndex = (slice(None), -1),
) -> Dict[Edge, float]:
    """Run the ACDC algorithm from the paper 'Towards Automated Circuit Discovery for
    Mechanistic Interpretability' (https://arxiv.org/abs/2304.14997).

    The algorithm does not assign scores to each edge, instead it finds the edge to be
    pruned given a certain value of tao. So we run the algorithm for several values of
    tao and give equal scores to all edges that are pruned for a given tao. Then we use
    test_edge_counts so pass edge counts to run_pruned such that all edges with the same
    score are pruned together.

    Note: only the first batch of train_data is used."""
    edges: OrderedSet[Edge] = graph_edges(model, factorized, reverse_topo_sort=True)
    src_nodes = graph_src_nodes(model, factorized)

    tao_values = t.arange(tao_range[0], tao_range[1] + tao_step, tao_step)
    prune_scores = dict([(edge, float("inf")) for edge in edges])
    for tao in (pbar_tao := tqdm(tao_values)):
        tao = tao.item()
        pbar_tao.set_description_str("ACDC \u03C4={:.7f}".format(tao))

        train_batch = next(iter(train_data))
        clean_batch, corrupt_batch = train_batch.clean, train_batch.corrupt
        with t.inference_mode():
            clean_out = model(clean_batch)[output_idx]
        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)

        patch_outs: Dict[SrcNode, t.Tensor] = get_src_outs(
            model, src_nodes, corrupt_batch
        )
        src_outs: Dict[SrcNode, t.Tensor] = {}

        prev_kl_div = 0.0
        edge_kl_divs: Dict[Edge, float] = defaultdict(float)
        included_edges: OrderedSet[Edge] = OrderedSet([])
        with remove_hooks() as handles:
            for edge in (pbar_edge := tqdm(edges)):
                desc = f"ACDC Included ={len(included_edges)}, Current Edge='{edge}'"
                pbar_edge.set_description_str(desc)

                src_hook = partial(src_out_hook, edge_src=edge.src, src_outs=src_outs)
                hndl_1 = edge.src.module(model).register_forward_hook(src_hook)
                patch_hook = partial(
                    path_patch_hook,
                    edge=edge,
                    src_outs=src_outs,
                    patch_src_out=patch_outs[edge.src],
                )
                hndl_2 = edge.dest.module(model).register_forward_pre_hook(patch_hook)
                with t.inference_mode():
                    out = model(clean_batch)[output_idx]
                out_logprobs = t.nn.functional.log_softmax(out, dim=-1)

                kl_div = t.nn.functional.kl_div(
                    out_logprobs, clean_logprobs, reduction="batchmean", log_target=True
                )
                mean_kl_div = kl_div.mean().item()
                edge_kl_divs[edge] = mean_kl_div
                if mean_kl_div - prev_kl_div < tao:
                    included_edges.add(edge)
                    prune_scores[edge] = min(tao, prune_scores[edge])
                    handles.extend([hndl_1, hndl_2])
                    prev_kl_div = mean_kl_div
                else:
                    hndl_1.remove()
                    hndl_2.remove()
        # edge_labels = dict(
        #     [
        #         (
        #             e,
        #             f"idx: {edges.index(e)}\n"
        #             + (
        #                 f"\u03C4 {tao:.4f}: {included_edges.index(e)}\n"
        #                 if e in included_edges
        #                 else ""
        #             )
        #             + f" {edge_kl_divs[e]:.6f}",
        #         )
        #         for e in edges
        #     ]
        # )
        # draw_graph(
        #     model,
        #     factorized,
        #     next(iter(train_data)).clean,
        #     edge_label_override=edge_labels,
        # )
    return prune_scores


def acdc_edge_counts(
    model: t.nn.Module, factorized: bool, prune_scores: Dict[Edge, float]
) -> List[int]:
    # Group prune_scores by score
    prune_scores_count: Dict[float, int] = defaultdict(int)
    for _, score in prune_scores.items():
        prune_scores_count[score] += 1
    # Sort prune_scores by score
    prune_scores_count = dict(
        sorted(prune_scores_count.items(), key=lambda item: item[0])
    )
    # Create edge_counts
    edge_counts = []
    for _, count in prune_scores_count.items():
        prev_count = edge_counts[-1] if edge_counts else 0
        edge_counts += [prev_count + count]
    return edge_counts_util(model, factorized, edge_counts)
