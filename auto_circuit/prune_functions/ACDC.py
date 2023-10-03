from collections import defaultdict
from copy import deepcopy
from itertools import product
from random import random
from typing import Dict, List, Set

import torch as t
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

from auto_circuit.data import PromptPairBatch
from auto_circuit.prune import run_pruned
from auto_circuit.types import (
    ActType,
    Edge,
    ExperimentType,
    SrcNode,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    get_sorted_src_outs,
    patch_mode,
)
from auto_circuit.visualize import draw_graph


def acdc_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
    tao_exps: List[int] = list(range(-5, -1)),
    output_dim: int = 1,
    test_mode: bool = False,
    show_graphs: bool = False,
) -> Dict[Edge, float]:
    """Run the ACDC algorithm from the paper 'Towards Automated Circuit Discovery for
    Mechanistic Interpretability' (https://arxiv.org/abs/2304.14997).

    The algorithm does not assign scores to each edge, instead it finds the edges to be
    pruned given a certain value of tao. So we run the algorithm for several values of
    tao and give equal scores to all edges that are pruned for a given tao. Then we use
    test_edge_counts to pass edge counts to run_pruned such that all edges with the same
    score are pruned together.

    Note: only the first batch of train_data is used."""
    test_model = deepcopy(model) if test_mode else None
    output_idx = tuple([slice(None)] * output_dim + [-1])
    edges: Set[Edge] = model.edges  # type: ignore
    edges: List[Edge] = list(sorted(edges, key=lambda x: x.dest.layer, reverse=True))

    prune_scores = dict([(edge, float("inf")) for edge in edges])
    for tao in (
        pbar_tao := tqdm([a * 10**b for a, b in product([1, 3, 5, 7, 9], tao_exps)])
    ):
        pbar_tao.set_description_str("ACDC \u03C4={:.7f}".format(tao), refresh=True)

        train_batch = next(iter(train_data))
        clean_batch, corrupt_batch = train_batch.clean, train_batch.corrupt

        patch_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, corrupt_batch)
        src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, clean_batch)

        with t.inference_mode():
            clean_out = model(clean_batch)[output_idx]
            kv_cache, toks, short_embd, attn_mask, resids = None, None, None, None, []
            if isinstance(model, HookedTransformer):
                print("train_batch.diverge_idx:", train_batch.diverge_idx)
                common_prefix_batch = clean_batch[:, : train_batch.diverge_idx]
                kv_cache = HookedTransformerKeyValueCache.init_cache(
                    model.cfg, model.cfg.device, common_prefix_batch.shape[0]
                )
                model(common_prefix_batch, past_kv_cache=kv_cache)
                kv_cache.freeze()
                clean_batch = clean_batch[:, train_batch.diverge_idx :]
                corrupt_batch = corrupt_batch[:, train_batch.diverge_idx :]

                assert model.tokenizer is not None
                assert model.tokenizer.padding_side == "left"
                _, toks, short_embd, attn_mask = model.input_to_embed(
                    clean_batch, past_kv_cache=kv_cache
                )
                _, cache = model.run_with_cache(clean_batch, past_kv_cache=kv_cache)
                n_layers = range(model.cfg.n_layers)
                resids = [cache[f"blocks.{i}.hook_resid_pre"].clone() for i in n_layers]
                del cache
                patch_outs = get_sorted_src_outs(model, corrupt_batch, kv_cache)
                src_outs = get_sorted_src_outs(model, clean_batch, kv_cache=kv_cache)

        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)

        patch_outs_tensor = t.stack(list(patch_outs.values())).detach()
        src_outs_tensor = t.stack(list(src_outs.values())).detach()

        prev_kl_div = 0.0
        removed_edges: OrderedSet[Edge] = OrderedSet([])

        with patch_mode(model, src_outs_tensor, patch_outs_tensor, reset_mask=True):
            for edge_idx, edge in enumerate((pbar_edge := tqdm(edges))):
                rmvd, left = len(removed_edges), edge_idx + 1 - len(removed_edges)
                desc = f"Removed: {rmvd}, Left: {left}, Current:'{edge}'"
                pbar_edge.set_description_str(desc, refresh=False)

                edge.patch_mask(model).data[edge.patch_idx] = 1
                with t.inference_mode():
                    if isinstance(model, HookedTransformer):
                        start_layer = int(edge.dest.module_name.split(".")[1])
                        out = model(
                            resids[start_layer],
                            past_kv_cache=kv_cache,
                            start_at_layer=start_layer,
                            tokens=toks,
                            shortformer_pos_embed=short_embd,
                            attention_mask=attn_mask,
                        )[output_idx]
                    else:
                        out = model(clean_batch)[output_idx]
                out_logprobs = t.nn.functional.log_softmax(out, dim=-1)

                if test_mode:
                    assert test_model is not None
                    render = show_graphs and (random() < 0.1 or len(edges) < 20)

                    print("ACDC model, with out=", out) if render else None
                    d = dict([(e, patch_outs[e.src]) for e in (removed_edges | {edge})])
                    if render:
                        draw_graph(model, clean_batch, d, output_dim, kv_cache)

                    n_edges = len(removed_edges) + 1
                    print("Test mode: running pruned model") if render else None
                    test_out = run_pruned(
                        model=test_model,
                        data_loader=train_data,
                        experiment_type=ExperimentType(ActType.CLEAN, ActType.CORRUPT),
                        test_edge_counts=[n_edges],
                        prune_scores=dict([(e, 1.0) for e in (removed_edges | {edge})]),
                        include_zero_edges=False,
                        output_dim=output_dim,
                        render_graph=render,
                    )[n_edges][0]
                    test_out_logprobs = t.nn.functional.log_softmax(test_out, dim=-1)
                    print("Test_out:", test_out) if render else None
                    assert t.allclose(out_logprobs, test_out_logprobs, atol=1e-3)

                kl_div = t.nn.functional.kl_div(
                    out_logprobs, clean_logprobs, reduction="batchmean", log_target=True
                )
                mean_kl_div = kl_div.mean().item()
                if mean_kl_div - prev_kl_div < tao:
                    removed_edges.add(edge)
                    prune_scores[edge] = min(tao, prune_scores[edge])
                    prev_kl_div = mean_kl_div
                else:
                    edge.patch_mask(model).data[edge.patch_idx] = 0
    return prune_scores


def acdc_edge_counts(
    model: t.nn.Module,
    experiment_type: ExperimentType,
    prune_scores: Dict[Edge, float],
) -> List[int]:
    # Group prune_scores by score
    tao_counts: Dict[float, int] = defaultdict(int)
    for _, score in prune_scores.items():
        tao_counts[score] += 1
    # Sort tao_counts by tao
    reverse = experiment_type.decrease_prune_scores
    tao_counts = dict(sorted(tao_counts.items(), key=lambda x: x[0], reverse=reverse))
    # Create edge_counts
    edge_counts = []
    for _, count in tao_counts.items():
        prev_count = edge_counts[-1] if edge_counts else 0
        edge_counts.append(prev_count + count)
    return edge_counts_util(model, edge_counts)
