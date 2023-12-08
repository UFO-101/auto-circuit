from copy import deepcopy
from itertools import product
from random import random
from typing import Dict, List, Set

import torch as t
from ordered_set import OrderedSet
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

from auto_circuit.data import PromptDataLoader
from auto_circuit.prune import run_pruned
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
from auto_circuit.visualize import draw_seq_graph


def acdc_prune_scores(
    model: t.nn.Module,
    train_data: PromptDataLoader,
    tao_exps: List[int] = list(range(-5, -1)),
    tao_bases: List[int] = [1, 3, 5, 7, 9],
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
    out_slice = model.out_slice
    edge_set: Set[Edge] = model.edges  # type: ignore
    edges: OrderedSet[Edge] = OrderedSet(
        sorted(edge_set, key=lambda x: x.dest.layer, reverse=True)
    )

    prune_scores = dict([(edge, float("inf")) for edge in edges])
    for tao in (
        pbar_tao := tqdm([a * 10**b for a, b in product(tao_bases, tao_exps)])
    ):
        pbar_tao.set_description_str("ACDC \u03C4={:.7f}".format(tao), refresh=True)

        train_batch = next(iter(train_data))
        clean_batch, corrupt_batch = train_batch.clean, train_batch.corrupt

        patch_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, corrupt_batch)
        src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, clean_batch)

        with t.inference_mode():
            clean_out = model(clean_batch)[out_slice]
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
                    clean_batch, past_kv_cache=kv_cache  # We patch edges not in graph
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

        set_all_masks(model, val=0.0)
        with patch_mode(model, src_outs_tensor, patch_outs_tensor):
            for edge_idx, edge in enumerate((pbar_edge := tqdm(edges))):
                rmvd, left = len(removed_edges), edge_idx + 1 - len(removed_edges)
                desc = f"Removed: {rmvd}, Left: {left}, Current:'{edge}'"
                pbar_edge.set_description_str(desc, refresh=False)

                edge.patch_mask(model).data[edge.patch_idx] = 1.0
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
                        )[out_slice]
                    else:
                        out = model(clean_batch)[out_slice]
                out_logprobs = t.nn.functional.log_softmax(out, dim=-1)

                if test_mode:
                    assert test_model is not None
                    render = show_graphs and (
                        random() < 0.02 or len(edges) < 20 or True
                    )

                    print("ACDC model, with out=", out) if render else None
                    tree = dict([(e, 1.0) for e in edges - (removed_edges | {edge})])
                    if render:
                        draw_seq_graph(model, clean_batch, tree, kv_cache=kv_cache)

                    print("Test mode: running pruned model") if render else None
                    test_out = run_pruned(
                        model=test_model,
                        dataloader=train_data,
                        test_edge_counts=[n_edges := len(tree)],
                        prune_scores=tree,
                        patch_type=PatchType.TREE_PATCH,
                        render_graph=render,
                    )[n_edges][0]
                    test_out_logprobs = t.nn.functional.log_softmax(test_out, dim=-1)
                    print("Test_out:", test_out) if render else None
                    assert t.allclose(out_logprobs, test_out_logprobs, atol=1e-3)

                kl_div = t.nn.functional.kl_div(
                    out_logprobs, clean_logprobs, reduction="batchmean", log_target=True
                )
                mean_kl_div = kl_div.mean().item()
                if mean_kl_div - prev_kl_div < tao:  # Edge is unimportant
                    removed_edges.add(edge)
                    prune_scores[edge] = min(tao, prune_scores[edge])
                    prev_kl_div = mean_kl_div
                else:  # Edge is important
                    edge.patch_mask(model).data[edge.patch_idx] = 0.0
    return prune_scores
