from collections import defaultdict
from copy import deepcopy
from functools import partial
from random import random
from typing import Dict, List, Set, Tuple

import torch as t
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from auto_circuit.data import PromptPairBatch
from auto_circuit.prune import run_pruned
from auto_circuit.types import (
    ActType,
    Edge,
    ExperimentType,
    HashableTensorIndex,
    SrcNode,
    TensorIndex,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    PatchInput,
    add_tensor_idx_to_combined_idx,
    edge_counts_util,
    get_head_dim,
    get_src_outs,
    graph_edges,
    graph_src_nodes,
    tensor_idx_to_combined_idx,
)
from auto_circuit.utils.misc import remove_hooks, set_module_by_name
from auto_circuit.visualize import draw_graph


def update_src_out_tensor(
    module: t.nn.Module,
    input: t.Tensor,
    output: t.Tensor,
    src: SrcNode,
    src_outs_tensor: t.Tensor,
    patch_slice: TensorIndex,
):
    src_outs_tensor[src.idx] = output[src.out_idx][patch_slice]


def acdc_prune_scores(
    model: t.nn.Module,
    factorized: bool,
    train_data: DataLoader[PromptPairBatch],
    tao_range: Tuple[float, float] = (0.1, 0.9),
    tao_step: float = 0.1,
    output_idx: TensorIndex = (slice(None), -1),
    patch_slice: TensorIndex = slice(None),
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
    edges: OrderedSet[Edge] = graph_edges(model, factorized, reverse_topo_sort=True)
    src_nodes = graph_src_nodes(model, factorized)

    tao_values = t.arange(tao_range[0], tao_range[1] + tao_step, tao_step)
    prune_scores = dict([(edge, float("inf")) for edge in edges])
    for tao in (pbar_tao := tqdm(tao_values)):
        tao = tao.item()
        pbar_tao.set_description_str("ACDC \u03C4={:.7f}".format(tao), refresh=True)

        train_batch = next(iter(train_data))
        clean_batch, corrupt_batch = train_batch.clean, train_batch.corrupt
        with t.inference_mode():
            clean_out = model(clean_batch)[output_idx]
            toks, short_embd, left_attn_mask, resids = None, None, None, []
            if isinstance(model, HookedTransformer):
                assert (
                    model.tokenizer is not None
                    and model.tokenizer.padding_side == "left"
                )
                _, toks, short_embd, left_attn_mask = model.input_to_embed(clean_batch)
                _, cache = model.run_with_cache(clean_batch)
                n_layers = range(model.cfg.n_layers)
                resids = [cache[f"blocks.{i}.hook_resid_pre"].clone() for i in n_layers]
                del cache
        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)

        patch_outs: Dict[SrcNode, t.Tensor] = get_src_outs(
            model, src_nodes, corrupt_batch
        )
        patch_outs = dict(
            sorted(patch_outs.items(), key=lambda x: x[0].idx, reverse=False)
        )
        assert [src.idx for src in patch_outs.keys()] == list(range(len(patch_outs)))
        patch_outs_tensor = t.stack(
            [out[patch_slice] for out in patch_outs.values()]
        ).detach()  # [src, batch, resid]

        src_outs: Dict[SrcNode, t.Tensor] = get_src_outs(model, src_nodes, clean_batch)
        src_outs = dict(sorted(src_outs.items(), key=lambda x: x[0].idx, reverse=False))
        assert [src.idx for src in src_outs.keys()] == list(range(len(src_outs)))
        src_outs_tensor = t.stack(
            [out[patch_slice] for out in src_outs.values()]
        ).detach()

        prev_kl_div = 0.0
        removed_edges: OrderedSet[Edge] = OrderedSet([])
        hooked_srcs: Set[SrcNode] = set([])

        patched_heads: Dict[str, List[HashableTensorIndex]] = {}
        # with t.profiler.profile(
        #     schedule=t.profiler.schedule(wait=488, warmup=2, active=10),
        #     activities=[
        #         t.profiler.ProfilerActivity.CPU,
        #         t.profiler.ProfilerActivity.CUDA,
        #     ],
        #     on_trace_ready=t.profiler.tensorboard_trace_handler('./log/ACDC-4-improved-patch-module'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof, remove_hooks() as handles:
        with remove_hooks() as handles:
            for edge_idx, edge in enumerate((pbar_edge := tqdm(edges))):
                desc = f"Remvd: {len(removed_edges)}, Left: {edge_idx + 1 - len(removed_edges)}, Current Edge='{edge}'"
                pbar_edge.set_description_str(desc, refresh=False)

                new_hks = set([])
                if edge.src not in hooked_srcs:
                    # src_hk = partial(src_out_hook, edge_src=edge.src, src_outs=src_outs)
                    src_hk = partial(
                        update_src_out_tensor,
                        src=edge.src,
                        src_outs_tensor=src_outs_tensor,
                        patch_slice=patch_slice,
                    )
                    new_hks.add(edge.src.module(model).register_forward_hook(src_hk))
                if isinstance((dest_mod := edge.dest.module(model)), PatchInput):
                    if edge.dest._in_idx in patched_heads[edge.dest.module_name]:
                        if edge.dest.in_idx == slice(None):
                            dest_mod.srcs_to_patch[edge.src.idx] = 1.0
                        else:
                            dest_mod.srcs_to_patch[-1, edge.src.idx] = 1.0
                    else:
                        assert edge.dest.in_idx != slice(None)
                        new_srcs_to_patch = t.zeros(
                            1, src_outs_tensor.shape[0], device=clean_batch.device
                        )
                        new_srcs_to_patch[-1, edge.src.idx] = 1.0
                        dest_mod.srcs_to_patch = t.cat(
                            (dest_mod.srcs_to_patch, new_srcs_to_patch), dim=0
                        )
                        dest_mod.patch_idx = add_tensor_idx_to_combined_idx(
                            edge.dest.in_idx, dest_mod.patch_idx
                        )
                        patched_heads[edge.dest.module_name].append(edge.dest._in_idx)  # type: ignore
                else:
                    if edge.dest.in_idx == slice(None):
                        srcs_to_patch = t.zeros(
                            src_outs_tensor.shape[0], device=clean_batch.device
                        )
                        srcs_to_patch[edge.src.idx] = 1.0
                    else:
                        srcs_to_patch = t.zeros(
                            1, src_outs_tensor.shape[0], device=clean_batch.device
                        )
                        srcs_to_patch[-1, edge.src.idx] = 1.0

                    patch_mod = PatchInput(
                        module=dest_mod,
                        patch_idx=tensor_idx_to_combined_idx(edge.dest.in_idx),
                        srcs_to_patch=srcs_to_patch,
                        src_outs=src_outs_tensor,
                        patch_outs=patch_outs_tensor,
                        head_dim=get_head_dim(edge.dest),
                        patch_slice=patch_slice,
                    )
                    set_module_by_name(model, edge.dest.module_name, patch_mod)
                    patched_heads[edge.dest.module_name] = [edge.dest._in_idx]

                with t.inference_mode():
                    if isinstance(model, HookedTransformer):
                        out = model(
                            resids[edge.dest.layer],
                            start_at_layer=edge.dest.layer,
                            tokens=toks,
                            shortformer_pos_embed=short_embd,
                            left_attention_mask=left_attn_mask,
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
                        draw_graph(model, factorized, clean_batch, d, output_idx)

                    n_edges = len(removed_edges) + 1
                    print("Test mode: running pruned model") if render else None
                    test_out = run_pruned(
                        test_model,
                        factorized,
                        train_data,
                        ExperimentType(ActType.CLEAN, ActType.CORRUPT),
                        [n_edges],
                        dict([(e, 1.0) for e in (removed_edges | {edge})]),
                        include_zero_edges=False,
                        output_idx=output_idx,
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
                    hooked_srcs.add(edge.src)
                    prune_scores[edge] = min(tao, prune_scores[edge])
                    handles |= new_hks
                    prev_kl_div = mean_kl_div
                else:
                    [hk.remove() for hk in new_hks]
                    # del edge.dest.module(model).patches[edge]
                    if edge.dest.in_idx == slice(None):
                        edge.dest.module(model).srcs_to_patch[edge.src.idx] = 0.0
                    else:
                        edge.dest.module(model).srcs_to_patch[-1, edge.src.idx] = 0.0
                # prof.step()
    return prune_scores


def acdc_edge_counts(
    model: t.nn.Module,
    factorized: bool,
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
    return edge_counts_util(model, factorized, edge_counts)
