from copy import deepcopy
from itertools import product
from random import random
from typing import Callable, Dict, List, Literal, Optional

import torch as t
from ordered_set import OrderedSet

from auto_circuit.tasks import Task
from auto_circuit.types import (
    CircuitOutputs,
    Edge,
    PatchType,
    PruneScores,
    SrcNode,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
)
from auto_circuit.utils.tensor_ops import multibatch_kl_div


def acdc_prune_scores(
    task: Task,
    tao_exps: List[int] = list(range(-5, -1)),
    tao_bases: List[int] = [1, 3, 5, 7, 9],
    faithfulness_target: Literal["kl_div", "mse"] = "kl_div",
    test_mode: bool = False,
    run_pruned_ref: Optional[Callable[..., CircuitOutputs]] = None,
    show_graphs: bool = False,
    draw_seq_graph_ref: Optional[Callable[..., None]] = None,
) -> PruneScores:
    """Run the ACDC algorithm from the paper 'Towards Automated Circuit Discovery for
    Mechanistic Interpretability' (https://arxiv.org/abs/2304.14997).

    The algorithm does not assign scores to each edge, instead it finds the edges to be
    pruned given a certain value of tao. So we run the algorithm for several values of
    tao and give equal scores to all edges that are pruned for a given tao. Then we use
    test_edge_counts to pass edge counts to run_circuits such that all edges with the
    same score are pruned together.

    Note: only the first batch of train_data is used."""
    model = task.model
    test_model = deepcopy(model) if test_mode else None
    out_slice = model.out_slice
    edges: OrderedSet[Edge] = OrderedSet(
        sorted(model.edges, key=lambda x: x.dest.layer, reverse=True)
    )

    prune_scores = model.new_prune_scores(init_val=t.inf)
    for tao in (
        pbar_tao := tqdm([a * 10**b for a, b in product(tao_bases, tao_exps)])
    ):
        pbar_tao.set_description_str("ACDC \u03C4={:.7f}".format(tao), refresh=True)

        train_batch = next(iter(task.train_loader))
        clean_batch, corrupt_batch = train_batch.clean, train_batch.corrupt

        patch_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, corrupt_batch)
        src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, clean_batch)

        with t.inference_mode():
            clean_out = model(clean_batch)[out_slice]
            toks, short_embd, attn_mask, resids = None, None, None, []
            if model.is_transformer:
                _, toks, short_embd, attn_mask = model.input_to_embed(clean_batch)
                _, cache = model.run_with_cache(clean_batch)
                n_layers = range(model.cfg.n_layers)
                resids = [cache[f"blocks.{i}.hook_resid_pre"].clone() for i in n_layers]
                del cache

        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)

        patch_outs_tensor = t.stack(list(patch_outs.values())).detach()
        src_outs_tensor = t.stack(list(src_outs.values())).detach()

        prev_faith = 0.0
        removed_edges: OrderedSet[Edge] = OrderedSet([])

        set_all_masks(model, val=0.0)
        with patch_mode(model, src_outs_tensor, patch_outs_tensor):
            for edge_idx, edge in enumerate((pbar_edge := tqdm(edges))):
                rmvd, left = len(removed_edges), edge_idx + 1 - len(removed_edges)
                desc = f"Removed: {rmvd}, Left: {left}, Current:'{edge}'"
                pbar_edge.set_description_str(desc, refresh=False)

                edge.patch_mask(model).data[edge.patch_idx] = 1.0
                with t.inference_mode():
                    if model.is_transformer:
                        start_layer = int(edge.dest.module_name.split(".")[1])
                        out = model(
                            resids[start_layer],
                            start_at_layer=start_layer,
                            tokens=toks,
                            shortformer_pos_embed=short_embd,
                            attention_mask=attn_mask,
                        )[out_slice]
                    else:
                        out = model(clean_batch)[out_slice]

                if test_mode:
                    assert test_model is not None and run_pruned_ref is not None
                    render = show_graphs and (random() < 0.02 or len(edges) < 20)

                    print("ACDC model, with out=", out) if render else None
                    if render:
                        assert draw_seq_graph_ref is not None
                        draw_seq_graph_ref(model, clean_batch, None, True, True)

                    print("Test mode: running pruned model") if render else None
                    ps = dict(
                        [(m, (s == t.inf).float()) for m, s in prune_scores.items()]
                    )
                    ps[edge.dest.module_name][edge.patch_idx] = 0.0  # Not in circuit
                    n_edges = edge_counts_util(
                        model.edges, prune_scores=ps, zero_edges=False
                    )[0]
                    test_out = run_pruned_ref(
                        model=test_model,
                        dataloader=task.train_loader,
                        test_edge_counts=[n_edges],
                        prune_scores=ps,
                        patch_type=PatchType.TREE_PATCH,
                        render_graph=render,
                    )[n_edges][train_batch.key]
                    test_out_logprobs = t.nn.functional.log_softmax(test_out, dim=-1)
                    print("Test_out:", test_out) if render else None
                    assert t.allclose(out_logprobs, test_out_logprobs, atol=1e-3)

                if faithfulness_target == "kl_div":
                    out_logprobs = t.nn.functional.log_softmax(out, dim=-1)
                    faith = multibatch_kl_div(out_logprobs, clean_logprobs).mean().item()
                elif faithfulness_target == "mse":
                    faith = t.nn.functional.mse_loss(out, clean_out).item()

                if faith - prev_faith < tao:  # Edge is unimportant
                    removed_edges.add(edge)
                    curr = edge.prune_score(prune_scores)
                    prune_scores[edge.dest.module_name][edge.patch_idx] = min(tao, curr)
                    prev_faith = faith
                else:  # Edge is important - don't patch it
                    edge.patch_mask(model).data[edge.patch_idx] = 0.0
    return prune_scores
