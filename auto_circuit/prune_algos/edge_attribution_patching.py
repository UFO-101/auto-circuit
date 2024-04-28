from typing import Optional, Set

import torch as t
import transformer_lens as tl

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge, PruneScores
from auto_circuit.utils.graph_utils import (
    set_all_masks,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val


def edge_attribution_patching_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    answer_diff: bool = True,
) -> PruneScores:
    """
    Prune scores by Edge Attribution patching.

    This is an exact replication of the technique introduced in "Attribution Patching
    Outperforms Automated Circuit Discovery"
    [(Syed et al. (2023))](https://arxiv.org/abs/2310.10348), as
    implemented in
    [their codebase](https://github.com/Aaquib111/edge-attribution-patching/utils/prune_utils.py).

    It is equivalent to
    [`mask_gradient_prune_scores`][auto_circuit.prune_algos.mask_gradient.mask_gradient_prune_scores]
    with `grad_function="logit"` and `mask_val=0.0`. We verify that the output is
    exactly the same in `test_edge_attribution_patching.py`. This
    implementation is much slower, so we don't use it in practice, but it's useful for
    validating the correctness of the fast implementation.

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for input and ablation.
        official_edges: Not used.

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.

    Note:
        The implementation here uses `clean_act - corrupt_act`, as described in the
        paper, rather than `corrupt_act - clean_act`, as in author's implementation. It
        doesn't matter either way as we only consider the magnitude of the scores.
    """
    model = model
    assert model.is_transformer
    out_slice = model.out_slice

    set_all_masks(model, val=0.0)
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    model.zero_grad()
    prune_scores = model.new_prune_scores()

    for batch in dataloader:
        clean_grad_cache = {}

        def backward_cache_hook(act: t.Tensor, hook: tl.hook_points.HookPoint):
            clean_grad_cache[hook.name] = act.detach()

        incoming_ends = [
            "hook_q_input",
            "hook_k_input",
            "hook_v_input",
            f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
        ]
        if not model.cfg.attn_only:
            incoming_ends.append("hook_mlp_in")

        def edge_acdcpp_back_filter(name: str) -> bool:
            return name.endswith(tuple(incoming_ends + ["hook_q", "hook_k", "hook_v"]))

        model.add_hook(edge_acdcpp_back_filter, backward_cache_hook, "bwd")
        logits = model(batch.clean)[out_slice]
        if answer_diff:
            loss = -batch_avg_answer_diff(logits, batch)
        else:
            loss = -batch_avg_answer_val(logits, batch)
        loss.backward()
        model.reset_hooks()

        _, corrupt_cache = model.run_with_cache(batch.corrupt, return_type="logits")
        _, clean_cache = model.run_with_cache(batch.clean, return_type="logits")

        for edge in model.edges:
            if edge.dest.head_idx is None:
                grad = clean_grad_cache[edge.dest.module_name]
            else:
                grad = clean_grad_cache[edge.dest.module_name][:, :, edge.dest.head_idx]
            if edge.src.head_idx is None:
                src_clean_act = clean_cache[edge.src.module_name]
                src_corrupt_act = corrupt_cache[edge.src.module_name]
            else:
                src_clean_act = clean_cache[edge.src.module_name][
                    :, :, edge.src.head_idx
                ]
                src_corrupt_act = corrupt_cache[edge.src.module_name][
                    :, :, edge.src.head_idx
                ]
            assert grad is not None
            score = (grad * (src_corrupt_act - src_clean_act)).sum().item()
            prune_scores[edge.dest.module_name][edge.patch_idx] += score
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return prune_scores  # type: ignore
