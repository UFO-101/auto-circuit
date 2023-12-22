# This is supposed to be an exact replication of the technique introduced in
# Attribution Patching Outperforms Automated Circuit Discovery (https://arxiv.org/abs/2310.10348)


import torch as t
import transformer_lens as tl

from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import (
    set_all_masks,
)
from auto_circuit.utils.misc import batch_avg_answer_diff, batch_avg_answer_val


def edge_attribution_patching_prune_scores(
    task: Task,
    answer_diff: bool = True,
) -> PruneScores:
    """Prune scores by Edge Attribution patching."""
    model = task.model
    assert isinstance(model.wrapped_model, tl.HookedTransformer)
    out_slice = model.out_slice

    set_all_masks(model, val=0.0)
    model.train()
    model.zero_grad()
    for batch in task.train_loader:

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
            loss = batch_avg_answer_diff(logits, batch)
        else:
            loss = batch_avg_answer_val(logits, batch)
        loss.backward()
        model.reset_hooks()

        logits, corrupt_cache = model.run_with_cache(
            batch.corrupt, return_type="logits"
        )
        logits, clean_cache = model.run_with_cache(batch.clean, return_type="logits")

        prune_scores = {}
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
            prune_scores[edge] = (grad * (src_clean_act - src_corrupt_act)).sum()
    model.eval()
    return prune_scores  # type: ignore
