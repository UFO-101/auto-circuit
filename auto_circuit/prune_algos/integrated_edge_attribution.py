from typing import Dict

import torch as t
import transformer_lens as tl

from auto_circuit.tasks import Task
from auto_circuit.types import Edge
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    set_all_masks,
)
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff


def integrated_edge_attribution_prune_scores(
    task: Task,
    samples: int = 50,
) -> Dict[Edge, float]:
    """Prune scores by Edge Attribution patching."""
    model = task.model
    assert model.is_transformer
    out_slice = model.out_slice

    set_all_masks(model, val=0.0)
    model.train()
    model.zero_grad()
    for batch in task.train_loader:
        clean_grad_cache = {}

        def backward_cache_hook(act: t.Tensor, hook: tl.hook_points.HookPoint):
            if hook.name in clean_grad_cache:
                clean_grad_cache[hook.name] = act.detach() / samples
            else:
                clean_grad_cache[hook.name] += act.detach() / samples

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
        for _ in (ig_pbar := tqdm(range(samples))):
            logits = model(batch.clean)[out_slice]
            loss = batch_avg_answer_diff(logits, batch)
            loss.backward()
        model.reset_hooks()

        logits, corrupt_cache = model.run_with_cache(
            batch.corrupt, return_type="logits"
        )
        logits, clean_cache = model.run_with_cache(batch.clean, return_type="logits")

        prune_scores = {}
        for edge in task.model.edges:
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
