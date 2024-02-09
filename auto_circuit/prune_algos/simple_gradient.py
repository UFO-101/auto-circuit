from typing import Dict, Literal

import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.tasks import Task
from auto_circuit.types import BatchKey, PruneScores
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val


def simple_gradient_prune_scores(
    task: Task,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val", "mse"],
    mask_val: float = 0.5,
) -> PruneScores:
    """Prune scores by attribution patching."""
    model = task.model
    out_slice = model.out_slice

    src_outs_dict: Dict[BatchKey, t.Tensor] = {}
    for batch in task.train_loader:
        patch_outs = get_sorted_src_outs(model, batch.corrupt)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    set_all_masks(model, val=mask_val)
    with train_mask_mode(model):
        for batch in task.train_loader:
            patch_src_outs = src_outs_dict[batch.key].clone().detach()
            with patch_mode(model, t.zeros_like(patch_src_outs), patch_src_outs):
                logits = model(batch.clean)[out_slice]
                if grad_function == "logit":
                    token_vals = logits
                elif grad_function == "prob":
                    token_vals = t.softmax(logits, dim=-1)
                elif grad_function == "logprob":
                    token_vals = log_softmax(logits, dim=-1)
                elif grad_function == "logit_exp":
                    numerator = t.exp(logits)
                    denominator = numerator.sum(dim=-1, keepdim=True)
                    token_vals = numerator / denominator.detach()
                else:
                    raise ValueError(f"Unknown grad_function: {grad_function}")
                if answer_function == "avg_diff":
                    loss = -batch_avg_answer_diff(token_vals, batch)
                elif answer_function == "avg_val":
                    loss = -batch_avg_answer_val(token_vals, batch)
                elif answer_function == "mse":
                    loss = t.nn.functional.mse_loss(token_vals, batch.answers)
                else:
                    raise ValueError(f"Unknown answer_function: {answer_function}")

                loss.backward()

    prune_scores = {}
    for edge in task.model.edges:
        grad = edge.patch_mask(model).grad
        assert grad is not None
        prune_scores[edge] = grad[edge.patch_idx].item()
    return prune_scores
