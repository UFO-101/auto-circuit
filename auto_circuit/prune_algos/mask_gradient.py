from typing import Dict, Literal, Optional

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

from auto_circuit.utils.custom_tqdm import tqdm

def mask_gradient_prune_scores(
    task: Task,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val", "mse"],
    mask_val: Optional[float] = None,
    integrated_grad_samples: Optional[int] = None,
) -> PruneScores:
    """Prune scores by attribution patching."""
    assert (mask_val is not None) ^ (integrated_grad_samples is not None)  # ^ means XOR
    model = task.model
    out_slice = model.out_slice

    src_outs_dict: Dict[BatchKey, t.Tensor] = {}
    for batch in task.train_loader:
        patch_outs = get_sorted_src_outs(model, batch.corrupt)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    with train_mask_mode(model):
        for sample in (ig_pbar:=tqdm(range((integrated_grad_samples or 0) + 1))):
            ig_pbar.set_description_str(f"Sample: {sample}")
            # Interpolate the mask value if integrating gradients. Else set the value.
            if integrated_grad_samples is not None:
                set_all_masks(model, val=sample / integrated_grad_samples)
            else:
                assert mask_val is not None and integrated_grad_samples is None
                set_all_masks(model, val=mask_val)

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

    prune_scores: PruneScores = {}
    for dest_wrapper in model.dest_wrappers:
        grad = dest_wrapper.patch_mask.grad
        assert grad is not None
        prune_scores[dest_wrapper.module_name] = grad.detach().clone()
    return prune_scores
