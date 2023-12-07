from typing import Dict, Literal, Set

import torch as t
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.misc import batch_avg_answer_diff, batch_avg_answer_val


def simple_gradient_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_diff: bool = False,
    mask_val: float = 0.5,
) -> Dict[Edge, float]:
    """Prune scores by attribution patching."""
    edges: Set[Edge] = model.edges  # type: ignore
    out_slice = model.out_slice

    src_outs_dict: Dict[int, t.Tensor] = {}
    for batch in train_data:
        patch_outs = get_sorted_src_outs(model, batch.clean)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    set_all_masks(model, val=mask_val)
    with train_mask_mode(model):
        for batch in train_data:
            patch_src_outs = src_outs_dict[batch.key].clone().detach()
            with patch_mode(model, t.zeros_like(patch_src_outs), patch_src_outs):
                logits = model(batch.corrupt)[out_slice]
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
                if answer_diff:
                    loss = batch_avg_answer_diff(token_vals, batch)
                else:
                    loss = batch_avg_answer_val(token_vals, batch)
                loss.backward()

    prune_scores = {}
    for edge in edges:
        grad = edge.patch_mask(model).grad
        assert grad is not None
        prune_scores[edge] = grad[edge.patch_idx]
    return prune_scores
