from typing import Any, Dict, List, Literal

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Measurements
from auto_circuit.utils.misc import batch_avg_answer_diff


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_diff(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
    prob_func: Literal["log_softmax", "softmax", "logit"] = "logit",
) -> Measurements:
    probs = []
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func == "logit"
        apply_prob_func = identity

    for edge_count, pruned_out in pruned_outs.items():
        avg_ans_probs = []
        for batch_idx, batch in enumerate(test_loader):
            batch_probs = apply_prob_func(pruned_out[batch_idx], dim=-1)
            avg_ans_probs.append(batch_avg_answer_diff(batch_probs, batch))
        probs.append((edge_count, t.stack(avg_ans_probs).mean().item()))
    return probs
