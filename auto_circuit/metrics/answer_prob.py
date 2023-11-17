from typing import Any, Dict, List, Literal, Optional

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.misc import batch_avg_answer_val


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_prob(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
    prob_func: Optional[Literal["log_softmax", "softmax"]] = None,
) -> Dict[int, float]:
    probs = {}
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func is None
        apply_prob_func = identity

    for edge_count, pruned_out in pruned_outs.items():
        avg_ans_probs = []
        for batch_idx, batch in enumerate(test_loader):
            batch_probs = apply_prob_func(pruned_out[batch_idx], dim=-1)
            avg_ans_probs.append(batch_avg_answer_val(batch_probs, batch))
        probs[edge_count] = t.stack(avg_ans_probs).mean().item()
    return probs