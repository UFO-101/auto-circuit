from typing import Any, Literal, Optional

import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import Measurements, PrunedOutputs, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import batch_avg_answer_val


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_val(
    task: Task,
    prune_scores: Optional[PruneScores],
    pruned_outs: Optional[PrunedOutputs],
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
) -> Measurements:
    assert pruned_outs is not None
    probs = []
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func == "logits"
        apply_prob_func = identity

    for edge_count, pruned_out in (pruned_out_pbar := tqdm(pruned_outs.items())):
        pruned_out_pbar.set_description_str(f"Answer Value for {edge_count} edges")
        avg_ans_probs = []
        for batch_idx, batch in enumerate(task.test_loader):
            batch_probs = apply_prob_func(pruned_out[batch_idx], dim=-1)
            avg_ans_probs.append(batch_avg_answer_val(batch_probs, batch))
        probs.append((edge_count, t.stack(avg_ans_probs).mean().item()))
    return probs
