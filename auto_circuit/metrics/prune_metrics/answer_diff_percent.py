from typing import Any, List, Literal

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.tasks import Task
from auto_circuit.types import CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_diff_percent(
    task: Task,
    circuit_outs: CircuitOutputs,
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
) -> Measurements:
    """
    Measure the percentage change in [the difference between the answer and wrong answer
    value] between the default model and the circuits.
    """
    return answer_diff_percent(task.model, task.test_loader, circuit_outs, prob_func)


def answer_diff_percent(
    model: PatchableModel,
    test_loader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
) -> Measurements:
    measurements = []
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func == "logits"
        apply_prob_func = identity

    default_avg_ans_val: List[t.Tensor] = []
    for batch in test_loader:
        default_out = model(batch.clean)[model.out_slice]
        batch_val = apply_prob_func(default_out, dim=-1)
        default_avg_ans_val.append(batch_avg_answer_diff(batch_val, batch))
    default_avg_ans_diff = t.stack(default_avg_ans_val).mean().item()

    for edge_count, batch_outs in (pruned_out_pbar := tqdm(circuit_outs.items())):
        pruned_out_pbar.set_description_str(f"Answer Diff for {edge_count} edges")
        avg_ans_diffs = []
        for batch in test_loader:
            batch_probs = apply_prob_func(batch_outs[batch.key], dim=-1)
            avg_ans_diffs.append(batch_avg_answer_diff(batch_probs, batch))
        # PromptDataLoaders have all batches the same size, so we mean the batch means
        avg_ans_diff = t.stack(avg_ans_diffs).mean().item()
        measurements.append((edge_count, (avg_ans_diff / default_avg_ans_diff) * 100))
    return measurements
