from typing import Any, Dict, List, Literal, Tuple

import torch as t

from auto_circuit.data import BatchKey, PromptDataLoader
from auto_circuit.tasks import Task
from auto_circuit.types import CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import (
    batch_answer_diff_percents,
    batch_avg_answer_diff,
)


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_diff_percent(
    task: Task,
    circuit_outs: CircuitOutputs,
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
    diff_of_means: bool = True,
) -> Measurements:
    """
    Measure the percentage change in [the difference between the answer and wrong answer
    value] between the default model and the circuits.
    """
    return answer_diff_percent(
        task.model, task.test_loader, circuit_outs, prob_func, diff_of_means
    )[0]


def answer_diff_percent(
    model: PatchableModel,
    test_loader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
    diff_of_means: bool = True,
) -> Tuple[Measurements, Measurements, List[Tuple[int, t.Tensor]]]:
    means: Measurements = []
    standard_devs: Measurements = []
    points: List[Tuple[int, t.Tensor]] = []
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func == "logits"
        apply_prob_func = identity

    batch_default_probs: Dict[BatchKey, t.Tensor] = {}
    for batch in test_loader:
        default_out = model(batch.clean)[model.out_slice]
        batch_val = apply_prob_func(default_out, dim=-1)
        batch_default_probs[batch.key] = batch_val

    for edge_count, batch_outs in (pruned_out_pbar := tqdm(circuit_outs.items())):
        pruned_out_pbar.set_description_str(f"Answer Diff for {edge_count} edges")
        # PromptDataLoaders have all batches the same size, so we mean the batch means
        if diff_of_means:
            pred_answer_diffs, target_answer_diffs = [], []
            for batch in test_loader:
                circ_probs = apply_prob_func(batch_outs[batch.key], dim=-1)
                default_probs = batch_default_probs[batch.key]
                pred_answer_diffs.append(batch_avg_answer_diff(circ_probs, batch))
                target_answer_diffs.append(batch_avg_answer_diff(default_probs, batch))
            mean_pred_diff = t.stack(pred_answer_diffs).mean().item()
            mean_target_diff = t.stack(target_answer_diffs).mean().item()
            means.append((edge_count, (mean_pred_diff / mean_target_diff) * 100))
            standard_devs.append((edge_count, 0.0))
            points.append((edge_count, t.tensor([])))
        else:
            ans_diff_percents = []
            for batch in test_loader:
                circ_probs = apply_prob_func(batch_outs[batch.key], dim=-1)
                default_probs = batch_default_probs[batch.key]
                circ_perc = batch_answer_diff_percents(circ_probs, default_probs, batch)
                ans_diff_percents.append(circ_perc)
            ans_diff_percents = t.cat(ans_diff_percents, dim=0)
            means.append((edge_count, ans_diff_percents.mean().item()))
            standard_devs.append((edge_count, ans_diff_percents.std().item()))
            points.append((edge_count, ans_diff_percents))
    return means, standard_devs, points
