from typing import Any, Literal

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_diff(
    model: PatchableModel,
    test_loader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
) -> Measurements:
    """
    The average difference in the logits (or some function of them) between the correct
    answers and the incorrect answers.

    Args:
        model: Not used.
        test_loader: The dataloader on which the `circuit_outs` were calculated.
        circuit_outs: The outputs of the ablated model for each circuit size.
        prob_func: The function to apply to the logits before calculating the answer
            difference.

    Returns:
        A list of tuples, where the first element is the number of edges pruned and the
            second element is the average answer difference for that number of edges.
    """
    measurements = []
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func == "logits"
        apply_prob_func = identity

    for edge_count, batch_outs in (pruned_out_pbar := tqdm(circuit_outs.items())):
        pruned_out_pbar.set_description_str(f"Answer Diff for {edge_count} edges")
        avg_ans_diff = []
        for batch in test_loader:
            batch_probs = apply_prob_func(batch_outs[batch.key], dim=-1)
            avg_ans_diff.append(batch_avg_answer_diff(batch_probs, batch))
        # PromptDataLoaders have all batches the same size, so we mean the batch means
        measurements.append((edge_count, t.stack(avg_ans_diff).mean().item()))
    return measurements
