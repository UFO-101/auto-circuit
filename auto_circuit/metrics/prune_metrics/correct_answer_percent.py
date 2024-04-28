import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import (
    correct_answer_greater_than_incorrect_proportion,
    correct_answer_proportion,
)


def measure_correct_ans_percent(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    pruned_outs: CircuitOutputs,
    out_of_correct_and_incorrect_answers: bool = False,
) -> Measurements:
    """
    Percentage of outputs for which the correct answer has the highest logit.

    Args:
        model: Not used.
        dataloader: The dataloader on which the `pruned_outs` were calculated.
        pruned_outs: The outputs of the ablated model for each circuit size.
        out_of_correct_and_incorrect_answers: Whether to calculate the proportion of
            prompts for which the correct answer has a higher logit than the incorrect
            answers (`True`). Otherwise, calculate the proportion of prompts for which
            the correct answer has the highest of all logits (`False`).

            This is useful when you are particularly interested in the counterfactual
            comparison to the corrupt prompts. For example, in the Sports Player post
            Rajamanoharan et al.
            [(2023)](https://www.alignmentforum.org/posts/3tqJ65kuTkBh8wrRH/)
            look at the proportion of prompts for which the correct sport has a
            greater logit than the two other sports.


    Note:
        This function assumes that each prompt in `dataloader` has only one correct
            answer. If not, an error will be raised.
    """
    measurements = []
    for edge_count, pruned_out in (pruned_out_pbar := tqdm(pruned_outs.items())):
        pruned_out_pbar.set_description_str(f"Correct Percent for {edge_count} edges")
        correct_proportions = []
        for batch in dataloader:
            assert isinstance(batch.answers, t.Tensor)
            logits = pruned_out[batch.key]
            if out_of_correct_and_incorrect_answers:
                correct_proportn = correct_answer_greater_than_incorrect_proportion(
                    logits, batch
                )
            else:
                correct_proportn = correct_answer_proportion(logits, batch)
            correct_proportions.append(correct_proportn)
        # PromptDataLoaders have all batches the same size, so we mean the batch means
        correct_proportion = t.stack(correct_proportions).float().mean().item() * 100
        measurements.append((edge_count, correct_proportion))
    return measurements
