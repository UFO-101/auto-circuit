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
