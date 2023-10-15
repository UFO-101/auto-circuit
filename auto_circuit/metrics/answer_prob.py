from typing import Dict, List

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch


def measure_answer_prob(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
    logprobs: bool = False,
) -> Dict[int, float]:
    probs = {}
    softmax = t.nn.functional.log_softmax if logprobs else t.nn.functional.softmax

    for edge_count, pruned_out in pruned_outs.items():
        pruned_out = t.cat(pruned_out)
        pruned_probs = softmax(pruned_out, dim=-1)
        answers = t.cat([batch.answers for batch in test_loader])
        answer_probs = t.gather(pruned_probs, dim=1, index=answers).squeeze(-1)
        average_answer_prob = answer_probs.mean().item()
        probs[edge_count] = average_answer_prob
    return probs
