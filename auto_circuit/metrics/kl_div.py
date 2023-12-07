from typing import Dict, List

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Measurements


def measure_kl_div(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
    compare_to_clean: bool = True,
) -> Measurements:
    kl_divs = []
    out_slice = model.out_slice
    # Measure KL divergence
    with t.inference_mode():
        if compare_to_clean:
            default_outs = t.cat(
                [model(batch.clean)[out_slice] for batch in test_loader]
            )
        else:
            default_outs = t.cat(
                [model(batch.corrupt)[out_slice] for batch in test_loader]
            )
    default_logprobs = t.nn.functional.log_softmax(default_outs, dim=-1)

    for edge_count, pruned_out in pruned_outs.items():
        pruned_out = t.cat(pruned_out)
        pruned_logprobs = t.nn.functional.log_softmax(pruned_out, dim=-1)
        kl = t.nn.functional.kl_div(
            pruned_logprobs,
            default_logprobs,
            reduction="batchmean",
            log_target=True,
        )
        # Numerical errors can cause tiny negative values in KL divergence
        kl_divs.append((edge_count, max(kl.mean().item(), 0)))
    return kl_divs
