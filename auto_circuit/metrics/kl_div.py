from typing import Dict, List, Tuple

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch


def measure_kl_div(
    model: t.nn.Module,
    test_loader: DataLoader[PromptPairBatch],
    pruned_outs: Dict[int, List[t.Tensor]],
) -> Tuple[Dict[int, float], ...]:
    # ) -> Dict[int, float]:
    kl_divs_clean, kl_divs_corrupt = {}, {}
    out_slice = model.out_slice
    # Measure KL divergence
    with t.inference_mode():
        clean_outs = t.cat([model(batch.clean)[out_slice] for batch in test_loader])
        corrupt_outs = t.cat([model(batch.corrupt)[out_slice] for batch in test_loader])
    clean_logprobs = t.nn.functional.log_softmax(clean_outs, dim=-1)
    corrupt_logprobs = t.nn.functional.log_softmax(corrupt_outs, dim=-1)

    for edge_count, pruned_out in pruned_outs.items():
        pruned_out = t.cat(pruned_out)
        pruned_logprobs = t.nn.functional.log_softmax(pruned_out, dim=-1)
        kl_clean = t.nn.functional.kl_div(
            pruned_logprobs,
            clean_logprobs,
            reduction="batchmean",
            log_target=True,
        )
        kl_corrupt = t.nn.functional.kl_div(
            pruned_logprobs,
            corrupt_logprobs,
            reduction="batchmean",
            log_target=True,
        )
        # Numerical errors can cause tiny negative values in KL divergence
        kl_divs_clean[edge_count] = max(kl_clean.mean().item(), 0)
        kl_divs_corrupt[edge_count] = max(kl_corrupt.mean().item(), 0)
    return kl_divs_clean, kl_divs_corrupt
    # return kl_divs_clean, None
