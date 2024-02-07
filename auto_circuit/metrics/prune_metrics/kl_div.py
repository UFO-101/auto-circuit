from typing import Optional

import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import Measurements, PrunedOutputs, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm


def measure_kl_div(
    task: Task,
    prune_scores: Optional[PruneScores],
    pruned_outs: Optional[PrunedOutputs],
    compare_to_clean: bool = True,
) -> Measurements:
    """Measure KL divergence between the default model and the pruned model."""
    assert pruned_outs is not None
    kl_divs = []
    out_slice = task.model.out_slice
    with t.inference_mode():
        default_outs = []
        for batch in task.test_loader:
            default_batch = batch.clean if compare_to_clean else batch.corrupt
            default_outs.append(task.model(default_batch)[out_slice])
    default_logprobs = t.nn.functional.log_softmax(t.cat(default_outs), dim=-1)

    for edge_count, pruned_out in (pruned_out_pbar := tqdm(pruned_outs.items())):
        pruned_out_pbar.set_description_str(f"KL Div for {edge_count} edges")
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
