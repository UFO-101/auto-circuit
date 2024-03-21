from typing import Dict, List

import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import BatchKey, CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import multibatch_kl_div


def measure_kl_div(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    compare_to_clean: bool = True,
) -> Measurements:
    """Measure KL divergence between the default model and the pruned model."""
    circuit_kl_divs: Measurements = []
    default_logprobs: Dict[BatchKey, t.Tensor] = {}
    with t.inference_mode():
        for batch in dataloader:
            default_batch = batch.clean if compare_to_clean else batch.corrupt
            logits = model(default_batch)[model.out_slice]
            default_logprobs[batch.key] = log_softmax(logits, dim=-1)

    for edge_count, circuit_out in (pruned_out_pbar := tqdm(circuit_outs.items())):
        pruned_out_pbar.set_description_str(f"KL Div for {edge_count} edges")
        circuit_logprob_list: List[t.Tensor] = []
        default_logprob_list: List[t.Tensor] = []
        for batch in dataloader:
            circuit_logprob_list.append(log_softmax(circuit_out[batch.key], dim=-1))
            default_logprob_list.append(default_logprobs[batch.key])
        kl = multibatch_kl_div(t.cat(circuit_logprob_list), t.cat(default_logprob_list))

        # Numerical errors can cause tiny negative values in KL divergence
        circuit_kl_divs.append((edge_count, max(kl.item(), 0)))
    return circuit_kl_divs
