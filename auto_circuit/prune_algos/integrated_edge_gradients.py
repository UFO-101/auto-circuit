from typing import Dict

import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.misc import batch_avg_answer_diff, batch_avg_answer_val


def integrated_edge_gradients_prune_scores(
    task: Task,
    samples: int = 50,
    answer_diff: bool = False,
) -> PruneScores:
    """Prune scores are the integrated gradient of each edge."""
    model = task.model
    out_slice = model.out_slice

    src_outs_dict: Dict[int, t.Tensor] = {}
    for batch in task.train_loader:
        patch_outs = get_sorted_src_outs(model, batch.clean)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    set_all_masks(model, val=0.0)
    with train_mask_mode(model) as patch_masks:
        for sample in (ig_pbar := tqdm(range(samples))):
            ig_pbar.set_description_str("Integrated Edge Gradients", refresh=False)
            [t.nn.init.constant_(mask, sample / samples) for mask in patch_masks]
            for batch in task.train_loader:
                patch_src_outs = src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_src_outs), patch_src_outs):
                    model_out = model(batch.corrupt)[out_slice]
                    masked_logprobs = log_softmax(model_out, dim=-1)
                    if answer_diff:
                        loss = batch_avg_answer_diff(masked_logprobs, batch)
                    else:
                        loss = batch_avg_answer_val(masked_logprobs, batch)
                    loss.backward()

    prune_scores = {}
    for edge in task.model.edges:
        grad = edge.patch_mask(model).grad
        assert grad is not None
        prune_scores[edge] = grad[edge.patch_idx].item()
    return prune_scores
