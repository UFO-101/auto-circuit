from typing import Dict, Set

import torch as t
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    patch_mode,
    train_mask_mode,
)


def integrated_edge_gradients_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
    output_dim: int = 1,
    samples: int = 50,
) -> Dict[Edge, float]:
    """Prune scores are the integrated gradient of each edge."""
    output_idx = tuple([slice(None)] * output_dim + [-1])
    edges: Set[Edge] = model.edges  # type: ignore

    src_outs_dict: Dict[int, t.Tensor] = {}
    for batch in train_data:
        patch_outs = get_sorted_src_outs(model, batch.clean)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    with train_mask_mode(model, 0.0) as patch_masks:
        for sample in (ig_pbar := tqdm(range(samples))):
            ig_pbar.set_description_str("Integrated Edge Gradients", refresh=False)
            [t.nn.init.constant_(mask, sample / samples) for mask in patch_masks]
            for batch in train_data:
                patch_src_outs = src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_src_outs), patch_src_outs):
                    masked_logprobs = log_softmax(
                        model(batch.corrupt)[output_idx], dim=-1
                    )
                    answer_probs = t.gather(
                        masked_logprobs, dim=1, index=batch.answers
                    ).squeeze(-1)
                    loss = answer_probs.mean()
                    loss.backward()

    prune_scores = {}
    for edge in edges:
        grad = edge.patch_mask(model).grad
        assert grad is not None
        prune_scores[edge] = grad[edge.patch_idx].abs()
    return prune_scores
