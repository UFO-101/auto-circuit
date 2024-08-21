from typing import Dict, Literal, Optional, Set, Union

import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType, BatchKey, Edge, PruneScores
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    patch_mode,
    set_all_masks,
    set_masks_at_src_idxs,
    train_mask_mode,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val


def mask_gradient_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val", "mse"],
    mask_val: Optional[float] = None,
    integrated_grad_samples: Optional[int] = None,
    ablation_type: AblationType = AblationType.RESAMPLE,
    clean_corrupt: Optional[Literal["clean", "corrupt"]] = "corrupt",
    layers: Optional[Union[int, list[int]]] = None
) -> PruneScores:
    """
    Prune scores equal to the gradient of the mask values that interpolates the edges
    between the clean activations and the ablated activations.

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for input.
        official_edges: Not used.
        grad_function: Function to apply to the logits before taking the gradient.
        answer_function: Loss function of the model output which the gradient is taken
            with respect to.
        mask_val: Value of the mask to use for the forward pass. Cannot be used if
            `integrated_grad_samples` is not `None`.
        integrated_grad_samples: If not `None`, we compute an approximation of the
            Integrated Gradients
            [(Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365) of the model
            output with respect to the mask values. This is computed by averaging the
            mask gradients over `integrated_grad_samples` samples of the mask values
            interpolated between 0 and 1. Cannot be used if `mask_val` is not `None`.
        ablation_type: The type of ablation to perform.
        clean_corrupt: Whether to use the clean or corrupt inputs to calculate the
            ablations.
        layers: If not `None`, we iterate over each layer in the model and compute 
            scores separately for each. Only used if `ig_samples` is not `None`. Follows
            [Marks et al., 2024](https://arxiv.org/abs/2403.19647)

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.

    Note:
        When `grad_function="logit"` and `mask_val=0` this function is exactly
        equivalent to
        [`edge_attribution_patching_prune_scores`][auto_circuit.prune_algos.edge_attribution_patching.edge_attribution_patching_prune_scores].
    """
    assert (mask_val is not None) ^ (integrated_grad_samples is not None)  # ^ means XOR
    assert (layers is None) or (integrated_grad_samples is not None)
    model = model
    out_slice = model.out_slice

    src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        dataloader,
        ablation_type=ablation_type,
        clean_corrupt=clean_corrupt,
    )
    prune_scores = model.new_prune_scores()
    with train_mask_mode(model):
        layers_iter = range((layers or 0)+1) if isinstance(layers, int) else layers
        for layer in (layer_bar := tqdm(layers_iter)):
            layer_bar.set_description_str(f"Layer: {layer}")
            src_idxs = [src.src_idx for src in model.srcs if src.layer == layer]
            max_src_idx = max(src_idxs) if layers else 0
            score_slice = src_idxs if layers else slice(None)
            for sample in (ig_pbar := tqdm(range((integrated_grad_samples or 0)+1))):
                ig_pbar.set_description_str(f"Sample: {sample}")
                # Interpolate the mask value if integrating gradients. Else set the value.
                if integrated_grad_samples is not None:
                    val = sample / integrated_grad_samples
                else: 
                    val = mask_val
                # Set the mask value at layer if layer. Else set the value for all layers.
                if layers is not None:
                    set_all_masks(model, val=0)
                    set_masks_at_src_idxs(model, val=val, src_idxs=src_idxs)
                else: 
                    set_all_masks(model, val=val)
                for batch in dataloader:
                    patch_src_outs = src_outs[batch.key].clone().detach()
                    with patch_mode(model, patch_src_outs):
                        logits = model(batch.clean)[out_slice]
                        if grad_function == "logit":
                            token_vals = logits
                        elif grad_function == "prob":
                            token_vals = t.softmax(logits, dim=-1)
                        elif grad_function == "logprob":
                            token_vals = log_softmax(logits, dim=-1)
                        elif grad_function == "logit_exp":
                            numerator = t.exp(logits)
                            denominator = numerator.sum(dim=-1, keepdim=True)
                            token_vals = numerator / denominator.detach()
                        else:
                            raise ValueError(f"Unknown grad_function: {grad_function}")

                        if answer_function == "avg_diff":
                            loss = -batch_avg_answer_diff(token_vals, batch)
                        elif answer_function == "avg_val":
                            loss = -batch_avg_answer_val(token_vals, batch)
                        elif answer_function == "mse":
                            loss = t.nn.functional.mse_loss(token_vals, batch.answers)
                        else:
                            raise ValueError(f"Unknown answer_function: {answer_function}")
            # set scores from layer (or all scores if layers is None)
            for dest_wrapper in model.dest_wrappers: 
                if dest_wrapper.in_srcs.stop >= max_src_idx:
                    grad = dest_wrapper.patch_mask.grad
                    assert grad is not None
                    scores = grad.detach().clone()[..., score_slice]
                    prune_scores[dest_wrapper.module_name][..., score_slice] = scores
    return prune_scores
