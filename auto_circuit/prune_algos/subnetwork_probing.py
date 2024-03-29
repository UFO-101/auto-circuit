# Some of this code is copied from:
# https://github.com/stevenxcao/subnetwork-probing
# From the paper Low-Complexity Probing via Finding Subnetworks:
# https://arxiv.org/abs/2104.03514

import math
from typing import Dict, Literal, Optional, Set

import plotly.graph_objects as go
import torch as t
from torch.nn.functional import log_softmax, mse_loss

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType, BatchKey, Edge, MaskFn, PruneScores
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    mask_fn_mode,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.patch_wrapper import sample_hard_concrete
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import (
    batch_answer_diff_percents,
    batch_avg_answer_val,
    correct_answer_proportion,
    multibatch_kl_div,
)

# Constants are copied from the paper's code
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))
regularize_const = temp * math.log(-left / right)

SP = "Subnetwork Probing"


def subnetwork_probing_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    learning_rate: float = 0.1,
    epochs: int = 20,
    regularize_lambda: float = 10,
    mask_fn: MaskFn = "hard_concrete",
    dropout_p: float = 0.0,
    init_val: float = init_mask_val,
    show_train_graph: bool = False,
    circuit_size: Optional[int] = None,
    tree_optimisation: bool = False,
    avoid_edges: Optional[Set[Edge]] = None,
    avoid_lambda: float = 1.0,
    faithfulness_target: Literal[
        "kl_div",
        "mse",
        "answer",
        "wrong_answer",
        "correct_percent",
        "logit_diff_percent",
    ] = "kl_div",
    validation_dataloader: Optional[PromptDataLoader] = None,
) -> PruneScores:
    """Optimize the patch masks using gradient descent."""
    assert len(dataloader) > 0, "Dataloader is empty"

    out_slice = model.out_slice
    n_edges = model.n_edges
    n_avoid = len(avoid_edges or [])

    clean_logits: Dict[BatchKey, t.Tensor] = {}
    with t.inference_mode():
        for batch in dataloader:
            clean_logits[batch.key] = model(batch.clean)[out_slice]

    val_clean_logits: Optional[Dict[BatchKey, t.Tensor]] = None
    if validation_dataloader is not None:
        val_clean_logits = {}
        with t.inference_mode():
            for batch in validation_dataloader:
                val_clean_out = model(batch.clean)[out_slice]
                val_clean_logits[batch.key] = log_softmax(val_clean_out, dim=-1)

    src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        dataloader,
        # ablation_type=AblationType.RESAMPLE,
        ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
        # clean_corrupt="corrupt" if tree_optimisation else "clean",
    )

    val_src_outs: Optional[Dict[BatchKey, t.Tensor]] = None
    if validation_dataloader is not None:
        val_src_outs = batch_src_ablations(
            model,
            validation_dataloader,
            # ablation_type=AblationType.RESAMPLE,
            ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
            # clean_corrupt="corrupt" if tree_optimisation else "clean",
        )

    losses, faiths, val_faiths, val_stds, regularizes = [], [], [], [], []
    set_all_masks(model, val=init_val if tree_optimisation else -init_val)
    with train_mask_mode(model) as patch_masks, mask_fn_mode(model, mask_fn, dropout_p):
        mask_params = patch_masks.values()
        optim = t.optim.Adam(mask_params, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            faith_str = f"{faithfulness_target}: {faiths[-1]:.3f}" if epoch > 0 else ""
            desc = f"Loss: {losses[-1]:.3f}, {faith_str}" if epoch > 0 else ""
            epoch_pbar.set_description_str(f"{SP} Epoch {epoch} " + desc, refresh=False)
            for batch_idx, batch in enumerate(dataloader):
                input_batch = batch.clean if tree_optimisation else batch.corrupt
                patch_outs = src_outs[batch.key].clone().detach()
                with patch_mode(model, patch_outs):
                    train_logits = model(input_batch)[out_slice]
                    if faithfulness_target == "kl_div":
                        faithful_term = multibatch_kl_div(
                            log_softmax(train_logits, dim=-1),
                            log_softmax(clean_logits[batch.key], dim=-1),
                        )
                    elif faithfulness_target == "mse":
                        faithful_term = mse_loss(train_logits, batch.answers)
                    elif faithfulness_target == "correct_percent":
                        faithful_term = correct_answer_proportion(train_logits, batch)
                    elif faithfulness_target == "logit_diff_percent":
                        logit_diffs = batch_answer_diff_percents(
                            train_logits, clean_logits[batch.key], batch
                        )
                        logit_diff_term = t.abs(80 - logit_diffs).mean()
                        # logit_diff_std = t.std(logit_diffs)
                        # faithful_term = logit_diff_std + logit_diff_term
                        faithful_term = logit_diff_term
                    else:
                        assert faithfulness_target in ["answer", "wrong_answer"]
                        wrong = faithfulness_target == "wrong_answer"
                        faithful_term = -batch_avg_answer_val(
                            train_logits, batch, wrong
                        )
                    masks = t.cat([patch_mask.flatten() for patch_mask in mask_params])
                    if mask_fn == "hard_concrete":
                        masks = sample_hard_concrete(masks, batch_size=1)
                    elif mask_fn == "sigmoid":
                        masks = t.sigmoid(masks)
                    n_mask = n_edges - masks.sum() if tree_optimisation else masks.sum()
                    if circuit_size:
                        n_mask = t.relu(n_mask - circuit_size)
                    regularize = n_mask / (circuit_size if circuit_size else n_edges)
                    for edge in avoid_edges or []:  # Penalize banned edges
                        wgt = (-1 if tree_optimisation else 1) * avoid_lambda / n_avoid
                        penalty = edge.patch_mask(model)[edge.patch_idx]
                        const = regularize_const if mask_fn == "hard_concrete" else 0.0
                        if mask_fn is not None:
                            penalty = t.sigmoid(penalty - const)
                        regularize += wgt * penalty
                    loss = faithful_term + regularize * regularize_lambda
                    losses.append(loss.item())
                    faiths.append(faithful_term.item())
                    regularizes.append(regularize.item() * regularize_lambda)
                    model.zero_grad()
                    loss.backward()
                    optim.step()

                if validation_dataloader is not None:
                    assert val_src_outs is not None and val_clean_logits is not None
                    val_batch = next(iter(validation_dataloader))
                    for validation_idx, validation_batch in enumerate(
                        validation_dataloader
                    ):
                        if validation_idx == batch_idx:
                            val_batch = validation_batch
                    val_patch_outs = val_src_outs[val_batch.key].clone().detach()
                    with patch_mode(model, val_patch_outs), t.no_grad():
                        val_input_batch = (
                            val_batch.clean if tree_optimisation else val_batch.corrupt
                        )
                        val_logits = model(val_input_batch)[out_slice]
                        val_faithful_term = batch_answer_diff_percents(
                            log_softmax(val_logits, dim=-1),
                            val_clean_logits[val_batch.key],
                            val_batch,
                        )
                        val_stds.append(t.std(val_faithful_term).item())
                        val_faiths.append(t.abs(100 - val_faithful_term).mean().item())

        xtreme_f = max if tree_optimisation else min
        xtreme_torch_f = t.max if tree_optimisation else t.min
        xtreme_val = abs(xtreme_f([xtreme_torch_f(msk).item() for msk in mask_params]))

    if show_train_graph:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, name="Loss"))
        fig.add_trace(go.Scatter(y=faiths, name=faithfulness_target.title()))
        fig.add_trace(
            go.Scatter(
                y=val_faiths,
                error_y=dict(type="data", array=val_stds),
                name=f"Val {faithfulness_target.title()}",
            )
        )
        fig.add_trace(go.Scatter(y=regularizes, name="Regularization"))
        fig.update_layout(title="Subnetwork Probing", xaxis_title="Step")
        fig.show()

    sign = -1 if tree_optimisation else 1
    prune_scores: PruneScores = {}
    for mod_name, patch_mask in model.patch_masks.items():
        prune_scores[mod_name] = xtreme_val + sign * patch_mask.detach().clone()
    return prune_scores
