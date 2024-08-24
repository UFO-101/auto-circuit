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

SP_FAITHFULNESS_TARGET = Literal[
    "kl_div", "mse", "answer", "wrong_answer", "correct_percent", "logit_diff_percent"
]


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
    faithfulness_target: SP_FAITHFULNESS_TARGET = "kl_div",
    validation_dataloader: Optional[PromptDataLoader] = None,
) -> PruneScores:
    """
    Optimize the edge mask values using gradient descent to maximize the faithfulness of
    and minimize the number of edges in the circuit. This is based loosely on Subnetwork
    Probing [(Cao et al., 2021)](https://arxiv.org/abs/2104.03514).

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for training input and ablation.
        official_edges: Not used.
        learning_rate: The learning rate for the optimizer.
        epochs: The number of epochs to train for.
        regularize_lambda: The weight of the regularization term, that tries to minimize
            the number of edges in the circuit.
        mask_fn: The function to use to transform the mask values before they are used
            to interpolate edges between the clean and ablated activations. Note that
            `"hard_concrete"` is generally recommended and is often critical for strong
            performance. See [`MaskFn`][auto_circuit.types.MaskFn] for more details.
        dropout_p: The dropout probability of the masks to use during training.
        init_val: The initial value of the mask values. This can be sensitive when using
            the `"hard_concrete"` mask function. The default value is the value used by
            Cao et al.
        show_train_graph: Whether to show a graph of the training loss.
        circuit_size: The size of the circuit to aim for. When this is not `None`, the
            regularization term equals `ReLU(n_mask - circuit_size)` (sign is corrected
            for the value of `tree_optimisation`).
        tree_optimisation: If `True`, the input to the model is the clean input, and the
            mask values are optimized to ablate as many edges as possible. If `False`,
            the corrupt input is used, and the mask values are optimized to Resample
            Ablate (with the clean activations) as few edges as possible.
        avoid_edges: A set of edges to avoid. An extra penalty is added to the loss for
            each edge in this set that is included in the circuit.
        avoid_lambda: The weight of the penalty for `avoid_edges`.
        faithfulness_target: The faithfulness metric to optimize the circuit for.
        validation_dataloader: If not `None` the faithfulness metric is also computed on
            this dataloader and plotted in the training graph (if `show_train_graph` is
            `True`).

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.
    """
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
        optim = t.optim.adam.Adam(mask_params, lr=learning_rate)
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
                        logit_diff_term = t.abs(100 - logit_diffs).mean()
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
                        val_faiths.append(val_faithful_term.mean().item())

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
