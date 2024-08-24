import math
from typing import Dict, List, Literal

import plotly.graph_objects as go
import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT, PruneAlgo
from auto_circuit.tasks import TASK_DICT, Task
from auto_circuit.types import (
    AblationType,
    AlgoKey,
    AlgoPruneScores,
    BatchKey,
    MaskFn,
    PruneScores,
    TaskPruneScores,
)
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    mask_fn_mode,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.tensor_ops import (
    batch_avg_answer_diff,
    multibatch_kl_div,
    prune_scores_threshold,
    sample_hard_concrete,
)


def train_same_under_knockouts(
    task_prune_scores: TaskPruneScores,
    algo_keys: List[AlgoKey],
    learning_rate: float,
    epochs: int,
    regularize_lambda: float,
    faithfulness_target: Literal["kl_div", "logit_diff"] = "kl_div",
) -> TaskPruneScores:
    """
    Wrapper of
    [`train_same_under_knockout_prune_scores`][auto_circuit.metrics.completeness_metrics.train_same_under_knockouts.train_same_under_knockout_prune_scores]
    for multiple tasks and algorithms.
    """
    task_completeness_scores: TaskPruneScores = {}
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        # if task_key != IOI_TOKEN_CIRCUIT_TASK.key:
        #     continue
        assert task.true_edge_count is not None
        true_circuit_size: int = task.true_edge_count
        task_pbar.set_description_str(f"Task: {task.name}")
        algo_completeness_scores: AlgoPruneScores = {}
        for algo_key, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            # if algo_key not in algo_keys:
            #     print("skipping algo", algo_key)
            #     continue
            algo = PRUNE_ALGO_DICT[algo_key]
            algo_pbar.set_description_str(f"Algo: {algo.name}")

            same_under_knockouts: PruneScores = train_same_under_knockout_prune_scores(
                task=task,
                algo=algo,
                algo_ps=prune_scores,
                circuit_size=true_circuit_size,
                learning_rate=learning_rate,
                epochs=epochs,
                regularize_lambda=regularize_lambda,
                faithfulness_target=faithfulness_target,
            )
            algo_completeness_scores[algo_key] = same_under_knockouts
        task_completeness_scores[task_key] = algo_completeness_scores
    return task_completeness_scores


mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))


def train_same_under_knockout_prune_scores(
    task: Task,
    algo: PruneAlgo,
    algo_ps: PruneScores,
    circuit_size: int,
    learning_rate: float,
    epochs: int,
    regularize_lambda: float,
    mask_fn: MaskFn = "hard_concrete",
    faithfulness_target: Literal["kl_div", "logit_diff"] = "kl_div",
) -> PruneScores:
    """
    Learn a subset of the circuit to ablate such that when the same edges are ablated
    from the full model, the KL divergence between the circuit and the full model is
    maximized.

    See:
    [`same_under_knockouts`][auto_circuit.metrics.completeness_metrics.same_under_knockouts.same_under_knockout]

    Args:
        task: The task to train on.
        algo: The pruning algorithm used to generate the circuit. This value is only
            used for visualization purposes.
        algo_ps: The pruning scores for the algorithm. The circuit is defined as the
            top `circuit_size` edges according to these scores.
        circuit_size: The size of the circuit to knockout.
        learning_rate: The learning rate for the optimization.
        epochs: The number of epochs to train for.
        regularize_lambda: The regularization strength for the number of edges that are
            knocked out. Can reasonably be set to 0.
        mask_fn: The mask parameterization to use for the optimization. `hard_concrete`
            is highly recommended.
        faithfulness_target: The target for the faithfulness term in the loss. The
            optimizer will try to maximize the difference in this target between the
            knocked-out circuit and the knocked-out full model.

    Returns:
        The learned ordering of edges to knockout.
    """
    circuit_threshold = prune_scores_threshold(algo_ps, circuit_size)
    model = task.model
    n_target = int(circuit_size / 5)

    corrupt_src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        task.test_loader,
        ablation_type=AblationType.RESAMPLE,
        clean_corrupt="corrupt",
    )

    loss_history, faith_history, reg_history = [], [], []
    with train_mask_mode(model) as patch_masks:
        mask_params = list(patch_masks.values())
        set_all_masks(model, val=0.0)

        # Make a boolean copy of the patch_masks that encodes the circuit
        circ_masks = [algo_ps[m].abs() >= circuit_threshold for m in patch_masks.keys()]
        actual_circuit_size = sum([mask.sum().item() for mask in circ_masks])
        print("actual_circuit_size", actual_circuit_size, "circuit_size", circuit_size)
        # assert actual_circuit_size == circuit_size

        set_all_masks(model, val=-init_mask_val)
        optim = t.optim.adam.Adam(mask_params, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            kl_str = faith_history[-1] if len(faith_history) > 0 else None
            epoch_pbar.set_description_str(f"Epoch: {epoch}, KL Div: {kl_str}")
            for batch in task.test_loader:
                patches = corrupt_src_outs[batch.key].clone().detach()
                with patch_mode(model, patches), mask_fn_mode(model, mask_fn):
                    optim.zero_grad()
                    model.zero_grad()

                    # Patch all the edges not in the circuit
                    with t.no_grad():
                        for circ, patch in zip(circ_masks, mask_params):
                            patch_all = t.full_like(patch.data, 99)
                            t.where(circ, patch.data, patch_all, out=patch.data)
                    circ_out = model(batch.clean)[model.out_slice]

                    # Don't patch edges not in the circuit
                    with t.no_grad():
                        for cir, patch in zip(circ_masks, mask_params):
                            patch_none = t.full_like(patch.data, -99)
                            t.where(cir, patch.data, patch_none, out=patch.data)
                    model_out = model(batch.clean)[model.out_slice]

                    if faithfulness_target == "kl_div":
                        circuit_logprobs = log_softmax(circ_out, dim=-1)
                        model_logprobs = log_softmax(model_out, dim=-1)
                        faith = -multibatch_kl_div(circuit_logprobs, model_logprobs)
                    else:
                        assert faithfulness_target == "logit_diff"
                        circ_logit_diff = batch_avg_answer_diff(circ_out, batch)
                        model_logit_diff = batch_avg_answer_diff(model_out, batch)
                        faith = -(model_logit_diff - circ_logit_diff)
                    faith_history.append(faith.item())

                    flat_masks = t.cat([mask.flatten() for mask in mask_params])
                    knockouts_samples = sample_hard_concrete(flat_masks, batch_size=1)
                    reg_term = t.relu(knockouts_samples.sum() - n_target) / n_target
                    reg_history.append(reg_term.item() * regularize_lambda)

                    loss = faith + reg_term * regularize_lambda
                    loss.backward()
                    loss_history.append(loss.item())
                    optim.step()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss_history, name="Loss"))
        fig.add_trace(go.Scatter(y=faith_history, name=faithfulness_target))
        fig.add_trace(go.Scatter(y=reg_history, name="Regularization"))
        fig.update_layout(
            title=f"Same Under Knockouts for Task: {task.name}, Algo: {algo.name}"
        )
        fig.show()

    completeness_prune_scores: PruneScores = {}
    for mod_name, patch_mask in model.patch_masks.items():
        completeness_prune_scores[mod_name] = patch_mask.detach().clone()
    return completeness_prune_scores
