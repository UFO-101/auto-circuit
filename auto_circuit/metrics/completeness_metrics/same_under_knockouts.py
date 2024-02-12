import math
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots
from torch.nn.functional import log_softmax

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT, PruneAlgo
from auto_circuit.tasks import TASK_DICT, Task
from auto_circuit.types import (
    AlgoKey,
    BatchKey,
    MaskFn,
    PruneScores,
    TaskKey,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    mask_fn_mode,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.tensor_ops import (
    multibatch_kl_div,
    prune_scores_threshold,
    sample_hard_concrete,
)

CompletenessScores = Dict[int, List[Tuple[float, float]]]
AlgoCompletenessScores = Dict[AlgoKey, CompletenessScores]
TaskCompletenessScores = Dict[TaskKey, AlgoCompletenessScores]


def same_under_knockouts_fig(
    task_completeness_scores: TaskCompletenessScores,
) -> go.Figure:
    n_cols = len(task_completeness_scores)
    titles = [TASK_DICT[task].name for task in task_completeness_scores.keys()]
    fig = subplots.make_subplots(rows=1, cols=n_cols, subplot_titles=titles)
    for col, algo_comp_scores in enumerate(task_completeness_scores.values(), start=1):
        xs = []
        for algo_key, completeness_scores in algo_comp_scores.items():
            for n_knockouts, kls in completeness_scores.items():
                new_xs = [x for x, _ in kls] + [(x + y) / 2 for x, y in kls]
                xs.extend(new_xs)
                scatter = go.Scatter(
                    x=new_xs,
                    y=[y for _, y in kls] + [(x + y) / 2 for x, y in kls],
                    name=PRUNE_ALGO_DICT[algo_key].name,
                    hovertext=[f"n_knockouts: {n_knockouts}"],
                    mode="lines+markers",
                    showlegend=(col == 1),
                )
                fig.add_trace(scatter, row=1, col=col)
        # Add line y=x without changing the axis limits
        scatter = go.Scatter(
            x=[min(xs), max(xs)],
            y=[min(xs), max(xs)],
            name="y=x",
            mode="lines",
            line=dict(dash="dash"),
            showlegend=(col == 1),
        )
        fig.add_trace(scatter, row=1, col=col)
    fig.update_layout(
        title="Same Under Knockouts",
        xaxis_title="KL Div",
        yaxis_title="Knockout KL Div",
        width=1300,
    )
    return fig


def run_same_under_knockouts(
    task_prune_scores: TaskPruneScores,
    algo_keys: List[AlgoKey],
    learning_rate: float,
    epochs: int,
    regularize_lambda: float,
    hard_concrete_threshold: float = 0.0,
) -> TaskCompletenessScores:
    task_completeness_scores: TaskCompletenessScores = {}
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        # if task_key != "Docstring Token Circuit":
        #     continue
        assert task.true_edge_count is not None
        true_circuit_size: int = task.true_edge_count
        task_pbar.set_description_str(f"Task: {task.name}")
        algo_completeness_scores: AlgoCompletenessScores = {}
        for algo_key, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            # if algo_key not in algo_keys:
            #     print("skipping algo", algo_key)
            #     continue
            algo = PRUNE_ALGO_DICT[algo_key]
            algo_pbar.set_description_str(f"Algo: {algo.name}")

            same_under_knockouts: CompletenessScores = run_same_under_knockout(
                task=task,
                algo=algo,
                prune_scores=prune_scores,
                circuit_size=true_circuit_size,
                learning_rate=learning_rate,
                epochs=epochs,
                regularize_lambda=regularize_lambda,
                hard_concrete_threshold=hard_concrete_threshold,
            )
            algo_completeness_scores[algo_key] = same_under_knockouts
        task_completeness_scores[task_key] = algo_completeness_scores
    return task_completeness_scores


mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))


def run_same_under_knockout(
    task: Task,
    algo: PruneAlgo,
    prune_scores: PruneScores,
    circuit_size: int,
    learning_rate: float,
    epochs: int,
    regularize_lambda: float,
    mask_fn: MaskFn = "hard_concrete",
    hard_concrete_threshold: float = 0.0,
) -> CompletenessScores:
    """
    Learn a subset of the circuit to knockout such that when the same edges are knocked
    out of the full model, the KL divergence between the circuit and the full model is
    maximized.
    """
    circuit_threshold = prune_scores_threshold(prune_scores, circuit_size)
    model = task.model
    n_target = int(circuit_size / 5)

    corrupt_src_outs_dict: Dict[BatchKey, t.Tensor] = {}
    for batch in task.test_loader:
        patch_outs = get_sorted_src_outs(model, batch.corrupt)
        corrupt_src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    loss_history, kl_div_history, reg_history = [], [], []
    with train_mask_mode(model) as patch_masks:
        mask_params = list(patch_masks.values())
        set_all_masks(model, val=0.0)

        # Make a boolean copy of the patch_masks that encodes the circuit
        circ_masks = [prune_scores[m] >= circuit_threshold for m in patch_masks.keys()]
        actual_circuit_size = sum([mask.sum().item() for mask in circ_masks])
        assert actual_circuit_size == circuit_size

        set_all_masks(model, val=-init_mask_val)
        optim = t.optim.Adam(mask_params, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            kl_str = kl_div_history[-1] if len(kl_div_history) > 0 else None
            epoch_pbar.set_description_str(f"Epoch: {epoch}, KL Div: {kl_str}")
            for batch in task.test_loader:
                patches = corrupt_src_outs_dict[batch.key].clone().detach()
                src_outs = t.zeros_like(patches)
                with patch_mode(model, src_outs, patches), mask_fn_mode(model, mask_fn):
                    optim.zero_grad()
                    model.zero_grad()

                    # Patch all the edges not in the circuit
                    with t.no_grad():
                        for circ, patch in zip(circ_masks, mask_params):
                            t.where(circ, patch, t.full_like(patch, 99), out=patch.data)
                    model_out = model(batch.clean)[model.out_slice]
                    circuit_logprobs = log_softmax(model_out, dim=-1)

                    # Don't patch edges not in the circuit
                    with t.no_grad():
                        for cir, patch in zip(circ_masks, mask_params):
                            t.where(cir, patch, t.full_like(patch, -99), out=patch.data)
                    model_out = model(batch.clean)[model.out_slice]
                    model_logprobs = log_softmax(model_out, dim=-1)
                    kl_div_term = -multibatch_kl_div(circuit_logprobs, model_logprobs)
                    kl_div_history.append(kl_div_term.item())

                    flat_masks = t.cat([mask.flatten() for mask in mask_params])
                    knockouts_samples = sample_hard_concrete(flat_masks, batch_size=1)
                    reg_term = t.relu(knockouts_samples.sum() - n_target) / n_target
                    reg_history.append(reg_term.item() * regularize_lambda)

                    loss = kl_div_term + reg_term * regularize_lambda
                    loss.backward()
                    loss_history.append(loss.item())
                    optim.step()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss_history, name="Loss"))
        fig.add_trace(go.Scatter(y=kl_div_history, name="KL Divergence"))
        fig.add_trace(go.Scatter(y=reg_history, name="Regularization"))
        fig.update_layout(
            title=f"Same Under Knockouts for Task: {task.name}, Algo: {algo.name}"
        )
        fig.show()

        # Test the circuit with the knockouts
        knockout_circuit_logprobs, knockout_model_logprobs = [], []
        circuit_logprobs, model_logprobs = [], []
        with t.inference_mode(), mask_fn_mode(model, mask_fn=None):
            # Discretize the circuit with knockouts
            for circ_mask, patch_mask in zip(circ_masks, mask_params):
                # Patch edges where learned mask is greater than hard_concrete_threshold
                patch_mask = (patch_mask >= hard_concrete_threshold).float()
                # Also patch edges not in the circuit
                t.where(circ_mask, patch_mask, t.ones_like(patch_mask), out=patch_mask)
            # Test the circuit with the knockouts
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                    model_out = model(batch.clean)[model.out_slice]
                    knockout_circuit_logprobs.append(log_softmax(model_out, dim=-1))

            # Test the full model with the same knockouts
            for circ_mask, patch_mask in zip(circ_masks, mask_params):
                # Don't patch edges not in the circuit (but keep patches in the circuit)
                t.where(circ_mask, patch_mask, t.zeros_like(patch_mask), out=patch_mask)
            knockouts_size = int(sum([mask.sum().item() for mask in mask_params]))
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                    model_out = model(batch.clean)[model.out_slice]
                    knockout_model_logprobs.append(log_softmax(model_out, dim=-1))

            # Test the circuit without knockouts (with tree patching)
            for circuit_mask, patch_mask in zip(circ_masks, mask_params):
                # Patch every edge not in the circuit
                patch_mask = (~circuit_mask.bool()).float()
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                    model_out = model(batch.clean)[model.out_slice]
                    circuit_logprobs.append(log_softmax(model_out, dim=-1))

            # Test the full model without knockouts
            for batch in task.test_loader:
                model_out = model(batch.clean)[model.out_slice]
                model_logprobs.append(log_softmax(model_out, dim=-1))

    knockout_kl = multibatch_kl_div(
        t.cat(knockout_circuit_logprobs), t.cat(knockout_model_logprobs)
    ).item()
    normal_kl = multibatch_kl_div(t.cat(circuit_logprobs), t.cat(model_logprobs)).item()
    print("model n_edges", model.n_edges, "knockouts_size", knockouts_size)
    print("circuit_size", circuit_size, "diff", circuit_size - knockouts_size)
    print("knockout_kl", knockout_kl)
    print("normal_kl", normal_kl)

    return {knockouts_size: [(normal_kl, knockout_kl)]}
