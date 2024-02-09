import math
from collections import defaultdict
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots
from torch.nn.functional import kl_div, log_softmax

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT, PruneAlgo
from auto_circuit.tasks import TASK_DICT, Task
from auto_circuit.types import (
    AlgoKey,
    BatchKey,
    Edge,
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
from auto_circuit.utils.tensor_ops import sample_hard_concrete

CompletenessScores = List[Tuple[float, float]]
AlgoCompletenessScores = Dict[AlgoKey, CompletenessScores]
TaskCompletenessScores = Dict[TaskKey, AlgoCompletenessScores]


def same_under_knockouts_fig(
    task_completeness_scores: TaskCompletenessScores,
) -> go.Figure:
    n_cols = len(task_completeness_scores)
    fig = subplots.make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[
            TASK_DICT[task].name for task in task_completeness_scores.keys()
        ],
    )
    for col, (task_key, algo_completeness_scores) in enumerate(
        task_completeness_scores.items(), start=1
    ):
        for algo_key, completeness_scores in algo_completeness_scores.items():
            fig.add_trace(
                go.Scatter(
                    x=[n_mask for n_mask, _ in completeness_scores]
                    + [(x + y) / 2 for x, y in completeness_scores],
                    y=[avg_kl_div for _, avg_kl_div in completeness_scores]
                    + [(x + y) / 2 for x, y in completeness_scores],
                    name=PRUNE_ALGO_DICT[algo_key].name,
                    mode="lines+markers",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
        # Get the min completeness score over all algorithms (nested list comprehension)
        min_x = min(
            [
                min([n_mask for n_mask, _ in completeness_scores])
                for completeness_scores in algo_completeness_scores.values()
            ]
        )
        max_x = max(
            [
                max([(x + y) / 2 for x, y in completeness_scores])
                for completeness_scores in algo_completeness_scores.values()
            ]
        )
        # Add line y=x without changing the axis limits
        fig.add_trace(
            go.Scatter(
                x=[min_x, max_x],
                y=[min_x, max_x],
                name="y=x",
                mode="lines",
                line=dict(dash="dash"),
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        title="Same Under Knockouts",
        # xaxis_title="Number of Circuit Edges Knocked Out",
        # yaxis_title="Avg KL_Div(Circuit w/ Knockouts, Model w/ Knockouts)",
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
            test_prune_scores = prune_scores

            # Some prune_algos optimize for a specific circuit size; extract those edges
            prune_score_dict = defaultdict(list)
            for edge, score in prune_scores.items():
                prune_score_dict[score].append(edge)
            for score, edges in prune_score_dict.items():
                if len(edges) == true_circuit_size:
                    test_prune_scores = dict([(e, score) for e in edges])
                    break

            same_under_knockouts: CompletenessScores = run_same_under_knockout(
                task=task,
                algo=algo,
                prune_scores=test_prune_scores,
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
    hard_concrete_threshold: float = 0.0,
) -> CompletenessScores:
    assert len(prune_scores) >= circuit_size
    sorted_edges: List[Edge] = list(
        sorted(prune_scores.keys(), key=lambda x: abs(prune_scores[x]), reverse=True)
    )
    circuit: List[Edge] = sorted_edges[:circuit_size]
    model = task.model
    n_knockouts = int(circuit_size / 5)

    corrupt_src_outs_dict: Dict[BatchKey, t.Tensor] = {}
    for batch in task.test_loader:
        patch_outs = get_sorted_src_outs(model, batch.corrupt)
        corrupt_src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    # Train the subset of the circuit to be different under knockout
    loss_history, kl_div_history, reg_history = [], [], []
    circuit_masks: List[t.Tensor] = []
    with train_mask_mode(model) as patch_masks:
        set_all_masks(model, val=0.0)
        for edge in circuit:
            edge.patch_mask(model).data[edge.patch_idx] = 1.0
        circuit_masks = [mask.detach().clone().bool() for mask in patch_masks]
        set_all_masks(model, val=-init_mask_val)
        optim = t.optim.Adam(patch_masks, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            kl_str = kl_div_history[-1] if len(kl_div_history) > 0 else None
            epoch_pbar.set_description_str(f"Epoch: {epoch}, KL Div: {kl_str}")
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(
                    model, t.zeros_like(patch_outs), patch_outs
                ), mask_fn_mode(model, mask_fn="hard_concrete"):
                    optim.zero_grad()
                    model.zero_grad()
                    with t.no_grad():
                        for circuit_mask, patch_mask in zip(circuit_masks, patch_masks):
                            t.where(
                                circuit_mask,
                                patch_mask,
                                t.ones_like(patch_mask) * 99,
                                out=patch_mask,
                            )
                    circuit_logprobs = log_softmax(
                        model(batch.clean)[model.out_slice], dim=-1
                    )
                    with t.no_grad():
                        for circuit_mask, patch_mask in zip(circuit_masks, patch_masks):
                            t.where(
                                circuit_mask,
                                patch_mask,
                                t.ones_like(patch_mask) * -99,
                                out=patch_mask,
                            )
                    model_logprobs = log_softmax(
                        model(batch.clean)[model.out_slice], dim=-1
                    )
                    kl_div_term = -kl_div(
                        circuit_logprobs,
                        model_logprobs,
                        reduction="batchmean",
                        log_target=True,
                    )
                    knockouts_samples = sample_hard_concrete(
                        t.cat([patch_mask.flatten() for patch_mask in patch_masks]),
                        batch_size=1,
                    )
                    reg_term = (
                        t.relu(knockouts_samples.sum() - n_knockouts) / n_knockouts
                    ) * regularize_lambda
                    loss = kl_div_term + reg_term
                    loss.backward()
                    kl_div_history.append(kl_div_term.item())
                    reg_history.append(reg_term.item())
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

        (
            knockout_circuit_logprobs,
            knockout_model_logprobs,
            circuit_logprobs,
            model_logprobs,
        ) = ([], [], [], [])
        with t.inference_mode(), mask_fn_mode(model, mask_fn=None):
            # Discretize the circuit with knockouts
            for circuit_mask, patch_mask in zip(circuit_masks, patch_masks):
                t.where(
                    patch_mask >= hard_concrete_threshold,
                    t.ones_like(patch_mask),
                    t.zeros_like(patch_mask),
                    out=patch_mask,
                )
                t.where(
                    circuit_mask, patch_mask, t.ones_like(patch_mask), out=patch_mask
                )
            # Test the circuit with the knockouts
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(
                    model, t.zeros_like(patch_outs), patch_outs
                ), t.inference_mode():
                    knockout_circuit_logprobs.append(
                        log_softmax(model(batch.clean)[model.out_slice], dim=-1)
                    )

            # Test the full model with the same knockouts
            for circuit_mask, patch_mask in zip(circuit_masks, patch_masks):
                t.where(
                    circuit_mask, patch_mask, t.zeros_like(patch_mask), out=patch_mask
                )
            knockouts_size = int(sum([mask.sum().item() for mask in patch_masks]))
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(
                    model, t.zeros_like(patch_outs), patch_outs
                ), t.inference_mode():
                    knockout_model_logprobs.append(
                        log_softmax(model(batch.clean)[model.out_slice], dim=-1)
                    )

            # Test the circuit without knockouts (with tree patching)
            for circuit_mask, patch_mask in zip(circuit_masks, patch_masks):
                t.where(
                    circuit_mask,
                    t.zeros_like(patch_mask),
                    t.ones_like(patch_mask),
                    out=patch_mask,
                )
            for batch in task.test_loader:
                patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
                with patch_mode(
                    model, t.zeros_like(patch_outs), patch_outs
                ), t.inference_mode():
                    circuit_logprobs.append(
                        log_softmax(model(batch.clean)[model.out_slice], dim=-1)
                    )

            # Test the full model without knockouts
            for batch in task.test_loader:
                model_logprobs.append(
                    log_softmax(model(batch.clean)[model.out_slice], dim=-1)
                )

    avg_knockout_kl_div = kl_div(
        t.cat(knockout_circuit_logprobs),
        t.cat(knockout_model_logprobs),
        reduction="batchmean",
        log_target=True,
    ).item()
    avg_normal_kl_div = kl_div(
        t.cat(circuit_logprobs),
        t.cat(model_logprobs),
        reduction="batchmean",
        log_target=True,
    ).item()
    print(
        "model n_edges",
        len(model.edges),
        "knockouts_size",
        knockouts_size,
        "circuit_size",
        circuit_size,
        "diff",
        circuit_size - knockouts_size,
    )
    print(
        "avg_knockout_kl_div",
        avg_knockout_kl_div,
        "avg_normal_kl_div",
        avg_normal_kl_div,
    )

    return [(avg_normal_kl_div, avg_knockout_kl_div)]
