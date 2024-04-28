from typing import Dict, List, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots
from torch.nn.functional import log_softmax

from auto_circuit.data import BatchKey
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT, Task
from auto_circuit.types import (
    AblationType,
    AlgoKey,
    PruneScores,
    TaskKey,
    TaskPruneScores,
)
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import patch_mode
from auto_circuit.utils.tensor_ops import multibatch_kl_div, prune_scores_threshold

# circuit_size, n_knockouts, normal_kl, knockout_kl
CompletenessScores = List[Tuple[int, int, float, float]]
"""
A list of tuples containing:
<ol>
    <li>The size of the circuit.</li>
    <li>The number of knockouts.</li>
    <li>The KL divergence between the circuit and the full model.</li>
    <li>The KL divergence between the circuit with knockouts and the full model with
    knockouts.</li>
</ol>
"""

AlgoCompletenessScores = Dict[AlgoKey, CompletenessScores]
"""
[`CompletenessScores`][auto_circuit.metrics.completeness_metrics.same_under_knockouts.CompletenessScores]
for each algorithm.
"""

TaskCompletenessScores = Dict[TaskKey, AlgoCompletenessScores]
"""
[`AlgoCompletenessScores`][auto_circuit.metrics.completeness_metrics.same_under_knockouts.AlgoCompletenessScores]
for each task and algorithm.
"""


def same_under_knockouts_fig(
    task_completeness_scores: TaskCompletenessScores,
) -> go.Figure:
    """
    Create a plotly figure showing the difference in KL divergence between the circuit
    and the full model with and without knockouts for each task and algorithm.

    Args:
        task_completeness_scores: The completeness scores for each task and algorithm.

    Returns:
        The plotly figure.
    """
    n_cols = len(task_completeness_scores)
    titles = [TASK_DICT[task].name for task in task_completeness_scores.keys()]
    fig = subplots.make_subplots(rows=1, cols=n_cols, subplot_titles=titles)
    for col, algo_comp_scores in enumerate(task_completeness_scores.values(), start=1):
        xs = []
        for algo_key, completeness_scores in algo_comp_scores.items():
            for (circ_size, n_knockouts, x, y) in completeness_scores:
                new_xs = [x, (x + y) / 2]
                xs.extend(new_xs)
                scatter = go.Scatter(
                    x=new_xs,
                    y=[y, (x + y) / 2],
                    name=PRUNE_ALGO_DICT[algo_key].name,
                    hovertext=[f"circ size: {circ_size}<br>n_knockouts: {n_knockouts}"],
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


def measure_same_under_knockouts(
    circuit_ps: TaskPruneScores,
    knockout_ps: TaskPruneScores,
) -> TaskCompletenessScores:
    """
    Wrapper of
    [`same_under_knockout`][auto_circuit.metrics.completeness_metrics.same_under_knockouts.same_under_knockout]
    for each task and algorithm in `circuit_ps` and `knockout_ps`.
    """
    task_completeness_scores: TaskCompletenessScores = {}
    for task_key, algos_ko_ps in (task_pbar := tqdm(knockout_ps.items())):
        task = TASK_DICT[task_key]
        assert task.true_edge_count is not None
        true_circuit_size: int = task.true_edge_count
        task_pbar.set_description_str(f"Task: {task.name}")
        algo_completeness_scores: AlgoCompletenessScores = {}
        for algo_key, algo_ko_ps in (algo_pbar := tqdm(algos_ko_ps.items())):
            algo = PRUNE_ALGO_DICT[algo_key]
            algo_pbar.set_description_str(f"Algo: {algo.name}")
            print()
            print("task", task.name, "algo", algo.name)
            same_under_knockouts: CompletenessScores = same_under_knockout(
                task=task,
                circuit_ps=circuit_ps[task_key][algo_key],
                knockout_ps=algo_ko_ps,
                circuit_size=true_circuit_size,
            )
            algo_completeness_scores[algo_key] = same_under_knockouts
        task_completeness_scores[task_key] = algo_completeness_scores
    return task_completeness_scores


def same_under_knockout(
    task: Task,
    circuit_ps: PruneScores,
    knockout_ps: PruneScores,
    circuit_size: int,
    knockout_threshold: float = 0.0,
) -> CompletenessScores:
    """
    Given a circuit and a set of edges to ablate, measure the difference in KL
    divergence between the circuit and the full model with and without the knockouts.

    This is the measure of completeness introduced by [Wang et al.
    (2022)](https://arxiv.org/abs/2211.00593) to test the IOI circuit.

    The optimization process that attempts to find the knockouts that maximize the
    difference is implemented separately in
    [`train_same_under_knockout_prune_scores`][auto_circuit.metrics.completeness_metrics.train_same_under_knockouts.train_same_under_knockout_prune_scores].

    Args:
        task: The task to measure the completeness for.
        circuit_ps: The circuit to test. The top `circuit_size` edges are taken to be
            the circuit.
        knockout_ps: The set of knockouts to test. All edges with scores greater than
            `knockout_threshold` are knocked out.
        circuit_size: The size of the circuit.
        knockout_threshold: The threshold for knockout edges.

    Returns:
        Tuple of completeness information.
    """
    model = task.model
    patch_masks: Dict[str, t.nn.Parameter] = model.patch_masks
    circuit_threshold = prune_scores_threshold(circuit_ps, circuit_size)

    corrupt_src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        task.test_loader,
        ablation_type=AblationType.RESAMPLE,
        clean_corrupt="corrupt",
    )

    mask_params = list(patch_masks.values())
    # Make a boolean copy of the patch_masks that encodes the circuit
    circ_masks = [circuit_ps[m].abs() >= circuit_threshold for m in patch_masks.keys()]
    actual_circuit_size: int = int(sum([mask.sum().item() for mask in circ_masks]))
    knockout_masks = [knockout_ps[m] >= knockout_threshold for m in patch_masks.keys()]
    # assert actual_circuit_size == circuit_size

    # Test the circuit with the knockouts
    ko_circ_logprobs, ko_model_logprobs = {}, {}
    circ_logprobs, model_logprobs = {}, {}
    with t.no_grad():
        # Discretize the circuit with knockouts
        for circ, knockout, patch in zip(circ_masks, knockout_masks, mask_params):
            # Patch edges where learned mask is greater than knockout_threshold
            patch.data = knockout.float()
            # Also patch edges not in the circuit
            t.where(circ, patch.data, t.ones_like(patch.data), out=patch.data)
        # Test the circuit with the knockouts
        for batch in task.test_loader:
            patch_outs = corrupt_src_outs[batch.key].clone().detach()
            with patch_mode(model, patch_outs):
                model_out = model(batch.clean)[model.out_slice]
                ko_circ_logprobs[batch.key] = log_softmax(model_out, dim=-1)

        # Test the full model with the same knockouts
        for circ, patch in zip(circ_masks, mask_params):
            # Don't patch edges not in the circuit (but keep patches in the circuit)
            t.where(circ, patch.data, t.zeros_like(patch.data), out=patch.data)
        knockouts_size = int(sum([mask.sum().item() for mask in mask_params]))
        for batch in task.test_loader:
            patch_outs = corrupt_src_outs[batch.key].clone().detach()
            with patch_mode(model, patch_outs):
                model_out = model(batch.clean)[model.out_slice]
                ko_model_logprobs[batch.key] = log_softmax(model_out, dim=-1)

        # Test the circuit without knockouts (with tree patching)
        for circ, patch in zip(circ_masks, mask_params):
            # Patch every edge not in the circuit
            patch.data = (~circ).float()
        for batch in task.test_loader:
            patch_outs = corrupt_src_outs[batch.key].clone().detach()
            with patch_mode(model, patch_outs):
                model_out = model(batch.clean)[model.out_slice]
                circ_logprobs[batch.key] = log_softmax(model_out, dim=-1)

        # Test the full model without knockouts
        for batch in task.test_loader:
            model_out = model(batch.clean)[model.out_slice]
            model_logprobs[batch.key] = log_softmax(model_out, dim=-1)

    # Sort the logprobs by batch key and stack them
    ko_circ_logprobs_ten = t.stack([o for _, o in sorted(ko_circ_logprobs.items())])
    ko_model_logprobs_ten = t.stack([o for _, o in sorted(ko_model_logprobs.items())])
    knockout_kl = multibatch_kl_div(ko_circ_logprobs_ten, ko_model_logprobs_ten).item()

    circ_logprobs_ten = t.stack([o for _, o in sorted(circ_logprobs.items())])
    model_logprobs_ten = t.stack([o for _, o in sorted(model_logprobs.items())])
    normal_kl = multibatch_kl_div(circ_logprobs_ten, model_logprobs_ten).item()

    return [(actual_circuit_size, knockouts_size, normal_kl, knockout_kl)]
