import torch as t
from typing import Dict, List, Tuple
from auto_circuit.data import BatchKey
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT, Task
import plotly.graph_objects as go
from plotly import subplots

from auto_circuit.types import AlgoKey, PruneScores, TaskKey, TaskPruneScores
from auto_circuit.utils.graph_utils import get_sorted_src_outs, mask_fn_mode, patch_mode
from auto_circuit.utils.tensor_ops import multibatch_kl_div, prune_scores_threshold
from torch.nn.functional import log_softmax
from auto_circuit.utils.custom_tqdm import tqdm

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


def test_completeness(
    task_prune_scores: TaskPruneScores,
    knockout_prune_scores: TaskPruneScores,
    algo_keys: List[AlgoKey],
) -> TaskCompletenessScores:
    task_completeness_scores: TaskCompletenessScores = {}
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        assert task.true_edge_count is not None
        true_circuit_size: int = task.true_edge_count
        task_pbar.set_description_str(f"Task: {task.name}")
        algo_completeness_scores: AlgoCompletenessScores = {}
        for algo_key, algo_ps in (algo_pbar := tqdm(algo_prune_scores.items())):
            if algo_key not in algo_keys:
                print("skipping algo", algo_key)
                continue
            algo = PRUNE_ALGO_DICT[algo_key]
            algo_pbar.set_description_str(f"Algo: {algo.name}")
            same_under_knockouts: CompletenessScores = test_same_under_knockout(
                task=task,
                algo_ps=algo_ps,
                completeness_ps=knockout_prune_scores[task_key][algo_key],
                circuit_size=true_circuit_size,
            )
            algo_completeness_scores[algo_key] = same_under_knockouts
        task_completeness_scores[task_key] = algo_completeness_scores
    return task_completeness_scores


def test_same_under_knockout(
    task: Task,
    algo_ps: PruneScores,
    completeness_ps: PruneScores,
    circuit_size: int,
    hard_concrete_threshold: float = 0.0,
) -> CompletenessScores:
    model = task.model
    circuit_threshold = prune_scores_threshold(algo_ps, circuit_size)

    corrupt_src_outs_dict: Dict[BatchKey, t.Tensor] = {}
    for batch in task.test_loader:
        patch_outs = get_sorted_src_outs(model, batch.corrupt)
        corrupt_src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    completeness_masks = list(completeness_ps.values())
    # Make a boolean copy of the patch_masks that encodes the circuit
    circ_masks = [algo_ps[m] >= circuit_threshold for m in completeness_ps.keys()]
    actual_circuit_size = sum([mask.sum().item() for mask in circ_masks])
    print("actual_circuit_size", actual_circuit_size, "circuit_size", circuit_size)
    # assert actual_circuit_size == circuit_size

    # Test the circuit with the knockouts
    knockout_circuit_logprobs, knockout_model_logprobs = [], []
    circuit_logprobs, model_logprobs = [], []
    with t.no_grad(), mask_fn_mode(model, mask_fn=None):
        # Discretize the circuit with knockouts
        for circ, patch in zip(circ_masks, completeness_masks):
            # Patch edges where learned mask is greater than hard_concrete_threshold
            patch.data = (patch.data >= hard_concrete_threshold).float()
            # Also patch edges not in the circuit
            t.where(circ, patch.data, t.ones_like(patch.data), out=patch.data)
        # Test the circuit with the knockouts
        for batch in task.test_loader:
            patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
            with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                model_out = model(batch.clean)[model.out_slice]
                knockout_circuit_logprobs.append(log_softmax(model_out, dim=-1))

        # Test the full model with the same knockouts
        for circ, patch in zip(circ_masks, completeness_masks):
            # Don't patch edges not in the circuit (but keep patches in the circuit)
            t.where(circ, patch.data, t.zeros_like(patch.data), out=patch.data)
        knockouts_size = int(sum([mask.sum().item() for mask in completeness_masks]))
        for batch in task.test_loader:
            patch_outs = corrupt_src_outs_dict[batch.key].clone().detach()
            with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                model_out = model(batch.clean)[model.out_slice]
                knockout_model_logprobs.append(log_softmax(model_out, dim=-1))

        # Test the circuit without knockouts (with tree patching)
        for circ, patch in zip(circ_masks, completeness_masks):
            # Patch every edge not in the circuit
            patch.data = (~circ.bool()).float()
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