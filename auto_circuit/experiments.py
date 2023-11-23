
#%%
import math
from typing import Any, Callable, Dict, List, Set, Tuple
import torch as t
import transformer_lens as tl
import auto_circuit.data
from functools import partial
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.activation_magnitude import activation_magnitude_prune_scores
from auto_circuit.prune_functions.integrated_edge_gradients import integrated_edge_gradients_prune_scores
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.prune_functions.simple_gradient import simple_gradient_prune_scores
from auto_circuit.types import AlgoPruneScores, Edge, EdgeCounts, Experiment, ExperimentPruneScores, PatchType
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model
from auto_circuit.utils.misc import relative_path_to_abs_path
from auto_circuit.visualize import kl_vs_edges_plot, roc_plot
from utils.custom_tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from auto_circuit.metrics.official_circuits.ioi_official import ioi_true_edges
from auto_circuit.metrics.official_circuits.docstring_official import docstring_true_edges
from auto_circuit.metrics.official_circuits.greaterthan_official import greaterthan_true_edges
from auto_circuit.metrics.ROC import measure_roc
from numbers import Number


def create_transformer_lens_experiment(name: str, model_name: str, dataset: str, true_edge_func: Callable[..., Set[Edge]], true_edges_attn_only: bool = False) -> Experiment:
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    assert model.tokenizer is not None
    model.eval()
    prepare_model(model, factorized=True, slice_output=True, seq_len=None, device=device)

    train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
        model.tokenizer,
        relative_path_to_abs_path(f"datasets/{dataset}.json"),
        device=device,
        prepend_bos=True,
        batch_size=32,
        train_test_split=[0.5, 0.5],
        length_limit=64,
        pad=True,
    )

    return Experiment(name, model, train_loader, test_loader, true_edge_func, true_edges_attn_only)


TRANSFORMER_EXPERIMENTS: List[Experiment] = [
    create_transformer_lens_experiment(
        name="Indirect Object Identification",
        model_name="gpt2-small",
        dataset="indirect_object_identification",
        true_edge_func=ioi_true_edges,
        true_edges_attn_only=True,
    ),
    create_transformer_lens_experiment(
        name="Greaterthan",
        model_name="gpt2-small",
        dataset="greater_than_gpt2-small_prompts",
        true_edge_func=greaterthan_true_edges,
    ),
    create_transformer_lens_experiment(
        name="Docstring",
        model_name="attn-only-4l",
        dataset="docstring_prompts",
        true_edge_func=docstring_true_edges,
    )
]

PRUNE_FUNCS: Dict[str, Callable] = {
    # f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
    #     parameter_integrated_grads_prune_scores,
    #     baseline_weights=pig_baseline,
    #     samples=pig_samples,
    # ),
    # "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
    # "ACDC": partial(
    #     acdc_prune_scores,
    #     # tao_exps=list(range(-6, 1)),
    #     tao_exps=[-5],
    #     tao_bases=[1],
    # ),
    # "Integrated edge gradients": partial(
    #     integrated_edge_gradients_prune_scores,
    #     samples=50,
    # ),
    "Prob Gradient": partial(simple_gradient_prune_scores, grad_function="prob"),
    # "Exp Logit Gradient": partial(simple_gradient_prune_scores, grad_function="logit_exp"),
    # "Logit Gradient": partial(simple_gradient_prune_scores, grad_function="logit"),
    # "Logprob Gradient": partial(simple_gradient_prune_scores, grad_function="logprob"),
    # "Subnetwork Probing": partial(
    #     subnetwork_probing_prune_scores,
    #     learning_rate=0.1,
    #     epochs=500,
    #     regularize_lambda=10,
    #     mask_fn=None,  # "hard_concrete",
    #     dropout_p=0.0,
    #     init_val=1.0,
    #     show_train_graph=True,
    # ),
}


def run_prune_funcs(experiments: List[Experiment]) -> ExperimentPruneScores:
    experiment_prune_scores: ExperimentPruneScores = {}
    for experiment in (experiment_pbar := tqdm(experiments)):
        experiment_pbar.set_description_str(f"Task: {experiment.name}")
        prune_scores_dict: AlgoPruneScores = {}
        for name, prune_func in (prune_score_pbar := tqdm(PRUNE_FUNCS.items())):
            prune_score_pbar.set_description_str(f"Prune scores: {name}")
            new_prune_scores = prune_func(experiment.model, experiment.train_loader)
            if name in prune_scores_dict:
                prune_scores_dict[name].update(new_prune_scores)
            else:
                prune_scores_dict[name] = new_prune_scores
        experiment_prune_scores[experiment] = prune_scores_dict
    return experiment_prune_scores
    
def create_combined_kl_plots(all_prune_scores: ExperimentPruneScores) -> go.Figure:
    experiment_kl_divs: Dict[Experiment, Dict[str, Dict[int, float]]] = {}
    for experiment, algo_prune_scores in (experiment_pbar := tqdm(all_prune_scores.items())):
        experiment_pbar.set_description_str(f"Task: {experiment.name}")
        model = experiment.model
        edges: Set[Edge] = model.edges  # type: ignore
        test_loader = experiment.test_loader
        algo_measurements: Dict[str, Any] = {}
        for prune_func_str, prune_scores in (prune_func_pbar := tqdm(algo_prune_scores.items())):
            prune_func_pbar.set_description_str(f"Pruning with {prune_func_str}")
            group_edges = prune_func_str.startswith("ACDC") or prune_func_str.startswith("IOI")
            edge_count_type = EdgeCounts.GROUPS if group_edges else EdgeCounts.LOGARITHMIC
            # test_edge_counts = edge_counts_util(edges, edge_count_type, prune_scores)
            test_edge_counts = edge_counts_util(edges, [1, 10, 100, 1000, 10000], prune_scores)

            pruned_outs = run_pruned(model, test_loader, test_edge_counts, prune_scores)
            points: List[Tuple[Number, Number]] = measure_kl_div(model, test_loader, pruned_outs)
            del pruned_outs

            # points = measure_roc(model, prune_scores, experiment.true_edges, experiment.true_edges_attn_only, group_edges)

            algo_measurements[prune_func_str] = points
            t.cuda.empty_cache()
        experiment_kl_divs[experiment] = algo_measurements

    data = []
    y_max = 0.0
    for experiment, algo_measurements in experiment_kl_divs.items():
        for algo, kl_divs in algo_measurements.items():
            for x, y in kl_divs:
                data.append({
                    'Task': experiment.name,
                    'Algorithm': algo,
                    'X': x,
                    'Y': y,
                })
                y_max = max(y_max, y)
    # fig = roc_plot(data)
    fig = kl_vs_edges_plot(data, y_max)
    # fig = px.line(data, x='Edges', y='KL Divergence', facet_col='Task', color='Algorithm', log_x=True, log_y=True, range_y=[1e-6, kl_max * 2])
    # fig.update_layout(
    #     title=f"Task Pruning: {PatchType.PATH_PATCH}",
    #     xaxis_title="Patched Edges",
    #     yaxis_title= "KL Divergence",
    #     template="plotly",
    # )
    # fig.update_xaxes(matches=None)
    return fig

def run_experiments():
    experiment_prune_scores = run_prune_funcs(TRANSFORMER_EXPERIMENTS)
    fig = create_combined_kl_plots(experiment_prune_scores)
    fig.show()

run_experiments()