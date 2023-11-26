#%%
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Set, Tuple

import plotly.graph_objects as go
import torch as t
import transformer_lens as tl
from utils.custom_tqdm import tqdm

import auto_circuit.data
from auto_circuit.metrics.answer_prob import measure_answer_prob
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.metrics.official_circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.official_circuits.greaterthan_official import (
    greaterthan_true_edges,
)
from auto_circuit.metrics.official_circuits.ioi_official import ioi_true_edges
from auto_circuit.metrics.ROC import measure_roc
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_functions.integrated_edge_gradients import (
    integrated_edge_gradients_prune_scores,
)
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.prune_functions.simple_gradient import simple_gradient_prune_scores
from auto_circuit.prune_functions.subnetwork_probing import (
    subnetwork_probing_prune_scores,
)
from auto_circuit.types import (
    AlgoPruneScores,
    Edge,
    EdgeCounts,
    Experiment,
    ExperimentPruneScores,
    Measurements,
    MetricMeasurements,
)
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import Y_MIN, edge_patching_plot, roc_plot


def create_transformer_lens_experiment(
    name: str,
    model_name: str,
    dataset: str,
    true_edge_func: Callable[..., Set[Edge]],
    true_edges_attn_only: bool = False,
) -> Experiment:
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    assert model.tokenizer is not None
    model.eval()
    prepare_model(
        model, factorized=True, slice_output=True, seq_len=None, device=device
    )

    train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
        tokenizer=model.tokenizer,
        path=repo_path_to_abs_path(f"datasets/{dataset}.json"),
        device=device,
        prepend_bos=True,
        batch_size=32,
        train_test_split=[0.5, 0.5],
        length_limit=64,
        pad=True,
    )

    return Experiment(
        name, model, train_loader, test_loader, true_edge_func, true_edges_attn_only
    )


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
    ),
]

PRUNE_FUNCS: Dict[str, Callable] = {
    # f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
    #     parameter_integrated_grads_prune_scores,
    #     baseline_weights=pig_baseline,
    #     samples=pig_samples,
    # ),
    "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
    # "ACDC": partial(
    #     acdc_prune_scores,
    #     # tao_exps=list(range(-6, 1)),
    #     tao_exps=[-5],
    #     tao_bases=[1],
    # ),
    "Integrated edge gradients": partial(
        integrated_edge_gradients_prune_scores,
        samples=50,
    ),
    "Prob Gradient": partial(simple_gradient_prune_scores, grad_function="prob"),
    "Exp Logit Gradient": partial(
        simple_gradient_prune_scores, grad_function="logit_exp"
    ),
    "Logit Gradient": partial(simple_gradient_prune_scores, grad_function="logit"),
    "Logprob Gradient": partial(simple_gradient_prune_scores, grad_function="logprob"),
    "Subnetwork Edge Probing": partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=500,
        regularize_lambda=0.5,
        mask_fn=None,  # "hard_concrete",
        dropout_p=0.0,
        init_val=1.0,
        show_train_graph=True,
    ),
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


def measure_experiment_metrics(
    all_prune_scores: ExperimentPruneScores,
) -> MetricMeasurements:
    measurements: MetricMeasurements = defaultdict(lambda: defaultdict(dict))
    for experiment, algo_prune_scores in (exp_pbar := tqdm(all_prune_scores.items())):
        exp_pbar.set_description_str(f"Task: {experiment.name}")
        model = experiment.model
        edges: Set[Edge] = model.edges  # type: ignore
        test_loader = experiment.test_loader
        for func_name, prune_scores in (func_pbar := tqdm(algo_prune_scores.items())):
            func_pbar.set_description_str(f"Pruning with {func_name}")
            group_edges = func_name.startswith("ACDC") or func_name.startswith("IOI")
            count_type = EdgeCounts.GROUPS if group_edges else EdgeCounts.LOGARITHMIC
            test_edge_counts = edge_counts_util(edges, count_type, prune_scores)

            pruned_outs = run_pruned(model, test_loader, test_edge_counts, prune_scores)
            kl_points: Measurements = measure_kl_div(model, test_loader, pruned_outs)
            answer_prob_points: Measurements = measure_answer_prob(
                model, test_loader, pruned_outs, prob_func="softmax"
            )
            del pruned_outs
            measurements["kl_div"][experiment.name][func_name] = kl_points
            measurements["answer_prob"][experiment.name][func_name] = answer_prob_points

            roc_points: Measurements = measure_roc(
                model=model,
                prune_scores=prune_scores,
                correct_edges=experiment.true_edges,
                head_nodes_only=experiment.true_edges_attn_only,
                group_edges=group_edges,
            )
            measurements["roc"][experiment.name][func_name] = roc_points

            t.cuda.empty_cache()
    return measurements


def measurement_figs(measurements: MetricMeasurements) -> Tuple[go.Figure, ...]:
    figs = []
    for metric, experiment_measurements in measurements.items():
        data, y_max = [], 0.0
        for experiment_name, algo_measurements in experiment_measurements.items():
            for algo, points in algo_measurements.items():
                for x, y in points:
                    data.append(
                        {
                            "Task": experiment_name,
                            "Algorithm": algo,
                            "X": x,
                            "Y": max(y, Y_MIN),
                        }
                    )
                    y_max = max(y_max, y)
        if metric == "kl_div":
            figs.append(edge_patching_plot(data, "KL Divergence", True, y_max))
        elif metric == "answer_prob":
            figs.append(edge_patching_plot(data, "Answer Probability", False))
        else:
            assert metric == "roc"
            figs.append(roc_plot(data))
    return tuple(figs)


# Cache measurements with current date and time
# save = False
# load = True
# if save:
#     now = datetime.now()
#     dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
#     repo_path = f".measurement_cache/experiment_measurements-{dt_string}.pkl"
#     with open(repo_path_to_abs_path(repo_path), "wb") as f:
#         pickle.dump(dict(measurements), f)
# if load:
#     repo_path = ".measurement_cache/experiment_measurements-26-11-2023_16-36-05.pkl"
#     with open(repo_path_to_abs_path(repo_path), "rb") as f:
#         measurements = pickle.load(f)

# experiment_steps: Dict[str, Callable] = {
#     "Calculate prune scores": run_prune_funcs,
#     "Measure experiment metric": measure_experiment_metrics,
#     "Draw figures": measurement_figs
# }
experiment_prune_scores = run_prune_funcs(TRANSFORMER_EXPERIMENTS)
measurements = measure_experiment_metrics(experiment_prune_scores)
figs = measurement_figs(measurements)
for fig in figs:
    fig.show()
