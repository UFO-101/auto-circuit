#%%
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import plotly.graph_objects as go
import torch as t

from auto_circuit.metrics.metrics import (
    ANSWER_LOGIT_METRIC,
    ANSWER_PROB_METRIC,
    CLEAN_KL_DIV_METRIC,
    METRIC_DICT,
    ROC_METRIC,
    Metric,
)
from auto_circuit.metrics.prune_scores_similarity import (
    prune_score_similarities_plotly,
)
from auto_circuit.prune import run_pruned
from auto_circuit.prune_algos.prune_algos import (
    CIRCUIT_PROBING_PRUNE_ALGO,
    INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    PruneAlgo,
)
from auto_circuit.tasks import (
    CAPITAL_CITIES_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import (
    AlgoPruneScores,
    MetricMeasurements,
    PatchType,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache
from auto_circuit.visualize import average_auc_plot, edge_patching_plot, roc_plot


def run_prune_funcs(tasks: List[Task], prune_algos: List[PruneAlgo]) -> TaskPruneScores:
    task_prune_scores: TaskPruneScores = {}
    for task in (experiment_pbar := tqdm(tasks)):
        experiment_pbar.set_description_str(f"Task: {task.name}")
        prune_scores_dict: AlgoPruneScores = {}
        for prune_algo in (prune_score_pbar := tqdm(prune_algos)):
            prune_score_pbar.set_description_str(f"Prune scores: {prune_algo.name}")
            prune_scores_dict[prune_algo.key] = prune_algo.func(task)
        task_prune_scores[task.key] = prune_scores_dict
    return task_prune_scores


def measure_circuit_metrics(
    metrics: List[Metric],
    task_prune_scores: TaskPruneScores,
    patch_type: PatchType,
    reverse_clean_corrupt: bool = False,
) -> MetricMeasurements:
    measurements: MetricMeasurements = defaultdict(lambda: defaultdict(dict))
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        task_pbar.set_description_str(f"Task: {task.name}")
        test_loader = task.test_loader
        for algo_key, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            algo = PRUNE_ALGO_DICT[algo_key]
            algo_pbar.set_description_str(f"Pruning with {algo.name}")
            pruned_outs = run_pruned(
                model=task.model,
                dataloader=test_loader,
                test_edge_counts=edge_counts_util(task.model.edges, None, prune_scores),
                prune_scores=prune_scores,
                patch_type=patch_type,
                reverse_clean_corrupt=reverse_clean_corrupt,
                render_graph=False,
                render_prune_scores=True,
                render_top_n=30,
                render_file_path="figures-6/docstring-viz.pdf",
            )
            for metric in (metric_pbar := tqdm(metrics)):
                metric_pbar.set_description_str(f"Measuring {metric.name}")
                measurement = metric.metric_func(task, prune_scores, pruned_outs)
                measurements[metric.key][task.key][algo.key] = measurement
            del pruned_outs
            t.cuda.empty_cache()
    return measurements


def measurement_figs(measurements: MetricMeasurements) -> Tuple[go.Figure, ...]:
    figs = []
    for metric_key, task_measurements in measurements.items():
        token_circuit = TASK_DICT[list(task_measurements.keys())[0]].token_circuit
        metric = METRIC_DICT[metric_key]
        data, y_max = [], 0.0
        for task_key, algo_measurements in task_measurements.items():
            task = TASK_DICT[task_key]
            # Assert all tasks have the same token_circuit value
            assert task.token_circuit == token_circuit

            for algo_key, points in algo_measurements.items():
                algo = PRUNE_ALGO_DICT[algo_key]
                if len(points) > 1:
                    for x, y in points:
                        data.append(
                            {
                                "Task": task.name,
                                "Algorithm": algo.name,
                                "X": max(x, 0.5) if metric.log_x else x,
                                "Y": y
                                if metric.y_min is None
                                else max(y, metric.y_min),
                            }
                        )
                        y_max = max(y_max, y)

        if metric == ROC_METRIC:
            figs.append(roc_plot(data, task_measurements))
        else:
            y_max = None if metric.y_min is None else y_max
            figs.append(
                edge_patching_plot(
                    data,
                    task_measurements,
                    metric.name,
                    metric.log_x,
                    metric.log_y,
                    metric.y_axes_match,
                    token_circuit,
                    y_max,
                    metric.y_min,
                )
            )
        figs.append(
            average_auc_plot(
                task_measurements,
                metric.name,
                metric.log_x,
                metric.log_y,
                metric.y_min,
                metric.lower_better,
            )
        )
    return tuple(figs)


TASKS: List[Task] = [
    # Token Circuits
    # IOI_TOKEN_CIRCUIT_TASK,
    # DOCSTRING_TOKEN_CIRCUIT_TASK,
    # Component Circuits
    # IOI_COMPONENT_CIRCUIT_TASK,
    # DOCSTRING_COMPONENT_CIRCUIT_TASK,
    # GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    # Autoencoder Component Circuits
    # IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    # GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK
    # ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    CAPITAL_CITIES_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
]

PRUNE_ALGOS: List[PruneAlgo] = [
    # GROUND_TRUTH_PRUNE_ALGO,
    # ACT_MAG_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    # EDGE_ATTR_PATCH_PRUNE_ALGO,
    # ACDC_PRUNE_ALGO,
    INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    # LOGPROB_GRAD_PRUNE_ALGO,
    LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    CIRCUIT_PROBING_PRUNE_ALGO,
    # SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    # CIRCUIT_TREE_PROBING_PRUNE_ALGO
]

METRICS: List[Metric] = [
    # ROC_METRIC,
    CLEAN_KL_DIV_METRIC,
    # CORRUPT_KL_DIV_METRIC,
    ANSWER_PROB_METRIC,
    ANSWER_LOGIT_METRIC,
    # LOGIT_DIFF_METRIC,
    # LOGIT_DIFF_PERCENT_METRIC,
]

compute_prune_scores = True
load_prune_scores = False
save_prune_scores = False

task_prune_scores: TaskPruneScores = defaultdict(dict)
if compute_prune_scores:
    task_prune_scores = run_prune_funcs(TASKS, PRUNE_ALGOS)
cache_folder_name = ".prune_scores_cache"
if load_prune_scores:
    filename = "task-prune-scores-09-01-2024_20-13-48.pkl"
    loaded_cache = load_cache(cache_folder_name, filename)
    task_prune_scores = {k: v | task_prune_scores[k] for k, v in loaded_cache.items()}
if save_prune_scores:
    base_filename = "task-prune-scores"
    save_cache(task_prune_scores, cache_folder_name, base_filename)

prune_scores_similartity_fig = prune_score_similarities_plotly(
    task_prune_scores, [10, 100, 1000], ground_truths=False
)
prune_scores_similartity_fig.show()
#%%

metric_measurements: MetricMeasurements = defaultdict(lambda: defaultdict(dict))
metric_measurements = measure_circuit_metrics(
    METRICS, task_prune_scores, PatchType.TREE_PATCH, reverse_clean_corrupt=False
)

# Cache metric_measurements with current date and time
save_metric_measurements = False
load_metric_measurements = False
(cache_folder_name,) = (".measurement_cache",)
if save_metric_measurements:
    base_filename = "seq-circuit"
    save_cache(metric_measurements, cache_folder_name, base_filename)
if load_metric_measurements:
    filename = "seq-circuit-13-12-2023_06-30-20.pkl"
    metric_measurements = load_cache(cache_folder_name, filename)

# experiment_steps: Dict[str, Callable] = {
#     "Calculate prune scores": run_prune_funcs,
#     "Measure experiment metric": measure_experiment_metrics,
#     "Draw figures": measurement_figs
# }
figs = measurement_figs(metric_measurements)
for i, fig in enumerate(figs):
    fig.show()
    folder: Path = repo_path_to_abs_path("figures-12")
    # Save figure as pdf in figures folder
    # fig.write_image(str(folder / f"new {i}.pdf"))

#%%
