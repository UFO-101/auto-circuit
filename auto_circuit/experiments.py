#%%
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import torch as t

from auto_circuit.completeness_algos.same_under_knockouts import (
    TaskCompletenessScores,
    run_same_under_knockouts,
    same_under_knockouts_fig,
)
from auto_circuit.metrics.metrics import (
    ANSWER_LOGIT_METRIC,
    ANSWER_PROB_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    LOGIT_DIFF_METRIC,
    METRIC_DICT,
    ROC_METRIC,
    Metric,
)
from auto_circuit.metrics.prune_scores_similarity import prune_score_similarities_plotly
from auto_circuit.prune import run_pruned
from auto_circuit.prune_algos.prune_algos import (
    CIRCUIT_PROBING_PRUNE_ALGO,
    CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    PruneAlgo,
)
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores
from auto_circuit.tasks import (
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    IOI_TOKEN_CIRCUIT_TASK,
    SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import (
    AlgoPruneScores,
    Edge,
    MetricMeasurements,
    PatchType,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache
from auto_circuit.visualize import (
    average_auc_plot,
    draw_seq_graph,
    edge_patching_plot,
    roc_plot,
)


def run_prune_funcs(tasks: List[Task], prune_algos: List[PruneAlgo]) -> TaskPruneScores:
    task_prune_scores: TaskPruneScores = {}
    for task in (experiment_pbar := tqdm(tasks)):
        experiment_pbar.set_description_str(f"Task: {task.name}")
        prune_scores_dict: AlgoPruneScores = {}
        for prune_algo in (prune_score_pbar := tqdm(prune_algos)):
            prune_score_pbar.set_description_str(f"Prune scores: {prune_algo.name}")
            ps = dict(list(prune_algo.func(task).items())[: task.true_edge_count])
            prune_scores_dict[prune_algo.key] = ps
        task_prune_scores[task.key] = prune_scores_dict
    return task_prune_scores


def run_constrained_prune_funcs(task_prune_scores: TaskPruneScores) -> TaskPruneScores:
    constrained_task_prune_scores: TaskPruneScores = {}
    for task_key in (experiment_pbar := tqdm(task_prune_scores.keys())):
        task = TASK_DICT[task_key]
        experiment_pbar.set_description_str(f"Task: {task.name}")
        constrained_ps: AlgoPruneScores = {}
        algo_prune_scores = task_prune_scores[task_key]
        for algo_key, algo_ps in (prune_score_pbar := tqdm(algo_prune_scores.items())):
            if (
                algo_key.startswith("Constrained")
                or algo_key not in ["Official Circuit", "Tree Probing"]
                or True
            ):
                continue
            sorted_edges: List[Edge] = list(
                sorted(algo_ps.keys(), key=lambda x: abs(algo_ps[x]), reverse=True)
            )
            algo_circuit = set([e for e in sorted_edges[: task.true_edge_count]])
            prune_score_pbar.set_description_str(f"Constrained Pruning: {algo_key}")
            constrained_algo = PruneAlgo(
                key="Constrained Circuit Probing " + algo_key,
                name=f"Not {PRUNE_ALGO_DICT[algo_key].name} Circuit Probing",
                short_name=f"¬{PRUNE_ALGO_DICT[algo_key].short_name} TP",
                func=partial(
                    subnetwork_probing_prune_scores,
                    learning_rate=0.1,
                    epochs=2000,
                    regularize_lambda=0.1,
                    mask_fn="hard_concrete",
                    show_train_graph=True,
                    circuit_size="true_size",
                    tree_optimisation=True,
                    avoid_edges=algo_circuit,
                    avoid_lambda=0.3,
                ),
            )
            PRUNE_ALGO_DICT[constrained_algo.key] = constrained_algo
            if constrained_algo.key not in algo_prune_scores:
                print(f"Running {constrained_algo.name}")
                constrained_ps[constrained_algo.key] = constrained_algo.func(task)
            else:
                print(f"Already ran {constrained_algo.name}")
        constrained_task_prune_scores[task_key] = constrained_ps
    return constrained_task_prune_scores


def default_factory() -> Dict[Any, Dict[Any, Any]]:
    return defaultdict(dict)


def measure_circuit_metrics(
    metrics: List[Metric],
    task_prune_scores: TaskPruneScores,
    patch_type: PatchType,
    reverse_clean_corrupt: bool = False,
) -> MetricMeasurements:
    measurements: MetricMeasurements = defaultdict(default_factory)
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
                        # !!!! Make multiple different ones if not sharing y-axis
                        # Also, why are the x-values not quite right?
                        y_max = max(y_max, y)

        if metric == ROC_METRIC:
            figs.append(roc_plot(data, task_measurements))
        else:
            y_max = None if metric.y_min is None or not metric.y_axes_match else y_max
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
    SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
    IOI_TOKEN_CIRCUIT_TASK,
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    # Component Circuits
    # SPORTS_PLAYERS_COMPONENT_CIRCUIT_TASK,
    # IOI_COMPONENT_CIRCUIT_TASK,
    # DOCSTRING_COMPONENT_CIRCUIT_TASK,
    # GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    # Autoencoder Component Circuits
    # IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    # GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK
    # ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    # CAPITAL_CITIES_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
]

PRUNE_ALGOS: List[PruneAlgo] = [
    GROUND_TRUTH_PRUNE_ALGO,
    # ACT_MAG_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    # EDGE_ATTR_PATCH_PRUNE_ALGO,
    # ACDC_PRUNE_ALGO,
    INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    # LOGPROB_GRAD_PRUNE_ALGO,
    LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    CIRCUIT_PROBING_PRUNE_ALGO,
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    CIRCUIT_TREE_PROBING_PRUNE_ALGO,
]

METRICS: List[Metric] = [
    ROC_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    ANSWER_PROB_METRIC,
    ANSWER_LOGIT_METRIC,
    LOGIT_DIFF_METRIC,
    # LOGIT_DIFF_PERCENT_METRIC,
]
figs = []

compute_prune_scores = False
save_prune_scores = False
load_prune_scores = False

task_prune_scores: TaskPruneScores = defaultdict(dict)
cache_folder_name = ".prune_scores_cache"
if compute_prune_scores:
    prune_scores = run_prune_funcs(TASKS, PRUNE_ALGOS)
    print("prune_scores.keys():", prune_scores.keys())
    constrained_ps = run_constrained_prune_funcs(prune_scores)
    print("constrained_ps.keys():", constrained_ps.keys())
    task_prune_scores = {k: v | constrained_ps[k] for k, v in prune_scores.items()}
    print("task_prune_scores.keys():", task_prune_scores.keys())
if load_prune_scores:
    # filename = "task-prune-scores-09-01-2024_20-13-48.pkl"
    # filename = "task-prune-scores-19-01-2024_20-19-04.pkl"
    # filename = "task-prune-scores-21-01-2024_01-19-29.pkl"
    # filename = "task-prune-scores-24-01-2024_20-12-00.pkl"
    # filename = "task-prune-scores-30-01-2024_17-56-05.pkl"

    # Sports Players 100 epochs
    # filename = "task-prune-scores-30-01-2024_19-41-40.pkl" !!!! Don't use this one

    # IOI and Docstring [100, 1000, ...] circuits 2000 epochs
    # filename = "task-prune-scores-31-01-2024_01-51-25.pkl"
    # Sports Players 500 epochs
    # filename = "task-prune-scores-31-01-2024_22-36-47.pkl"
    filename = (
        "icml-2024-sports-ioi-docstring-02-02-2024_04-09-35.pkl"  # 2 above combined
    )

    loaded_cache = load_cache(cache_folder_name, filename)
    task_prune_scores = {k: v | task_prune_scores[k] for k, v in loaded_cache.items()}
    run_constrained_prune_funcs(task_prune_scores)
if save_prune_scores:
    base_filename = "task-prune-scores"
    save_cache(task_prune_scores, cache_folder_name, base_filename)

if False:
    for task_key, algo_prune_scores in task_prune_scores.items():
        task = TASK_DICT[task_key]
        for algo_key, prune_scores in algo_prune_scores.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            print("task:", task.name, "algo:", algo.name)
            draw_seq_graph(
                model=task.model,
                input=next(iter(task.test_loader)).clean,
                prune_scores=prune_scores,
                seq_labels=task.test_loader.seq_labels,
            )
            break
        break

if False:
    prune_scores_similartity_fig = prune_score_similarities_plotly(
        task_prune_scores, [], ground_truths=True
    )
    prune_scores_similartity_fig.show()
    figs.append(prune_scores_similartity_fig)

compute_task_completeness_scores = False
save_task_completeness_scores = False
load_task_completeness_scores = True
task_completeness_scores: TaskCompletenessScores = {}
if compute_task_completeness_scores:
    task_completeness_scores: TaskCompletenessScores = run_same_under_knockouts(
        task_prune_scores,
        # algo_keys=["Official Circuit", "Circuit Probing", "Random"],
        algo_keys=["Official Circuit", "Random"],
        learning_rate=0.01,
        epochs=100,
        regularize_lambda=0,
        hard_concrete_threshold=0.0,
    )
cache_folder_name = ".completeness_scores"
if save_task_completeness_scores:
    base_filename = "task-completeness-scores"
    save_cache(task_completeness_scores, cache_folder_name, base_filename)
if load_task_completeness_scores:
    # IOI and Docstring [100, 1000, ...] circuits
    # 2000 epochs pruning - 500 epochs completeness
    # filename = "task-completeness-scores-02-02-2024_02-05-58.pkl"
    # filename = "task-completeness-scores-02-02-2024_04-50-10.pkl"
    filename = "combined_task_completeness_scores-06-02-2024_18-56-54.pkl"
    task_completeness_scores = load_cache(cache_folder_name, filename)
if task_completeness_scores:
    completeness_fig = same_under_knockouts_fig(task_completeness_scores)
    completeness_fig.show()

compute_metric_measurements = False
save_metric_measurements = False
load_metric_measurements = False

cache_folder_name = ".measurement_cache"
metric_measurements = None
if compute_metric_measurements:
    metric_measurements = measure_circuit_metrics(
        METRICS, task_prune_scores, PatchType.TREE_PATCH, reverse_clean_corrupt=False
    )
    if save_metric_measurements:
        base_filename = "seq-circuit"
        save_cache(metric_measurements, cache_folder_name, base_filename)
if load_metric_measurements:
    # filename = "seq-circuit-13-12-2023_06-30-20.pkl"
    # filename = "seq-circuit-24-01-2024_20-17-35.pkl"

    # IOI and Docstring [100, 1000, ...] circuits 2000 epochs
    # filename = "seq-circuit-31-01-2024_02-20-11.pkl"
    # Sports Players 500 epochs
    # filename = "seq-circuit-01-02-2024_03-39-54.pkl"
    # 2 above combined
    # filename = "icml-2024-sports-ioi-docstring-02-02-2024_04-35-30.pkl"
    # Mean ablate sports ground truth only
    # ("icml-2024-sports-ioi-dostring" +
    # "mean-ablate-sport-ground-truth-02-02-2024_06-17-46.pkl")
    # filename = "icml-2024-all3-fix-official-02-02-2024_07-28-15.pkl"
    filename = "icml-2024-all3-fix-official-02-02-2024_07-38-20.pkl"
    # filename = "icml-2024-all3-fix-official-mean-ablate-02-02-2024_07-45-23.pkl"

    metric_measurements = load_cache(cache_folder_name, filename)

# experiment_steps: Dict[str, Callable] = {
#     "Calculate prune scores": run_prune_funcs,
#     "Measure experiment metric": measure_experiment_metrics,
#     "Draw figures": measurement_figs
# }
if metric_measurements is not None:
    figs += list(measurement_figs(metric_measurements))
    for i, fig in enumerate(figs):
        fig.show()
        folder: Path = repo_path_to_abs_path("figures-12")
        # Save figure as pdf in figures folder
        # fig.write_image(str(folder / f"new {i}.pdf"))

#%%
