#%%
import pickle
from collections import defaultdict
from datetime import datetime
from typing import List, Set, Tuple

import plotly.graph_objects as go
import torch as t

from auto_circuit.metrics.metrics import (
    ANSWER_LOGIT_METRIC,
    ANSWER_PROB_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    LOGIT_DIFF_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    ROC_METRIC,
)
from auto_circuit.prune import run_pruned
from auto_circuit.prune_algos.prune_algos import (
    CIRCUIT_PROBING_PRUNE_ALGO,
    CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
)
from auto_circuit.tasks import (
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
)
from auto_circuit.types import (
    AlgoPruneScores,
    Edge,
    Metric,
    MetricMeasurements,
    PatchType,
    PruneAlgo,
    Task,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import average_auc_plot, edge_patching_plot, roc_plot


def run_prune_funcs(tasks: List[Task], prune_algos: List[PruneAlgo]) -> TaskPruneScores:
    task_prune_scores: TaskPruneScores = {}
    for task in (experiment_pbar := tqdm(tasks)):
        experiment_pbar.set_description_str(f"Task: {task.name}")
        prune_scores_dict: AlgoPruneScores = {}
        for prune_algo in (prune_score_pbar := tqdm(prune_algos)):
            prune_score_pbar.set_description_str(f"Prune scores: {prune_algo.name}")
            prune_scores_dict[prune_algo] = prune_algo.func(task)
        task_prune_scores[task] = prune_scores_dict
    return task_prune_scores


def measure_circuit_metrics(
    metrics: List[Metric],
    task_prune_scores: TaskPruneScores,
    patch_type: PatchType,
    reverse_clean_corrupt: bool = False,
) -> MetricMeasurements:
    measurements: MetricMeasurements = defaultdict(lambda: defaultdict(dict))
    for task, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task_pbar.set_description_str(f"Task: {task.name}")
        model = task.model
        edges: Set[Edge] = model.edges  # type: ignore
        test_loader = task.test_loader
        for algo, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            algo_pbar.set_description_str(f"Pruning with {algo.name}")
            pruned_outs = run_pruned(
                model=model,
                dataloader=test_loader,
                test_edge_counts=edge_counts_util(edges, prune_scores=prune_scores),
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
                measurements[metric][task][algo] = measurement
            del pruned_outs
            t.cuda.empty_cache()
    return measurements


def measurement_figs(measurements: MetricMeasurements) -> Tuple[go.Figure, ...]:
    figs = []
    for metric, task_measurements in measurements.items():

        data, y_max = [], 0.0
        for task, algo_measurements in task_measurements.items():
            for algo, points in algo_measurements.items():
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

        # Assert all tasks have the same token_circuit value
        assert len(set([task.token_circuit for task in task_measurements.keys()])) == 1
        token_circuit = list(task_measurements.keys())[0].token_circuit

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


TOKEN_CIRCUIT_TASKS: List[Task] = [
    # IOI_TOKEN_CIRCUIT_TASK,
    DOCSTRING_TOKEN_CIRCUIT_TASK,
]

COMPONENT_CIRCUIT_TASKS: List[Task] = [
    IOI_COMPONENT_CIRCUIT_TASK,
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
]

PRUNE_ALGOS: List[PruneAlgo] = [
    GROUND_TRUTH_PRUNE_ALGO,
    # ACT_MAG_PRUNE_ALGO,
    # RANDOM_PRUNE_ALGO,
    # EDGE_ATTR_PATCH_PRUNE_ALGO,
    # ACDC_PRUNE_ALGO,
    # INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    # LOGPROB_GRAD_PRUNE_ALGO,
    # LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    # SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    CIRCUIT_PROBING_PRUNE_ALGO,
    # SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    # CIRCUIT_TREE_PROBING_PRUNE_ALGO
]

METRICS: List[Metric] = [
    ROC_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    ANSWER_PROB_METRIC,
    ANSWER_LOGIT_METRIC,
    LOGIT_DIFF_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
]

task_prune_scores: TaskPruneScores = defaultdict(dict)
metric_measurements: MetricMeasurements = defaultdict(lambda: defaultdict(dict))
task_prune_scores = run_prune_funcs(TOKEN_CIRCUIT_TASKS, PRUNE_ALGOS)
metric_measurements = measure_circuit_metrics(
    METRICS, task_prune_scores, PatchType.TREE_PATCH, reverse_clean_corrupt=False
)

# Cache metric_measurements with current date and time
save = False
load = False
if save:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    repo_path = f".measurement_cache/seq-circuit-{dt_string}.pkl"
    # repo_path = ".measurement_cache/token_pos_tree_patch_2.pkl"
    with open(repo_path_to_abs_path(repo_path), "wb") as f:
        pickle.dump(dict(metric_measurements), f)
if load:
    # cache_path = "experiment_measurements-26-11-2023_16-36-05.pkl"
    # cache_path = "seq_experiment_measurements-28-11-2023_15-52-47.pkl"
    # cache_path = "token_pos_exp_1.pkl"
    # cache_path = "token_pos_tree_patch_2.pkl"
    # cache_path = "experiment_measurements-10-12-2023_21-50-32.pkl"
    # cache_path = "experiment_measurements-10-12-2023_23-23-01.pkl"
    # cache_path = "experiment_measurements-10-12-2023_23-45-00.pkl"
    # cache_path = "comp-circuit-12-12-2023_00-28-44.pkl"
    # cache_path = "tok-circuit_measurements-11-12-2023_23-37-24.pkl"
    # cache_path = "inv-tok-circuit-12-12-2023_01-18-14.pkl"
    # cache_path = "inv-comp-circuit-12-12-2023_02-07-32.pkl"
    # cache_path = "seq-circuit-12-12-2023_22-23-54.pkl"
    # cache_path = "seq-circuit-12-12-2023_23-14-49.pkl"
    # cache_path = "seq-circuit-12-12-2023_23-37-41.pkl"
    # cache_path = "ABS-seq-circuit-13-12-2023_01-54-31.pkl"
    # cache_path = "seq-circuit-13-12-2023_04-10-38.pkl"
    # cache_path = "seq-circuit-13-12-2023_04-45-15.pkl"
    cache_path = "seq-circuit-13-12-2023_06-30-20.pkl"
    with open(repo_path_to_abs_path(".measurement_cache/" + cache_path), "rb") as f:
        loaded_measurements = pickle.load(f)
    # merge with existing metric_measurements
    same_idx = defaultdict(int)
    # for metric, loaded_task_measurements in loaded_measurements.items():
    #     corresponding_metrics = list(filter(lambda m: m.name == metric.name,
    # metric_measurements.keys()))
    #     corresponding_task_measurements = metric_measurements[
    # corresponding_metrics[same_idx[metric.name]]]
    #     same_idx[metric.name] += 1
    #     for task, loaded_algo_measurements in loaded_task_measurements.items():
    #         try:
    #             corresponding_task = next(filter(lambda t: t.name == task.name,
    # corresponding_task_measurements.keys()))
    #             corresponding_algo_measurements =
    # corresponding_task_measurements[corresponding_task]
    #             loaded_measurements[metric][task][LOGPROB_DIFF_GRAD_PRUNE_ALGO] =
    # corresponding_algo_measurements[LOGPROB_DIFF_GRAD_PRUNE_ALGO]
    #         except StopIteration:
    #             continue
    #         for algo, points in loaded_algo_measurements.items():
    #             if algo.name == "Edge Attribution Patching":
    #                 del loaded_measurements[metric][task][algo]
    #                 break
    for metric, loaded_task_measurements in loaded_measurements.items():
        corresponding_metrics = list(
            filter(lambda m: m.name == metric.name, metric_measurements.keys())
        )
        corresponding_task_measurements = metric_measurements[
            corresponding_metrics[same_idx[metric.name]]
        ]
        same_idx[metric.name] += 1
        for task, loaded_algo_measurements in loaded_task_measurements.items():
            try:
                corresponding_task = next(
                    filter(
                        lambda t: t.name == task.name,
                        corresponding_task_measurements.keys(),
                    )
                )
                corresponding_algo_measurements = corresponding_task_measurements[
                    corresponding_task
                ]
                loaded_measurements[metric][task][
                    SUBNETWORK_TREE_PROBING_PRUNE_ALGO
                ] = corresponding_algo_measurements[SUBNETWORK_TREE_PROBING_PRUNE_ALGO]
                loaded_measurements[metric][task][
                    CIRCUIT_TREE_PROBING_PRUNE_ALGO
                ] = corresponding_algo_measurements[CIRCUIT_TREE_PROBING_PRUNE_ALGO]
            except StopIteration:
                continue
            # for algo, points in loaded_algo_measurements.items():
            #     if algo.name == "Edge Attribution Patching":
            #         del loaded_measurements[metric][task][algo]
            #         break
    metric_measurements = loaded_measurements

# experiment_steps: Dict[str, Callable] = {
#     "Calculate prune scores": run_prune_funcs,
#     "Measure experiment metric": measure_experiment_metrics,
#     "Draw figures": measurement_figs
# }
figs = measurement_figs(metric_measurements)
for i, fig in enumerate(figs):
    fig.show()
    # folder: Path = repo_path_to_abs_path("figures-11")
    # Save figure as pdf in figures folder
    # fig.write_image(str(folder / f"new {i}.pdf"))

#%%
for metric, task_measurements in metric_measurements.items():
    for task, algo_measurements in task_measurements.items():
        for algo, points in algo_measurements.items():
            if algo.name == "GT":
                metric_measurements[metric][task][GROUND_TRUTH_PRUNE_ALGO] = points
                del metric_measurements[metric][task][algo]
                break
#%%

count = 0
#%%
for metric, task_measurements in metric_measurements.items():
    if metric.name == "KL Divergence":
        del metric_measurements[metric]
        if count == 0:
            metric_measurements[CLEAN_KL_DIV_METRIC] = task_measurements
            count += 1
        else:
            metric_measurements[CORRUPT_KL_DIV_METRIC] = task_measurements
        break
