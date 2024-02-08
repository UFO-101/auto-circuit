#%%
from collections import defaultdict
from pathlib import Path
from typing import List

from auto_circuit.metrics.completeness_metrics.same_under_knockouts import (
    TaskCompletenessScores,
    run_same_under_knockouts,
    same_under_knockouts_fig,
)
from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import roc_plot
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_circuit_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    ANSWER_LOGIT_METRIC,
    ANSWER_PROB_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    LOGIT_DIFF_METRIC,
    PruneMetric,
)
from auto_circuit.metrics.prune_scores_similarity import prune_score_similarities_plotly
from auto_circuit.prune_algos.prune_algos import (
    CIRCUIT_PROBING_PRUNE_ALGO,
    CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    OPPOSITE_TREE_PROBING_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    PruneAlgo,
)
from auto_circuit.tasks import (
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    IOI_TOKEN_CIRCUIT_TASK,
    SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import (
    AlgoPruneScores,
    PatchType,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache
from auto_circuit.visualize import draw_seq_graph


def run_prune_funcs(tasks: List[Task], prune_algos: List[PruneAlgo]) -> TaskPruneScores:
    task_prune_scores: TaskPruneScores = {}
    for task in (experiment_pbar := tqdm(tasks)):
        experiment_pbar.set_description_str(f"Task: {task.name}")
        prune_scores_dict: AlgoPruneScores = {}
        for prune_algo in (prune_score_pbar := tqdm(prune_algos)):
            prune_score_pbar.set_description_str(f"Prune scores: {prune_algo.name}")
            ps = dict(list(prune_algo.func(task).items()))
            prune_scores_dict[prune_algo.key] = ps
        task_prune_scores[task.key] = prune_scores_dict
    return task_prune_scores


TASKS: List[Task] = [
    # Token Circuits
    # SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
    # IOI_TOKEN_CIRCUIT_TASK,
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

PRUNE_METRICS: List[PruneMetric] = [
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    ANSWER_PROB_METRIC,
    ANSWER_LOGIT_METRIC,
    LOGIT_DIFF_METRIC,
    # LOGIT_DIFF_PERCENT_METRIC,
]
figs = []

# ------------------------------------ Prune Scores ------------------------------------

compute_prune_scores = False
save_prune_scores = False
load_prune_scores = False

task_prune_scores: TaskPruneScores = defaultdict(dict)
cache_folder_name = ".prune_scores_cache"
if compute_prune_scores:
    task_prune_scores = run_prune_funcs(TASKS, PRUNE_ALGOS)
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
if save_prune_scores:
    base_filename = "task-prune-scores"
    save_cache(task_prune_scores, cache_folder_name, base_filename)

# -------------------------------- Draw Circuit Graphs ---------------------------------

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

# ------------------------------ Prune Scores Similarity -------------------------------

if False:
    prune_scores_similartity_fig = prune_score_similarities_plotly(
        task_prune_scores, [], ground_truths=True
    )
    prune_scores_similartity_fig.show()
    figs.append(prune_scores_similartity_fig)

# ------------------------------------ Completeness ------------------------------------

compute_task_completeness_scores = False
save_task_completeness_scores = False
load_task_completeness_scores = False
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
    figs.append(completeness_fig)

# ----------------------------- Opposite Task Prune Scores -----------------------------

compute_opposite_task_prune_scores = True
save_opposite_task_prune_scores = False
load_opposite_task_prune_scores = False
opposite_task_prune_scores: TaskPruneScores = {}
opposite_prune_scores_cache_folder_name = ".opposite_prune_scores_cache"
if compute_opposite_task_prune_scores:
    opposite_task_prune_scores = run_prune_funcs(TASKS, [OPPOSITE_TREE_PROBING_PRUNE_ALGO])
if save_opposite_task_prune_scores:
    base_filename = "opposite-task-prune-scores"
    save_cache(opposite_task_prune_scores, opposite_prune_scores_cache_folder_name, base_filename)
if load_opposite_task_prune_scores:
    filename = "opposite-task-prune-scores-07-02-2024_17-34-33.pkl"
    opposite_task_prune_scores = load_cache(opposite_prune_scores_cache_folder_name, filename)
if opposite_task_prune_scores:
    opposite_prune_metric_measurements = measure_circuit_metrics(
        [ANSWER_PROB_METRIC, LOGIT_DIFF_METRIC],
        opposite_task_prune_scores,
        PatchType.TREE_PATCH
    )
    figs += list(measurement_figs(opposite_prune_metric_measurements))

# ---------------------------------------- ROC -----------------------------------------

compute_roc_measurements = False
save_roc_measurements = False
load_roc_measurements = False
roc_measurements: TaskMeasurements = {}
roc_cache_folder_name = ".roc_measurements"
if compute_roc_measurements:
    roc_measurements: TaskMeasurements = measure_roc(task_prune_scores)
if save_roc_measurements:
    base_filename = "roc-measurements"
    save_cache(roc_measurements, cache_folder_name, base_filename)
if load_roc_measurements:
    filename = "lala.pkl"
    roc_measurements = load_cache(roc_cache_folder_name, filename)
if roc_measurements:
    roc_fig = roc_plot(roc_measurements)
    roc_fig.show()
    figs.append(roc_fig)


# ----------------------------- Prune Metric Measurements ------------------------------

compute_prune_metric_measurements = False
save_prune_metric_measurements = False
load_prune_metric_measurements = False

cache_folder_name = ".measurement_cache"
metric_measurements = None
if compute_prune_metric_measurements:
    metric_measurements = measure_circuit_metrics(
        PRUNE_METRICS,
        task_prune_scores,
        PatchType.TREE_PATCH,
        reverse_clean_corrupt=False,
    )
    if save_prune_metric_measurements:
        base_filename = "seq-circuit"
        save_cache(metric_measurements, cache_folder_name, base_filename)
if load_prune_metric_measurements:
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

# -------------------------------------- Figures ---------------------------------------

if metric_measurements is not None:
    figs += list(measurement_figs(metric_measurements))
    for i, fig in enumerate(figs):
        fig.show()
        folder: Path = repo_path_to_abs_path("figures-12")
        # Save figure as pdf in figures folder
        # fig.write_image(str(folder / f"new {i}.pdf"))

#%%
