#%%
from pathlib import Path
from typing import List

import plotly.graph_objects as go
import torch as t

from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import task_roc_plot
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    ANSWER_PROB_METRIC,
    LOGIT_DIFF_METRIC,
)
from auto_circuit.metrics.prune_scores_similarity import prune_score_similarities_plotly
from auto_circuit.prune_algos.prune_algos import (
    GROUND_TRUTH_PRUNE_ALGO,
    OPPOSITE_TREE_PROBING_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    run_prune_algos,
)
from auto_circuit.tasks import (
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    IOI_TOKEN_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import (
    AblationType,
    PatchType,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache
from auto_circuit.utils.tensor_ops import prune_scores_threshold
from auto_circuit.visualize import draw_seq_graph

TASKS: List[Task] = [
    # Token Circuits
    # SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
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
figs: List[go.Figure] = []

# --------------------------------- Load Prune Scores ----------------------------------

# 2000 epoch IOI Docstring tensor prune_scores post-kv-cache-fix
# batch_size=128, batch_count=2, default seed (for both)
filename = "task-prune-scores-16-02-2024_23-27-49.pkl"

# 1000 epoch Sport Players tensor prune_scores post-kv-cache-fix
# batch_size=(10, 20), batch_count=(10, 5), default seed
# filename = "task-prune-scores-16-02-2024_22-22-43.pkl"
cache_folder_name = ".prune_scores_cache"
task_prune_scores = load_cache(cache_folder_name, filename)

# -------------------------------- Draw Circuit Graphs ---------------------------------

if True:
    for task_key, algo_prune_scores in task_prune_scores.items():
        # if not task_key.startswith("Docstring"):
        #     continue
        task = TASK_DICT[task_key]
        if task.key != IOI_TOKEN_CIRCUIT_TASK.key or task.true_edge_count is None:
            continue
        for algo_key, ps in algo_prune_scores.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            keys = [GROUND_TRUTH_PRUNE_ALGO.key]
            if algo_key not in keys:
                continue
            th = prune_scores_threshold(ps, task.true_edge_count)
            circ_edges = dict([(d, (m.abs() >= th).float()) for d, m in ps.items()])
            print("circ_edge_count", sum([m.sum() for m in circ_edges.values()]))
            circ = dict(
                [(d, t.where(m.abs() >= th, m, t.zeros_like(m))) for d, m in ps.items()]
            )
            print("task:", task.name, "algo:", algo.name)
            draw_seq_graph(
                model=task.model,
                prune_scores=circ,
                seq_labels=task.test_loader.seq_labels,
                show_all_edges=False,
            )

# ------------------------------ Prune Scores Similarity -------------------------------

if True:
    prune_scores_similartity_fig = prune_score_similarities_plotly(
        task_prune_scores, [], ground_truths=True
    )
    figs.append(prune_scores_similartity_fig)

# ----------------------------- Opposite Task Prune Scores -----------------------------

compute_opposite_task_prune_scores = False
save_opposite_task_prune_scores = False
load_opposite_task_prune_scores = False
opposite_task_prune_scores: TaskPruneScores = {}
opposite_prune_scores_cache_folder_name = ".opposite_prune_scores_cache"
if compute_opposite_task_prune_scores:
    opposite_task_prune_scores = run_prune_algos(
        TASKS, [OPPOSITE_TREE_PROBING_PRUNE_ALGO]
    )
if save_opposite_task_prune_scores:
    base_filename = "opposite-task-prune-scores"
    save_cache(
        opposite_task_prune_scores,
        opposite_prune_scores_cache_folder_name,
        base_filename,
    )
if load_opposite_task_prune_scores:
    filename = "opposite-task-prune-scores-07-02-2024_17-34-33.pkl"
    opposite_task_prune_scores = load_cache(
        opposite_prune_scores_cache_folder_name, filename
    )
if opposite_task_prune_scores:
    opposite_prune_metric_measurements = measure_prune_metrics(
        [AblationType.RESAMPLE],
        [ANSWER_PROB_METRIC, LOGIT_DIFF_METRIC],
        opposite_task_prune_scores,
        PatchType.TREE_PATCH,
    )
    figs += list(measurement_figs(opposite_prune_metric_measurements, auc_plots=False))

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
    roc_fig = task_roc_plot(roc_measurements)
    figs.append(roc_fig)


# -------------------------------------- Figures ---------------------------------------

for i, fig in enumerate(figs):
    fig.show()
    folder: Path = repo_path_to_abs_path("figures-12")
    # Save figure as pdf in figures folder
    # fig.write_image(str(folder / f"new {i}.pdf"))

#%%
