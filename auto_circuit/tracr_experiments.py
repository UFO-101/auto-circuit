#%%
from collections import defaultdict
from typing import List

from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import roc_plot
from auto_circuit.prune_algos.prune_algos import (
    ACDC_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    LOGIT_DIFF_GRAD_PRUNE_ALGO,
    LOGIT_MSE_GRAD_PRUNE_ALGO,
    MSE_ACDC_PRUNE_ALGO,
    MSE_CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    MSE_SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import (
    TASK_DICT,
    TRACR_REVERSE_TOKEN_CIRCUIT_TASK,
    TRACR_XPROPORTION_TOKEN_CIRCUIT_TASK,
)
from auto_circuit.types import (
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.misc import load_cache, save_cache
from auto_circuit.visualize import draw_seq_graph

# ------------------------------------ Prune Scores ------------------------------------
compute_prune_scores = False
save_prune_scores = False
load_prune_scores = True

task_prune_scores: TaskPruneScores = defaultdict(dict)
cache_folder_name = ".prune_scores_cache"
if compute_prune_scores:
    REVERSE_ALGOS: List[PruneAlgo] = [
        GROUND_TRUTH_PRUNE_ALGO,
        RANDOM_PRUNE_ALGO,
        ACDC_PRUNE_ALGO,
        LOGIT_DIFF_GRAD_PRUNE_ALGO,  # Fast implementation of Edge Attribution Patching
        SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    ]
    reverse_ps = run_prune_algos([TRACR_REVERSE_TOKEN_CIRCUIT_TASK], REVERSE_ALGOS)

    XPROPORTION_ALGOS: List[PruneAlgo] = [
        GROUND_TRUTH_PRUNE_ALGO,
        RANDOM_PRUNE_ALGO,
        MSE_ACDC_PRUNE_ALGO,
        LOGIT_MSE_GRAD_PRUNE_ALGO,  # Fast implementation of EAP with MSE loss
        # SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
        MSE_SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
        MSE_CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    ]
    xprop_ps = run_prune_algos(
        [TRACR_XPROPORTION_TOKEN_CIRCUIT_TASK], XPROPORTION_ALGOS
    )
    # task_prune_scores: TaskPruneScores = xprop_ps
    task_prune_scores: TaskPruneScores = reverse_ps | xprop_ps
if load_prune_scores:
    filename = "tracr-task-prune-scores-23-02-2024_00-13-23.pkl"
    loaded_cache = load_cache(cache_folder_name, filename)
    task_prune_scores = {k: v | task_prune_scores[k] for k, v in loaded_cache.items()}
if save_prune_scores:
    base_filename = "tracr-task-prune-scores"
    save_cache(task_prune_scores, cache_folder_name, base_filename)


# -------------------------------- Draw Circuit Graphs ---------------------------------
if False:
    for task_key, algo_prune_scores in task_prune_scores.items():
        task = TASK_DICT[task_key]
        for algo_key, prune_scores in algo_prune_scores.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            if not algo == LOGIT_MSE_GRAD_PRUNE_ALGO:
                continue
            print("task:", task.name, "algo:", algo.name)
            draw_seq_graph(
                model=task.model,
                input=next(iter(task.test_loader)).clean,
                prune_scores=prune_scores,
                seq_labels=task.test_loader.seq_labels,
                show_all_edges=False,
            )
            break
        break

# ---------------------------------------- ROC -----------------------------------------

roc_measurements: TaskMeasurements = measure_roc(task_prune_scores)
roc_fig = roc_plot(roc_measurements).show()
