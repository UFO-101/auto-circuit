#%%
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
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
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
from auto_circuit.visualize import draw_seq_graph

# ------------------------------------ Prune Scores ------------------------------------

REVERSE_ALGOS: List[PruneAlgo] = [
    GROUND_TRUTH_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    ACDC_PRUNE_ALGO,
    LOGIT_DIFF_GRAD_PRUNE_ALGO,  # Fast implementation of Edge Attribution Patching
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
]
# reverse_ps = run_prune_algos([TRACR_REVERSE_TOKEN_CIRCUIT_TASK], REVERSE_ALGOS)

XPROPORTION_ALGOS: List[PruneAlgo] = [
    GROUND_TRUTH_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    # MSE_ACDC_PRUNE_ALGO,
    LOGIT_MSE_GRAD_PRUNE_ALGO,  # Fast implementation of Edge Attribution Patching (MSE)
    # SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    # MSE_SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    # MSE_CIRCUIT_TREE_PROBING_PRUNE_ALGO
]
xprop_ps = run_prune_algos([TRACR_XPROPORTION_TOKEN_CIRCUIT_TASK], XPROPORTION_ALGOS)

task_prune_scores: TaskPruneScores = xprop_ps
# task_prune_scores: TaskPruneScores = reverse_ps | xprop_ps

# -------------------------------- Draw Circuit Graphs ---------------------------------
if True:
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
