from functools import partial
from typing import List

from auto_circuit.prune_algos.circuit_probing import circuit_probing_prune_scores
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT, PruneAlgo
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import AlgoPruneScores, Edge, TaskPruneScores
from auto_circuit.utils.custom_tqdm import tqdm


def run_constrained_prune_funcs(task_prune_scores: TaskPruneScores) -> TaskPruneScores:
    """
    For each task and each algorithm's [`PruneScores`][auto_circuit.types.PruneScores],
    run
    [`circuit_probing_prune_scores`][auto_circuit.prune_algos.circuit_probing.circuit_probing_prune_scores]
    with the `avoid_edges` parameter set to the top `true_edge_count` edges of the
    [`PruneScores`][auto_circuit.types.PruneScores] of the given task and algorithm.

    This is intended to test if we can find a circuit very different from the original
    solutions found by the given
    [`PruneAlgo`s][auto_circuit.prune_algos.prune_algos.PruneAlgo], that still perform
    well on the task.

    Args:
        task_prune_scores: Prune scores for each task and algorithm.

    Returns:
        A new set of prune scores for each task and algorithm, that attempts to be as
            different as possible from the original solutions found by the given
            algorithms while still performing well on the task.

    Note:
        This is an experimental function and the internal parameters used to find the
        new edges may not be well tuned.
    """
    constrained_task_prune_scores: TaskPruneScores = {}
    for task_key in (experiment_pbar := tqdm(task_prune_scores.keys())):
        task = TASK_DICT[task_key]
        experiment_pbar.set_description_str(f"Task: {task.name}")
        constrained_ps: AlgoPruneScores = {}
        algo_prune_scores = task_prune_scores[task_key]
        for algo_key, algo_ps in (prune_score_pbar := tqdm(algo_prune_scores.items())):
            if algo_key.startswith("Constrained") or algo_key not in [
                "Official Circuit",
                "Tree Probing",
            ]:
                continue
            sorted_edges: List[Edge] = list(
                sorted(algo_ps.keys(), key=lambda x: abs(algo_ps[x]), reverse=True)
            )
            algo_circuit = set([e for e in sorted_edges[: task.true_edge_count]])
            prune_score_pbar.set_description_str(f"Constrained Pruning: {algo_key}")
            constrained_algo = PruneAlgo(
                key="Constrained Circuit Probing " + algo_key,
                name=f"Not {PRUNE_ALGO_DICT[algo_key].name} Circuit Probing",
                _short_name=f"¬{PRUNE_ALGO_DICT[algo_key].short_name} TP",
                func=partial(
                    circuit_probing_prune_scores,
                    learning_rate=0.1,
                    epochs=2000,
                    regularize_lambda=0.1,
                    mask_fn="hard_concrete",
                    show_train_graph=True,
                    circuit_sizes=["true_size"],
                    tree_optimisation=True,
                    avoid_edges=algo_circuit,
                    avoid_lambda=0.3,
                ),
            )
            PRUNE_ALGO_DICT[constrained_algo.key] = constrained_algo
            if constrained_algo.key not in algo_prune_scores:
                print(f"Running {constrained_algo.name}")
                constrained_ps[constrained_algo.key] = constrained_algo.func(
                    task.model, task.train_loader, task.true_edges
                )
            else:
                print(f"Already ran {constrained_algo.name}")
        constrained_task_prune_scores[task_key] = constrained_ps
    return constrained_task_prune_scores
