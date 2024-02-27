#%%
from pytest import approx

from auto_circuit.metrics.completeness_metrics.same_under_knockouts import (
    CompletenessScores,
    same_under_knockout,
)
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    PruneMetric,
)
from auto_circuit.prune_algos.prune_algos import RANDOM_PRUNE_ALGO, PruneAlgo
from auto_circuit.tasks import IOI_TOKEN_CIRCUIT_TASK, Task
from auto_circuit.types import (
    AblationMeasurements,
    AblationType,
    AlgoPruneScores,
    Measurements,
    PatchType,
    PruneScores,
    TaskPruneScores,
)


def test_kl_div_equal():
    ablation: AblationType = AblationType.RESAMPLE
    task: Task = IOI_TOKEN_CIRCUIT_TASK
    rand_algo: PruneAlgo = RANDOM_PRUNE_ALGO
    kl_metric: PruneMetric = CLEAN_KL_DIV_METRIC
    assert task.true_edge_count is not None
    n_circuit_edges: int = task.true_edge_count
    circuit_ps: PruneScores = rand_algo.func(task)
    algo_ps: AlgoPruneScores = {rand_algo.key: circuit_ps}
    task_ps: TaskPruneScores = {task.key: algo_ps}

    prune_measurements: AblationMeasurements = measure_prune_metrics(
        [ablation],
        [kl_metric],
        task_ps,
        PatchType.TREE_PATCH,
        test_edge_counts=[n_circuit_edges],
    )
    circ_kls: Measurements = prune_measurements[ablation][kl_metric.key][task.key][
        rand_algo.key
    ]
    assert len(circ_kls) == 1
    kl_edge_count, rand_circuit_kl = circ_kls[0]
    assert kl_edge_count == n_circuit_edges

    knockout_ps: PruneScores = task.model.new_prune_scores()
    completeness_scores: CompletenessScores = same_under_knockout(
        task=task,
        circuit_ps=circuit_ps,
        knockout_ps=knockout_ps,
        circuit_size=n_circuit_edges,
    )
    assert len(completeness_scores) == 1
    circ_size, n_knockouts, normal_kl, knockout_kl = next(iter(completeness_scores))

    assert normal_kl == approx(rand_circuit_kl)


# test_kl_div_equal()
