# %%
import pytest

from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
)
from auto_circuit.prune_algos.prune_algos import (
    GROUND_TRUTH_PRUNE_ALGO,
    run_prune_algos,
)
from auto_circuit.tasks import (
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
)
from auto_circuit.types import AblationType, PatchType


@pytest.mark.slow
def test_docstring_ground_truth_correct_answer_percent():
    ablation = AblationType.RESAMPLE
    task = DOCSTRING_TOKEN_CIRCUIT_TASK
    algo = GROUND_TRUTH_PRUNE_ALGO
    prune_metric = CORRECT_ANSWER_PERCENT_METRIC
    gt_ps = run_prune_algos([task], [algo])
    ablation_measurements = measure_prune_metrics(
        [ablation], [prune_metric], gt_ps, PatchType.TREE_PATCH
    )

    for ablation, prune_metric_measurements in ablation_measurements.items():
        for prune_metric_key, task_measurements in prune_metric_measurements.items():
            assert prune_metric_key == prune_metric.key
            for task_key, algo_measurements in task_measurements.items():
                assert task_key == task.key
                for algo_key, measurements in algo_measurements.items():
                    assert algo_key == algo.key
                    for edge_count, correct_answer_percent in measurements:
                        # assert edge_count == 31
                        print("edge_count", edge_count)
                        print("correct_answer_percent", correct_answer_percent)


@pytest.mark.slow
def test_sports_players_ground_truth_correct_answer_greater_than_incorrect_percent():
    ablation = AblationType.TOKENWISE_MEAN_CORRUPT
    task = SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK
    algo = GROUND_TRUTH_PRUNE_ALGO
    prune_metric = CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC
    gt_ps = run_prune_algos([task], [algo])
    ablation_measurements = measure_prune_metrics(
        [ablation], [prune_metric], gt_ps, PatchType.TREE_PATCH
    )

    for ablation, prune_metric_measurements in ablation_measurements.items():
        for prune_metric_key, task_measurements in prune_metric_measurements.items():
            assert prune_metric_key == prune_metric.key
            for task_key, algo_measurements in task_measurements.items():
                assert task_key == task.key
                for algo_key, measurements in algo_measurements.items():
                    assert algo_key == algo.key
                    for (
                        edge_count,
                        correct_answer_greater_than_incorrect_percent,
                    ) in measurements:
                        print("edge_count", edge_count)
                        print("result", correct_answer_greater_than_incorrect_percent)
                        assert edge_count == 4518
                        assert correct_answer_greater_than_incorrect_percent > 70


# test_docstring_ground_truth_correct_answer_percent()
# test_sports_players_ground_truth_correct_answer_greater_than_incorrect_percent()
