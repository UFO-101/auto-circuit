#%%
import pytest

from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import LOGIT_DIFF_PERCENT_METRIC
from auto_circuit.prune_algos.prune_algos import (
    GROUND_TRUTH_PRUNE_ALGO,
    run_prune_algos,
)
from auto_circuit.tasks import IOI_TOKEN_CIRCUIT_TASK
from auto_circuit.types import AblationType, PatchType


@pytest.mark.slow
def test_ioi_ground_truth_logit_diff():
    ablation_type = AblationType.TOKENWISE_MEAN_CORRUPT
    task = IOI_TOKEN_CIRCUIT_TASK
    algo = GROUND_TRUTH_PRUNE_ALGO
    prune_metric = LOGIT_DIFF_PERCENT_METRIC
    gt_ps = run_prune_algos([task], [algo])
    ablation_measurements = measure_prune_metrics(
        [ablation_type], [prune_metric], gt_ps, PatchType.TREE_PATCH
    )

    for ablation, prune_metric_measurements in ablation_measurements.items():
        assert ablation == ablation_type
        for prune_metric_key, task_measurements in prune_metric_measurements.items():
            assert prune_metric_key == prune_metric.key
            for task_key, algo_measurements in task_measurements.items():
                assert task_key == task.key
                for algo_key, measurements in algo_measurements.items():
                    assert algo_key == algo.key
                    for edge_count, logit_diff_percent in measurements:
                        print(
                            "edge_count",
                            edge_count,
                            "logit_diff_percent",
                            logit_diff_percent,
                        )
                        # assert edge_count == 771
                        # assert 75 < logit_diff_percent < 95


# test_ioi_ground_truth_logit_diff()
