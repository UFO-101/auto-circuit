#%%
from pytest import approx

from auto_circuit.metrics.prune_metrics.prune_metrics import (
    LOGIT_DIFF_METRIC,
)
from auto_circuit.prune import run_circuits
from auto_circuit.tasks import IOI_TOKEN_CIRCUIT_TASK
from auto_circuit.types import AblationType, CircuitOutputs, Measurements, PatchType


def test_ioi_zero_and_all_edge_circuits_logit_diff_matches_unpatched_model():
    task = IOI_TOKEN_CIRCUIT_TASK
    model = task.model
    n_edges = model.n_edges
    circuit_outs = run_circuits(
        model=model,
        dataloader=task.test_loader,
        test_edge_counts=[0, n_edges],
        prune_scores=model.new_prune_scores(),
        patch_type=PatchType.EDGE_PATCH,
        ablation_type=AblationType.RESAMPLE,
    )
    circ_measurements: Measurements = LOGIT_DIFF_METRIC.metric_func(task, circuit_outs)
    assert len(circ_measurements) == 2
    assert circ_measurements[0][0] == 0
    assert circ_measurements[1][0] == n_edges

    circuit_outs: CircuitOutputs = {0: {}, n_edges: {}}
    for batch in task.test_loader:
        default_corrupt_logits = model(batch.corrupt)[model.out_slice]
        circuit_outs[0][batch.key] = default_corrupt_logits
        default_clean_logits = model(batch.clean)[model.out_slice]
        circuit_outs[n_edges][batch.key] = default_clean_logits
    model_measurements: Measurements = LOGIT_DIFF_METRIC.metric_func(task, circuit_outs)

    for (n_circ_edges, circ_logit_diff), (n_model_edges, model_logit_diff) in zip(
        circ_measurements, model_measurements
    ):
        assert n_circ_edges == n_model_edges
        assert circ_logit_diff == approx(model_logit_diff, abs=1e-5)


# test_ioi_zero_and_all_edge_circuits_logit_diff_matches_unpatched_model()
