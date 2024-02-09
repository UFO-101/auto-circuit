#%%
import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.circuits.tracr.reverse_official import (
    tracr_reverse_true_edges,
)
from auto_circuit.metrics.official_circuits.circuits.tracr.xproportion_official import (
    tracr_xproportion_official_edges,
)
from auto_circuit.model_utils.tracr.tracr_models import TRACR_TASK_KEY, get_tracr_model
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.ground_truth import ground_truth_prune_scores
from auto_circuit.tasks import Task
from auto_circuit.types import CircuitOutputs, PatchType
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val


def test_tracr_model_outputs(tracr_task_key: TRACR_TASK_KEY = "reverse"):
    device = "cpu"
    tl_model: tl.HookedTransformer
    tl_model, _ = get_tracr_model(tracr_task_key, device)
    dataset_path = f"datasets/tracr/tracr_{tracr_task_key}_len_5_prompts.json"
    _, test_loader = load_datasets_from_json(
        model=None,
        path=repo_path_to_abs_path(dataset_path),
        device=t.device(device),
        prepend_bos=False,
        batch_size=1,
        train_test_split=[0.5, 0.5],
        length_limit=128,
        return_seq_length=True,
        tail_divergence=False,
        random_subet=False,
        pad=False,
    )
    for batch in test_loader:
        logits = tl_model(batch.clean)
        if tracr_task_key == "reverse":
            argmaxs = logits[:, 1:].argmax(dim=-1)
            assert t.equal(argmaxs, batch.answers[..., 0])
        elif tracr_task_key == "xproportion":
            assert t.allclose(logits[:, 1:, 0], batch.answers[..., 0])
        break


def test_reverse_tracr_task_and_avg_answer_val():
    device = "cpu"
    tl_model, _ = get_tracr_model("reverse", device)
    task = Task(
        key="test_tracr_reverse",
        name="test_tracr_reverse",
        batch_size=1,
        batch_count=1,
        token_circuit=True,
        _model_def=tl_model,
        _dataset_name="tracr/tracr_reverse_len_5_prompts",
        separate_qkv=True,
        _true_edge_func=None,
        slice_output="not_first_seq",
    )
    task_model = task.model
    assert isinstance(task_model, PatchableModel)
    assert isinstance(task_model.wrapped_model, tl.HookedTransformer)
    resid_end_node = max(task_model.dests, key=lambda x: x.layer)
    resid_end_edges = {e: 1.0 for e in task_model.edges if e.dest == resid_end_node}
    edge_count = len(resid_end_edges)
    batch = next(iter(task.test_loader))
    clean_logits = task_model(batch.clean)[task_model.out_slice]
    with t.inference_mode():
        circuit_outs: CircuitOutputs = run_circuits(
            task_model,
            task.test_loader,
            [edge_count],
            resid_end_edges,
            PatchType.EDGE_PATCH,
        )
    patched_out = circuit_outs[edge_count][batch.key]
    assert patched_out.shape == clean_logits.shape
    assert t.allclose(patched_out, clean_logits, atol=1e-5)

    # Test batch_avg_answer_val with the answer specified at multiple token positions
    avg_ans_val = batch_avg_answer_val(patched_out, batch)
    assert t.isclose(avg_ans_val.mean(), t.tensor(1.0), atol=1e-5)

    # Test batch_avg_answer_diff with the answer specified at multiple token positions
    avg_ans_diff = batch_avg_answer_diff(patched_out, batch)
    avg_ans_not_eq = (batch.answers != batch.wrong_answers).mean(dtype=t.float32)
    assert t.isclose(avg_ans_diff, avg_ans_not_eq, atol=1e-5)


def test_tracr_task_official_circuit(tracr_task_key: TRACR_TASK_KEY = "xproportion"):
    device = "cpu"
    tl_model, _ = get_tracr_model(tracr_task_key, device)
    if tracr_task_key == "reverse":
        true_edge_func = tracr_reverse_true_edges
    elif tracr_task_key == "xproportion":
        true_edge_func = tracr_xproportion_official_edges
    else:
        raise ValueError(f"Unknown task {tracr_task_key}")
    task = Task(
        key=f"test_tracr_{tracr_task_key}",
        name=f"test_tracr_{tracr_task_key}",
        batch_size=4,
        batch_count=1,
        token_circuit=True,
        _model_def=tl_model,
        _dataset_name=f"tracr/tracr_{tracr_task_key}_len_5_prompts",
        separate_qkv=True,
        _true_edge_func=true_edge_func,
        slice_output="not_first_seq",
    )
    task_model = task.model
    batch = next(iter(task.test_loader))
    clean_logits = task_model(batch.clean)[task_model.out_slice]
    official_circuit_edge_count = task.true_edge_count
    assert official_circuit_edge_count is not None
    with t.inference_mode():
        circuit_outs: CircuitOutputs = run_circuits(
            task_model,
            task.test_loader,
            [official_circuit_edge_count],  # type: ignore
            ground_truth_prune_scores(task),
            PatchType.TREE_PATCH,
        )
    circuit_out = circuit_outs[official_circuit_edge_count][batch.key]
    assert circuit_out.shape == clean_logits.shape
    assert t.allclose(circuit_out, clean_logits, atol=1e-5)


# test_tracr_model_outputs("xproportion")
test_reverse_tracr_task_and_avg_answer_val()
# test_tracr_task_official_circuit("xproportion")
