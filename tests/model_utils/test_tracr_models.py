#%%
import torch as t
import transformer_lens as tl
from tracr.compiler.assemble import AssembledTransformerModel

from auto_circuit.data import load_datasets_from_json
from auto_circuit.model_utils.tracr.tracr_models import get_tracr_model
from auto_circuit.prune import run_pruned
from auto_circuit.tasks import Task
from auto_circuit.types import PatchType
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel


def test_reverse_tracr_model():
    device = "cpu"
    tl_model: tl.HookedTransformer
    tracr_model: AssembledTransformerModel
    tl_model, tracr_model = get_tracr_model("reverse", device)
    train_loader, test_loader = load_datasets_from_json(
        model=None,
        path=repo_path_to_abs_path("datasets/tracr/tracr_reverse_len_5_prompts.json"),
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
        logits = tl_model(batch.clean)[:, 1:]
        argmaxs = logits.argmax(dim=-1)
        assert t.equal(argmaxs, batch.answers[0])
        break


def test_reverse_tracr_task():
    device = "cpu"
    tl_model, tracr_model = get_tracr_model("reverse", device)
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
    with t.inference_mode():
        batch = next(iter(task.test_loader))
        clean_logits = task_model(batch.clean)[task_model.out_slice]
        pruned_outs = run_pruned(
            task_model,
            task.test_loader,
            [edge_count],
            resid_end_edges,
            PatchType.EDGE_PATCH,
        )
    patched_out = pruned_outs[edge_count][0]
    assert patched_out.shape == clean_logits.shape
    assert t.allclose(patched_out, clean_logits, atol=1e-5)


# test_reverse_tracr_model()
# test_reverse_tracr_task()
