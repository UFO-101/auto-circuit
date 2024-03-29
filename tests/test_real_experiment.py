import os

import torch as t

from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.random_edges import random_prune_scores
from auto_circuit.tasks import Task
from auto_circuit.types import PatchType
from auto_circuit.utils.graph_utils import edge_counts_util

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def test_kl_vs_edges():
    """Test that experiments satisfy basic requirements with a real model."""
    task = Task(
        key="test_kl_vs_edges",
        name="test_kl_vs_edges",
        batch_size=1,
        batch_count=1,
        token_circuit=False,
        _model_def="attn-only-4l",
        _dataset_name="mini_prompts",
    )

    test_batch = next(iter(task.test_loader))
    with t.inference_mode():
        clean_out = task.model(test_batch.clean)[:, -1]
        corrupt_out = task.model(test_batch.corrupt)[:, -1]

    prune_scores = random_prune_scores(task.model, task.train_loader, task.true_edges)
    test_edge_counts = edge_counts_util(task.model.edges, [0.0, 5, 1.0])
    pruned_outs = run_circuits(
        model=task.model,
        dataloader=task.test_loader,
        test_edge_counts=test_edge_counts,
        prune_scores=prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        render_graph=False,
    )
    key = test_batch.key
    assert t.allclose(corrupt_out, pruned_outs[0][key], atol=1e-3)
    assert not t.allclose(clean_out, pruned_outs[5][key], atol=1e-3)
    assert t.allclose(clean_out, pruned_outs[len(task.model.edges)][key], atol=1e-3)
