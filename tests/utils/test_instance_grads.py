#%%


import torch as t
from transformer_lens import HookedTransformer

from auto_circuit.tasks import Task
from auto_circuit.types import AblationType, PruneScores
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import (
    patch_mode,
    set_all_masks,
    set_mask_batch_size,
    train_mask_mode,
)
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff


def test_instance_grads(mini_tl_transformer: HookedTransformer):

    # create task
    batch_size = 2
    batch_count = 1
    task = Task(
        key="test_eap",
        name="test_eap",
        batch_size=batch_size,
        batch_count=batch_count,
        token_circuit=False,
        _model_def=mini_tl_transformer,
        _dataset_name="mini_prompts",
    )
    model = task.model
    train_loader = task.train_loader

    # compute src patch out
    src_patch_out = src_ablations(
        model, next(iter(train_loader)).clean, ablation_type=AblationType.ZERO
    )

    # collecting prune scores batches for each module, concatenated after
    prune_scores_batch: PruneScores = {}
    with set_mask_batch_size(model, batch_size), train_mask_mode(model):
        set_all_masks(model, val=0.0)
        for batch in train_loader:
            with patch_mode(model, src_patch_out.clone().detach()):
                # combine clean and corrupt to get different values for testing
                logits = model(t.cat([batch.clean[0:1], batch.corrupt[0:1]]))[
                    model.out_slice
                ]
                loss = -batch_avg_answer_diff(logits, batch)
                loss.backward(t.ones_like(loss))
            for dest_wrapper in model.dest_wrappers:
                assert dest_wrapper.patch_mask.size(0) == batch_size
                grad = dest_wrapper.patch_mask.grad
                assert grad is not None
                prune_scores_batch[dest_wrapper.module_name] = grad.detach().clone()
            model.zero_grad()

    ex_prune_score = next(iter(prune_scores_batch.values()))
    # check expanded batch size
    assert ex_prune_score.size(0) == batch_size
    # check gradients are not the same
    assert not t.allclose(ex_prune_score[0], ex_prune_score[1])
    # check masks collapsed on exit
    assert next(iter(model.dest_wrappers)).patch_mask.ndim == ex_prune_score.ndim - 1
