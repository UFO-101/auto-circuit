#%%

from collections import defaultdict
from typing import Dict, List

import torch as t
from transformer_lens import HookedTransformer

from auto_circuit.tasks import Task
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import (
    patch_mode,
    set_all_masks,
    set_mask_batch_size,
    train_mask_mode,
)
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff


def test_instance_grads(
    mini_tl_transformer: HookedTransformer, dataset_name: str = "mini_prompts"
):

    # create task
    batch_size = 2
    task = Task(
        key="test_eap",
        name="test_eap",
        batch_size=batch_size,
        batch_count=1,
        token_circuit=False,
        _model_def=mini_tl_transformer,
        _dataset_name=dataset_name,
    )
    model = task.model
    train_loader = task.train_loader

    # compute patch outs
    patch_outs = {}
    for batch in train_loader:
        patch_outs[batch.key] = src_ablations(
            model, batch.clean, ablation_type=AblationType.ZERO
        )

    # compute prune scores in inputs
    prune_scores: Dict[str, List[t.Tensor]] = defaultdict(list)
    with set_mask_batch_size(model, batch_size):
        with train_mask_mode(model):
            set_all_masks(model, val=0.0)

            for batch in train_loader:
                patch_src_outs = patch_outs[batch.key].clone().detach()
                with patch_mode(model, patch_src_outs):
                    logits = model(batch.clean)[model.out_slice]
                    loss = -batch_avg_answer_diff(logits, batch)
                    loss.backward()
                for dest_wrapper in model.dest_wrappers:
                    prune_scores[dest_wrapper.module_name].append(
                        dest_wrapper.patch_mask.grad.detach().clone()
                    )
                model.zero_grad()

    ex_prune_score = next(iter(prune_scores.values()))[0]
    # check expanded batch size
    assert ex_prune_score.size(0) == batch_size
    # check masks collapsed on exit
    assert next(iter(model.dest_wrappers)).patch_mask.ndim == ex_prune_score.ndim - 1
