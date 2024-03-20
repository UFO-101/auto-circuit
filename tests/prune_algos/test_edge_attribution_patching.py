#%%

import torch as t
from transformer_lens import HookedTransformer

from auto_circuit.prune_algos.edge_attribution_patching import (
    edge_attribution_patching_prune_scores,
)
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.tasks import Task


def test_eap(
    mini_tl_transformer: HookedTransformer,
    dataset_name: str = "mini_prompts",
):
    task = Task(
        key="test_eap",
        name="test_eap",
        batch_size=1,
        batch_count=1,
        token_circuit=False,
        _model_def=mini_tl_transformer,
        _dataset_name=dataset_name,
    )
    eap_ps = edge_attribution_patching_prune_scores(
        model=task.model,
        dataloader=task.train_loader,
        official_edges=task.true_edges,
        answer_diff=True,
    )
    simple_grad_ps = mask_gradient_prune_scores(
        model=task.model,
        dataloader=task.train_loader,
        official_edges=task.true_edges,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )
    for mod_name, patch_mask in eap_ps.items():
        assert t.allclose(patch_mask, simple_grad_ps[mod_name], atol=1e-5)


# model = mini_tl_transformer()
# dataset_name = "mini_prompts"
# test_eap(model, dataset_name)
