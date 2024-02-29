#%%
from copy import deepcopy

import pytest
import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
    ioi_head_based_official_edges,
)
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import batch_src_ablations, src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model, set_all_masks
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph
from tests.conftest import DEVICE


def test_micro_model_unfactorized_edges(micro_model: t.nn.Module, debug: bool = False):
    node_to_ablate = "B0.1"
    if debug:
        input_batch = t.tensor([[[1.0, 2.0]]], device=DEVICE)  # Simple for debugging
        patch_batch = -input_batch
        seq_len = input_batch.shape[1]
    else:
        seq_len = 3
        input_batch = t.rand([4, seq_len, 2], device=DEVICE) * 10.0
        patch_batch = t.rand([4, seq_len, 2], device=DEVICE) * 10.0

    default_out = micro_model(input_batch)

    fctrzd_model = deepcopy(micro_model)
    fctrzd_model = patchable_model(
        fctrzd_model, factorized=True, seq_len=seq_len, device=DEVICE
    )
    fctrzd_model_unpatched_out = fctrzd_model(input_batch)
    assert t.allclose(default_out, fctrzd_model_unpatched_out)

    unfctrzd_model = deepcopy(micro_model)
    unfctrzd_model = patchable_model(
        unfctrzd_model, factorized=False, seq_len=seq_len, device=DEVICE
    )
    unfctrzd_model_unpatched_out = unfctrzd_model(input_batch)
    assert t.allclose(default_out, unfctrzd_model_unpatched_out)

    fctrzd_patches = src_ablations(fctrzd_model, patch_batch, AblationType.RESAMPLE)
    set_all_masks(fctrzd_model, 0.0)
    for edge in fctrzd_model.edges:
        if edge.src.name == node_to_ablate:
            if debug:
                print("Factorized - Patching edge:", edge, "seq_idx:", edge.seq_idx)
            edge.patch_mask(fctrzd_model).data[edge.patch_idx] = 1.0
    with patch_mode(fctrzd_model, fctrzd_patches):
        fctrzd_patched_out = fctrzd_model(input_batch)
        if debug:
            draw_seq_graph(fctrzd_model, input_batch, show_all_edges=True)

    unfctrzd_patches = src_ablations(unfctrzd_model, patch_batch, AblationType.RESAMPLE)
    set_all_masks(unfctrzd_model, 0.0)
    for edge in unfctrzd_model.edges:
        if edge.src.name == node_to_ablate:
            if debug:
                print("Unfactorized - Patching edge:", edge, "seq_idx:", edge.seq_idx)
            edge.patch_mask(unfctrzd_model).data[edge.patch_idx] = 1.0
    with patch_mode(unfctrzd_model, unfctrzd_patches):
        unfctrzd_patched_out = unfctrzd_model(input_batch)
        if debug:
            draw_seq_graph(unfctrzd_model, input_batch, show_all_edges=True)

    assert not t.allclose(default_out, fctrzd_patched_out)
    assert t.allclose(fctrzd_patched_out, unfctrzd_patched_out)


def test_mini_transformer_unfactorized_edges(
    mini_tl_transformer: tl.HookedTransformer, debug: bool = False
):
    node_to_ablate = "A0.1"
    if debug:
        # Single token for debugging
        input_batch = t.tensor([[1]], dtype=t.int32, device=DEVICE)
        patch_batch = -input_batch
        seq_len = input_batch.shape[1]
    else:
        seq_len = 3
        input_batch = t.randint(0, 100, size=[4, seq_len], device=DEVICE)
        patch_batch = t.randint(0, 100, size=[4, seq_len], device=DEVICE)

    default_out = mini_tl_transformer(input_batch)

    fctrzd_model = deepcopy(mini_tl_transformer)
    fctrzd_model = patchable_model(
        fctrzd_model,
        factorized=True,
        seq_len=seq_len,
        separate_qkv=False,
        device=DEVICE,
    )
    fctrzd_model_unpatched_out = fctrzd_model(input_batch)
    assert t.allclose(default_out, fctrzd_model_unpatched_out)

    unfctrzd_model = deepcopy(mini_tl_transformer)
    unfctrzd_model = patchable_model(
        unfctrzd_model,
        factorized=False,
        seq_len=seq_len,
        separate_qkv=False,
        device=DEVICE,
    )
    unfctrzd_model_unpatched_out = unfctrzd_model(input_batch)
    assert t.allclose(default_out, unfctrzd_model_unpatched_out)

    fctrzd_patches = src_ablations(fctrzd_model, patch_batch, AblationType.RESAMPLE)
    set_all_masks(fctrzd_model, 0.0)
    for edge in fctrzd_model.edges:
        if edge.src.name == node_to_ablate:
            if debug:
                print("Factorized - Patching edge:", edge, "seq_idx:", edge.seq_idx)
            edge.patch_mask(fctrzd_model).data[edge.patch_idx] = 1.0
    with patch_mode(fctrzd_model, fctrzd_patches):
        fctrzd_patched_out = fctrzd_model(input_batch)
        if debug:
            draw_seq_graph(fctrzd_model, input_batch, show_all_edges=True)

    unfctrzd_patches = src_ablations(unfctrzd_model, patch_batch, AblationType.RESAMPLE)
    set_all_masks(unfctrzd_model, 0.0)
    for edge in unfctrzd_model.edges:
        if edge.src.name == node_to_ablate:
            if debug:
                print("Unfactorized - Patching edge:", edge, "seq_idx:", edge.seq_idx)
            edge.patch_mask(unfctrzd_model).data[edge.patch_idx] = 1.0
    with patch_mode(unfctrzd_model, unfctrzd_patches):
        unfctrzd_patched_out = unfctrzd_model(input_batch)
        if debug:
            draw_seq_graph(unfctrzd_model, input_batch, show_all_edges=True)

    assert not t.allclose(default_out, fctrzd_patched_out)
    assert t.allclose(fctrzd_patched_out, unfctrzd_patched_out, atol=1e-5)


# micro_model = micro_model()
# test_micro_model_unfactorized_edges(micro_model, debug=True)
# mini_tl_transformer = mini_tl_transformer()
# test_mini_transformer_unfactorized_edges(mini_tl_transformer, debug=True)


@pytest.mark.slow
def test_ioi_node_based_circuit_factorized_vs_unfactorized(
    gpt2: tl.HookedTransformer, debug: bool = False
):
    train_loader, test_loader = load_datasets_from_json(
        model=gpt2,
        path=repo_path_to_abs_path("datasets/ioi_single_template_prompts.json"),
        device=DEVICE,
        batch_size=1,
        train_test_split=[1, 1],
        length_limit=2,
        return_seq_length=True,
        tail_divergence=True,
    )
    seq_len = train_loader.seq_len
    kv_caches = train_loader.kv_cache, test_loader.kv_cache
    diverge_idx = train_loader.diverge_idx
    first_train_batch = next(iter(train_loader))

    default_out = gpt2(first_train_batch.clean)

    fctrzd_gpt2 = deepcopy(gpt2)
    fctrzd_gpt2 = patchable_model(
        fctrzd_gpt2,
        factorized=True,
        separate_qkv=False,
        seq_len=seq_len,
        device=DEVICE,
        kv_caches=kv_caches,
    )
    fctrzd_patches = batch_src_ablations(
        fctrzd_gpt2, train_loader, AblationType.RESAMPLE, clean_corrupt="corrupt"
    )
    edges_in_circuit = ioi_head_based_official_edges(
        fctrzd_gpt2, token_positions=True, seq_start_idx=diverge_idx
    )
    set_all_masks(fctrzd_gpt2, 1.0)
    for edge in fctrzd_gpt2.edges:
        if edge in edges_in_circuit:
            edge.patch_mask(fctrzd_gpt2).data[edge.patch_idx] = 0.0
    fctrzd_patch = fctrzd_patches[first_train_batch.key]
    with patch_mode(fctrzd_gpt2, fctrzd_patch):
        fctrzd_patched_out = fctrzd_gpt2(first_train_batch.clean)

    unfctrzd_gpt2 = deepcopy(gpt2)
    unfctrzd_gpt2 = patchable_model(
        unfctrzd_gpt2,
        factorized=False,
        separate_qkv=False,
        seq_len=seq_len,
        device=DEVICE,
        kv_caches=kv_caches,
    )
    unfctrzd_patches = batch_src_ablations(
        unfctrzd_gpt2, train_loader, AblationType.RESAMPLE, clean_corrupt="corrupt"
    )
    edges_in_circuit = ioi_head_based_official_edges(
        unfctrzd_gpt2, token_positions=True, seq_start_idx=diverge_idx
    )
    set_all_masks(unfctrzd_gpt2, 0.0)
    for edge in unfctrzd_gpt2.edges:
        # We have to iterate through all edges NOT in the circuit because unfactorized
        # models have edge masks that shouldn't be used. So we can't set_all_masks(1.0)
        if edge not in edges_in_circuit:
            edge.patch_mask(unfctrzd_gpt2).data[edge.patch_idx] = 1.0

    unfctrzd_patch = unfctrzd_patches[first_train_batch.key]
    with patch_mode(unfctrzd_gpt2, unfctrzd_patch):
        unfctrzd_patched_out = unfctrzd_gpt2(first_train_batch.clean)
        if debug:
            draw_seq_graph(
                unfctrzd_gpt2,
                first_train_batch.clean,
                show_all_edges=False,
                seq_labels=train_loader.seq_labels,
            )

    assert not t.allclose(default_out, fctrzd_patched_out)
    assert t.allclose(fctrzd_patched_out, unfctrzd_patched_out, atol=1e-5)


# model = gpt2()
# test_ioi_node_based_circuit_factorized_vs_unfactorized(model, debug=False)
