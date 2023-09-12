import os
from typing import Any, Dict

import pytest
import torch as t
import torch.backends.mps
import transformer_lens

from auto_circuit.data import load_datasets_from_json
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.types import ActType, EdgeCounts, ExperimentType
from auto_circuit.utils.graph_utils import edge_counts_util, graph_edges

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.fixture
def setup_data() -> Dict[str, Any]:
    # device = (
    #     "cuda"
    #     if t.cuda.is_available()
    #     else "mps"
    #     if True and torch.backends.mps.is_available()
    #     else "cpu"
    # )
    device = "cpu"
    cfg = transformer_lens.HookedTransformerConfig(
        d_vocab=50257,
        n_layers=2,
        d_model=4,
        n_ctx=64,
        n_heads=2,
        d_head=2,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device=device,
    )
    model = transformer_lens.HookedTransformer(cfg)
    model.init_weights()

    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    return {"model": model, "device": device}


def test_get_test_edge_counts(setup_data: Dict[str, Any]):
    model = setup_data["model"]
    factorized = True
    n_model_edges = len(graph_edges(model, factorized))

    none_and_all = [0.0, 0.5, 1.0]
    edge_counts = edge_counts_util(model, factorized, none_and_all)
    assert edge_counts == [0, n_model_edges // 2, n_model_edges]

    edge_counts = edge_counts_util(model, factorized, EdgeCounts.ALL)
    assert edge_counts == list(range(n_model_edges + 1))


def test_kl_vs_edges(setup_data: Dict[str, Any]):
    """Test that experiments satisfy basic requirements with a real model."""
    model, device = setup_data["model"], setup_data["device"]
    factorized = True
    data_file = "datasets/indirect_object_identification.json"
    data_path = os.path.join(os.getcwd(), data_file)

    experiment_type = ExperimentType(
        input_type=ActType.CLEAN, patch_type=ActType.CORRUPT
    )

    train_loader, test_loader = load_datasets_from_json(
        model.tokenizer,
        data_path,
        device=device,
        prepend_bos=True,
        batch_size=1,
        train_test_split=[1, 1],
        length_limit=2,
    )
    test_input = next(iter(test_loader))

    with t.inference_mode():
        clean_out = model(test_input.clean)[:, -1]
        corrupt_out = model(test_input.corrupt)[:, -1]

    prune_scores = random_prune_scores(model, factorized, train_loader)
    test_edge_counts = edge_counts_util(model, factorized, [0.0, 5, 1.0])
    pruned_outs = run_pruned(
        model, factorized, test_loader, experiment_type, test_edge_counts, prune_scores
    )
    assert t.allclose(clean_out, pruned_outs[0][0], atol=1e-3)
    assert not t.allclose(corrupt_out, pruned_outs[5][0], atol=1e-3)
    edges = graph_edges(model, factorized)
    assert t.allclose(corrupt_out, pruned_outs[len(edges)][0], atol=1e-3)
