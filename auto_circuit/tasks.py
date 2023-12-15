from typing import Callable, Set

import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.official_circuits.greaterthan_official import (
    greaterthan_true_edges,
)
from auto_circuit.metrics.official_circuits.ioi_official import ioi_true_edges
from auto_circuit.types import Edge, Task
from auto_circuit.utils.graph_utils import prepare_model
from auto_circuit.utils.misc import repo_path_to_abs_path


def transformer_lens_experiment(
    name: str,
    model_name: str,
    dataset: str,
    batch_size: int,
    true_edge_func: Callable[..., Set[Edge]],
    token_circuit: bool = False,
) -> Task:
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    assert model.tokenizer is not None
    model.eval()

    train_loader, test_loader = load_datasets_from_json(
        tokenizer=model.tokenizer,
        path=repo_path_to_abs_path(f"datasets/{dataset}.json"),
        device=device,
        prepend_bos=True,
        batch_size=batch_size,
        train_test_split=[0.8, 0.2],
        length_limit=640,
        return_seq_length=token_circuit,
        pad=True,
    )
    seq_len = train_loader.seq_len
    prepare_model(
        model, factorized=True, slice_output=True, seq_len=seq_len, device=device
    )

    return Task(
        name,
        model,
        train_loader,
        test_loader,
        true_edge_func,
        token_circuit,
    )


IOI_TOKEN_CIRCUIT_TASK: Task = transformer_lens_experiment(
    name="Indirect Object Identification",
    model_name="gpt2-small",
    dataset="ioi_single_template_prompts",
    batch_size=64,
    true_edge_func=ioi_true_edges,
    token_circuit=True,
)
IOI_COMPONENT_CIRCUIT_TASK: Task = transformer_lens_experiment(
    name="Indirect Object Identification",
    model_name="gpt2-small",
    dataset="ioi_prompts",
    batch_size=64,
    true_edge_func=ioi_true_edges,
    token_circuit=False,
)
DOCSTRING_TOKEN_CIRCUIT_TASK: Task = transformer_lens_experiment(
    name="Docstring",
    model_name="attn-only-4l",
    dataset="docstring_prompts",
    batch_size=128,
    true_edge_func=docstring_true_edges,
    token_circuit=True,
)
DOCSTRING_COMPONENT_CIRCUIT_TASK: Task = transformer_lens_experiment(
    name="Docstring",
    model_name="attn-only-4l",
    dataset="docstring_prompts",
    batch_size=128,
    true_edge_func=docstring_true_edges,
    token_circuit=False,
)
GREATERTHAN_COMPONENT_CIRCUIT_TASK: Task = transformer_lens_experiment(
    name="Greaterthan",
    model_name="gpt2-small",
    dataset="greaterthan_gpt2-small_prompts",
    batch_size=64,
    true_edge_func=greaterthan_true_edges,
    token_circuit=False,
)
