from copy import deepcopy
from enum import Enum
from typing import Literal, Optional, Tuple

import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
    ioi_head_based_official_edges,
    ioi_true_edges,
    ioi_true_edges_mlp_0_only,
)
from auto_circuit.metrics.prune_metrics.answer_diff_percent import answer_diff_percent
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.circuit_probing import circuit_probing_prune_scores
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    PatchType,
    PruneScores,
)
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path


def load_tl_model(name: str, device: t.device) -> tl.HookedTransformer:
    tl_model = tl.HookedTransformer.from_pretrained(
        name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_attn_in = True
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.eval()
    for param in tl_model.parameters():
        param.requires_grad = False
    return tl_model


class IOI_CIRCUIT_TYPE(Enum):
    NODES = 1
    EDGES = 2
    EDGES_MLP_0_ONLY = 3

    def __str__(self) -> str:
        return self.name.split(".")[-1].title()


def ioi_circuit_single_template_logit_diff_percent(
    gpt2: tl.HookedTransformer,
    dataset_size: int,
    prepend_bos: bool,
    template: Literal["ABBA", "BABA"],
    template_idx: int,
    factorized: bool = False,
    circuit: IOI_CIRCUIT_TYPE = IOI_CIRCUIT_TYPE.NODES,
    ablation_type: AblationType = AblationType.TOKENWISE_MEAN_CORRUPT,
    tok_pos: bool = True,
    patch_type: PatchType = PatchType.TREE_PATCH,
    learned: bool = False,
    diff_of_mean_logit_diff: bool = False,
    batch_size: Optional[int] = None,
) -> Tuple[int, float, float, t.Tensor, PruneScores]:
    """
    Run a single template through the IOI circuit and return the logit diff percent.

    Returns:
        Tuple[int, float, float, PruneScores]: The number of edges in the circuit, the
            mean logit diff percent, the standard deviation of the logit diff percent,
            and the prune scores of the circuit.
    """
    assert gpt2.cfg.model_name == "gpt2"
    assert gpt2.cfg.device is not None
    if type(circuit) == str and "Edges" in circuit:
        assert factorized
    if batch_size is None:
        batch_size = dataset_size

    path = repo_path_to_abs_path(
        f"datasets/ioi/ioi_{template}_template_{template_idx}_prompts.json"
    )
    patchable_gpt2 = deepcopy(gpt2)
    train_loader, test_loader = load_datasets_from_json(
        model=patchable_gpt2,
        path=path,
        device=t.device(gpt2.cfg.device),
        prepend_bos=prepend_bos,
        batch_size=batch_size,
        train_test_size=(8 * dataset_size, dataset_size)
        if learned
        else (0, dataset_size),
        shuffle=False,
        return_seq_length=tok_pos,
        tail_divergence=False,
    )

    patchable_gpt2 = patchable_model(
        model=patchable_gpt2,
        factorized=factorized,
        slice_output="last_seq",
        seq_len=test_loader.seq_len if tok_pos else None,
        separate_qkv=True,
        device=t.device(gpt2.cfg.device),
    )

    assert test_loader.word_idxs is not None
    if circuit == IOI_CIRCUIT_TYPE.EDGES:
        official_circ = ioi_true_edges
    elif circuit == IOI_CIRCUIT_TYPE.EDGES_MLP_0_ONLY:
        official_circ = ioi_true_edges_mlp_0_only
    else:
        assert circuit == IOI_CIRCUIT_TYPE.NODES
        official_circ = ioi_head_based_official_edges

    ioi_official_edges = official_circ(
        patchable_gpt2,
        word_idxs=test_loader.word_idxs,
        token_positions=tok_pos,
        seq_start_idx=test_loader.diverge_idx,
    )
    test_edge_counts = len(ioi_official_edges)

    if learned:
        circuit_ps: PruneScores = circuit_probing_prune_scores(
            model=patchable_gpt2,
            dataloader=train_loader,
            official_edges=ioi_official_edges,
            epochs=1000,
            learning_rate=0.1,
            regularize_lambda=0.1,
            mask_fn="hard_concrete",
            show_train_graph=True,
            tree_optimisation=True,
            circuit_sizes=["true_size"],
            faithfulness_target="logit_diff_percent",
            validation_dataloader=test_loader,
        )
    else:
        circuit_ps = patchable_gpt2.circuit_prune_scores(ioi_official_edges)

    circ_outs: CircuitOutputs = run_circuits(
        model=patchable_gpt2,
        dataloader=test_loader,
        test_edge_counts=[test_edge_counts],
        prune_scores=circuit_ps,
        patch_type=patch_type,
        ablation_type=ablation_type,
        render_graph=False,
        render_all_edges=False,
    )
    (
        logit_diff_percent_mean,
        logit_diff_percent_std,
        logit_diff_percents,
    ) = answer_diff_percent(
        patchable_gpt2,
        test_loader,
        circ_outs,
        prob_func="logits",
        diff_of_means=diff_of_mean_logit_diff,
    )

    assert len(logit_diff_percent_mean) == 1 and len(logit_diff_percent_std) == 1
    assert type(logit_diff_percent_mean[0][0]) == int
    assert type(logit_diff_percent_mean[0][1]) == float
    assert type(logit_diff_percent_std[0][0]) == int
    assert type(logit_diff_percent_std[0][1]) == float
    assert type(logit_diff_percents[0][0]) == int
    assert type(logit_diff_percents[0][1]) == t.Tensor
    assert (
        logit_diff_percent_mean[0][0]
        == logit_diff_percent_std[0][0]
        == logit_diff_percents[0][0]
    )

    del patchable_gpt2
    return (
        logit_diff_percent_mean[0][0],
        logit_diff_percent_mean[0][1],
        logit_diff_percent_std[0][1],
        logit_diff_percents[0][1],
        circuit_ps,
    )
