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
from auto_circuit.prune_algos.subnetwork_probing import SP_FAITHFULNESS_TARGET
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    PatchType,
    PruneScores,
)
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path


def load_tl_model(name: str, device: t.device) -> tl.HookedTransformer:
    """
    Load a `HookedTransformer` model with the necessary config to perform edge patching
    (with separate edges to Q, K, and V). Sets `requires_grad` to `False` for all model
    weights (this does not affect Mask gradients).
    """
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
    """
    Type of IOI circuit. The original IOI paper discovered important attention heads and
    interactions between them.
    """

    NODES = 1
    """`NODES` ablates the outputs of the head in the circuit."""
    EDGES = 2
    """
    `EDGES` ablates the edges identified by the IOI paper. Note that the IOI paper
    considered intermediate MLPs to be part of the direct path between two attention
    heads, so this includes many edges to or from MLPs.
    """
    EDGES_MLP_0_ONLY = 3
    """
    Therefore we also provide `EDGES_MLP_0_ONLY` which includes only the first MLP layer
    (as this seems to retain most of the performance of the full `EDGES` circuit).
    """

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
    learned_faithfulness_target: SP_FAITHFULNESS_TARGET = "logit_diff_percent",
    diff_of_mean_logit_diff: bool = False,
    batch_size: Optional[int] = None,
) -> Tuple[int, float, float, t.Tensor, PruneScores]:
    """
    Run a single template format through the IOI circuit and return the logit diff
    recovered.

    Args:
        gpt2: A GPT2 `HookedTransformer`.
        dataset_size: The size of the dataset to use.
        prepend_bos: Whether to prepend the `BOS` token to the prompts.
        template: The type of template to use. (This is the order of names).
        template_idx: The index of the template to use (`0` to `14`).
        factorized: Use a 'factorized' model (Edge Patching, not Node Patching).
        circuit: The type of circuit to use (see `IOI_CIRCUIT_TYPE`).
        ablation_type: The type of ablation to use.
        tok_pos: Whether to ablate different token positions separately.
        patch_type: The type of patch to use (ablate the circuit or the complement).
        learned: Whether to learn a new circuit using `Subnetwork Probing` (in this case
            `IOI_CIRCUIT_TYPE` is only used to determine the number of edges in the
            learned circuit).
        learned_faithfulness_target: The faithfulness target used to learn the circuit.
        learned_faithfulness_target: The faithfulness metric to optimize the learned
            circuit for.
        diff_of_mean_logit_diff: If `true` we compute:
            ```
            (mean(circuit) / mean(model)) * 100
            ```
            like the IOI paper. If `false` we compute:
            ```
            (mean(circuit / model)) * 100
            ```
        batch_size: The batch size to use.

    Returns:
        The number of edges in the circuit, the mean logit diff percent, the standard
            deviation of the logit diff percent, and the prune scores of the circuit.
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
            epochs=200,
            learning_rate=0.1,
            regularize_lambda=0.1,
            mask_fn="hard_concrete",
            show_train_graph=True,
            tree_optimisation=True,
            circuit_sizes=["true_size"],
            faithfulness_target=learned_faithfulness_target,
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
