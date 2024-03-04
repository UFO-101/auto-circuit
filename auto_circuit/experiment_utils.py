from copy import deepcopy
from typing import Literal

import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
    ioi_head_based_official_edges,
)
from auto_circuit.metrics.prune_metrics.answer_diff_percent import answer_diff_percent
from auto_circuit.prune import run_circuits
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    Measurements,
    PatchType,
    PruneScores,
)
from auto_circuit.utils.graph_utils import edge_counts_util, patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path


def ioi_node_circuit_single_template_logit_diff_percent(
    gpt2: tl.HookedTransformer,
    test_batch_size: int,
    prepend_bos: bool,
    template: Literal["ABBA", "BABA"],
    template_idx: int,
    factorized: bool = False,
) -> float:
    assert gpt2.cfg.device is not None
    path = repo_path_to_abs_path(
        f"datasets/ioi/ioi_{template}_template_{template_idx}_prompts.json"
    )
    patchable_gpt2 = deepcopy(gpt2)
    _, test_loader = load_datasets_from_json(
        model=patchable_gpt2,
        path=path,
        device=t.device(gpt2.cfg.device),
        prepend_bos=prepend_bos,
        batch_size=test_batch_size,
        train_test_size=(0, test_batch_size),
        shuffle=False,
        return_seq_length=True,
        tail_divergence=False,
    )

    patchable_gpt2 = patchable_model(
        model=patchable_gpt2,
        factorized=factorized,
        slice_output="last_seq",
        seq_len=test_loader.seq_len,
        separate_qkv=False,
        device=t.device(gpt2.cfg.device),
    )

    assert test_loader.word_idxs is not None
    ioi_node_edges = ioi_head_based_official_edges(
        patchable_gpt2,
        word_idxs=test_loader.word_idxs,
        token_positions=True,
        seq_start_idx=test_loader.diverge_idx - int(prepend_bos),
    )
    circuit_ps: PruneScores = patchable_gpt2.circuit_prune_scores(ioi_node_edges)

    circ_outs: CircuitOutputs = run_circuits(
        model=patchable_gpt2,
        dataloader=test_loader,
        test_edge_counts=edge_counts_util(
            patchable_gpt2.edges, prune_scores=circuit_ps
        ),
        prune_scores=circuit_ps,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
    )
    logit_diff_percents: Measurements = answer_diff_percent(
        patchable_gpt2, test_loader, circ_outs
    )
    assert len(logit_diff_percents) == 1
    del patchable_gpt2
    return logit_diff_percents[0][1]
