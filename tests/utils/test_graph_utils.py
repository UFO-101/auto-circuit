import os
from typing import List, Set

import transformer_lens as tl

from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.types import Edge, EdgeCounts
from auto_circuit.utils.graph_utils import edge_counts_util, patchable_model

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def test_edge_counts_util(mini_tl_transformer: tl.HookedTransformer):
    model = patchable_model(mini_tl_transformer, factorized=True, separate_qkv=False)
    edges: Set[Edge] = model.edges
    n_model_edges = len(model.edges)

    none_and_all = [0.0, 0.5, 1.0]
    edge_counts = edge_counts_util(edges, none_and_all)
    assert edge_counts == [0, n_model_edges // 2, n_model_edges]

    edge_counts = edge_counts_util(edges, EdgeCounts.ALL)
    assert edge_counts == list(range(n_model_edges + 1))


def test_groups_edge_counts(micro_model: MicroModel):
    model = patchable_model(micro_model, factorized=True, separate_qkv=True)
    edges: Set[Edge] = model.edges
    edge_list = list(edges)

    counts: List[int] = edge_counts_util(edges, EdgeCounts.GROUPS, {}, True, True)
    assert counts == [0, len(edges)]

    counts: List[int] = edge_counts_util(edges, EdgeCounts.GROUPS, {}, False, False)
    assert counts == []

    prune_scores = {edge_list[0]: 0.0}
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [1]

    prune_scores = {edge_list[0]: 1.0, edge_list[1]: 1.0}
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [2]

    prune_scores = {edge_list[0]: 1.0, edge_list[1]: 1.0, edge_list[2]: 2.0}
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [1, 3]

    prune_scores = {edge_list[0]: 1.0, edge_list[1]: 2.0, edge_list[2]: 2.0}
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [2, 3]
