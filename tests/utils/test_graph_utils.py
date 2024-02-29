#%%
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

    none_and_all = [0.0, 0.5, 1.0]
    edge_counts = edge_counts_util(edges, none_and_all)
    assert edge_counts == [0, model.n_edges // 2, model.n_edges]

    edge_counts = edge_counts_util(edges, EdgeCounts.ALL)
    assert edge_counts == list(range(model.n_edges + 1))


# model = mini_tl_transformer()
# test_edge_counts_util(model)


def test_groups_edge_counts(micro_model: MicroModel):
    model = patchable_model(micro_model, factorized=True)
    edges: Set[Edge] = model.edges
    edge_list = list(edges)
    edge_0 = edge_list[0]
    edge_1 = edge_list[1]
    edge_2 = edge_list[2]

    ps = model.new_prune_scores()
    counts: List[int] = edge_counts_util(edges, EdgeCounts.GROUPS, ps, True, True)
    assert counts == [0, len(edges)]

    ps = model.new_prune_scores()
    counts: List[int] = edge_counts_util(edges, EdgeCounts.GROUPS, ps, False, False)
    assert counts == []

    prune_scores = model.new_prune_scores()
    prune_scores[edge_0.dest.module_name][edge_0.patch_idx] = 1.0
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [1]

    prune_scores = model.new_prune_scores()
    prune_scores[edge_0.dest.module_name][edge_0.patch_idx] = 1.0
    prune_scores[edge_1.dest.module_name][edge_1.patch_idx] = 1.0
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [2]

    prune_scores = model.new_prune_scores()
    prune_scores[edge_0.dest.module_name][edge_0.patch_idx] = 1.0
    prune_scores[edge_1.dest.module_name][edge_1.patch_idx] = 1.0
    prune_scores[edge_2.dest.module_name][edge_2.patch_idx] = 2.0
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [1, 3]

    prune_scores = model.new_prune_scores()
    prune_scores[edge_0.dest.module_name][edge_0.patch_idx] = 1.0
    prune_scores[edge_1.dest.module_name][edge_1.patch_idx] = 2.0
    prune_scores[edge_2.dest.module_name][edge_2.patch_idx] = 2.0
    counts: List[int] = edge_counts_util(
        edges, EdgeCounts.GROUPS, prune_scores, False, False
    )
    assert counts == [2, 3]


# model = micro_model()
# test_groups_edge_counts(model)

# %%
