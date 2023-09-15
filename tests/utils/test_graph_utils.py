import os

import pytest
import transformer_lens as tl
from ordered_set import OrderedSet

from auto_circuit.types import EdgeCounts
from auto_circuit.utils.graph_utils import edge_counts_util, graph_edges

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def component_name(name: str) -> str:
    if name.endswith(".Q") or name.endswith(".K") or name.endswith(".V"):
        return name[:-2]
    return name


@pytest.mark.parametrize("reverse", [True, False])
def test_topo_sort(mini_tl_transformer: tl.HookedTransformer, reverse: bool):
    model = mini_tl_transformer
    edges = graph_edges(model, factorized=True, reverse_topo_sort=reverse)
    init_node = edges[0].dest.name if reverse else edges[0].src.name
    nodes_seen: OrderedSet[str] = OrderedSet([component_name(init_node)])
    for edge in edges:
        check_node = edge.dest.name if reverse else edge.src.name
        assert component_name(check_node) in nodes_seen
        nodes_seen.add(component_name(edge.dest.name))
        nodes_seen.add(component_name(edge.src.name))


def test_edge_counts_util(mini_tl_transformer: tl.HookedTransformer):
    model = mini_tl_transformer
    factorized = True
    n_model_edges = len(graph_edges(model, factorized))

    none_and_all = [0.0, 0.5, 1.0]
    edge_counts = edge_counts_util(model, factorized, none_and_all)
    assert edge_counts == [0, n_model_edges // 2, n_model_edges]

    edge_counts = edge_counts_util(model, factorized, EdgeCounts.ALL)
    assert edge_counts == list(range(n_model_edges + 1))
