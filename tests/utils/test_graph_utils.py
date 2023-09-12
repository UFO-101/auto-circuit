import os
from typing import Any, Dict

import pytest
from ordered_set import OrderedSet

from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.utils.graph_utils import graph_edges

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.fixture
def setup_data() -> Dict[str, Any]:
    device = "cpu"
    model = MicroModel(n_layers=2)
    return {"model": model, "device": device}


def test_reverse_topo_sort(setup_data: Dict[str, Any]):
    model, _ = setup_data["model"], setup_data["device"]
    edges = graph_edges(model, factorized=True, reverse_topo_sort=True)
    nodes_seen: OrderedSet[str] = OrderedSet([edges[0].dest.name])
    for edge in edges:
        assert edge.dest.name in nodes_seen
        nodes_seen.add(edge.dest.name)
        nodes_seen.add(edge.src.name)
