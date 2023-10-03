import os

import transformer_lens as tl

from auto_circuit.types import EdgeCounts
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def test_edge_counts_util(mini_tl_transformer: tl.HookedTransformer):
    model = mini_tl_transformer
    prepare_model(model, factorized=True, device="cpu")
    n_model_edges = len(model.edges)  # type: ignore

    none_and_all = [0.0, 0.5, 1.0]
    edge_counts = edge_counts_util(model, none_and_all)
    assert edge_counts == [0, n_model_edges // 2, n_model_edges]

    edge_counts = edge_counts_util(model, EdgeCounts.ALL)
    assert edge_counts == list(range(n_model_edges + 1))
