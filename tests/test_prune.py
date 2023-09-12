#%%
import os
from typing import Any, Dict

import pytest
import torch as t
from ordered_set import OrderedSet

from auto_circuit.data import load_datasets_from_json
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune import run_pruned
from auto_circuit.types import ActType, Edge, ExperimentType
from auto_circuit.utils.graph_utils import graph_edges

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.fixture
def setup_data() -> Dict[str, Any]:
    device = "cpu"
    model = MicroModel(n_layers=2)
    return {"model": model, "device": device}


def test_pruning(setup_data: Dict[str, Any]):
    """Check that pruning works by pruning a "MicroModel"
    where the correct output can be worked out by hand.

    To visualize, set render_graph=True in run_pruned."""
    model, device = setup_data["model"], setup_data["device"]
    factorized = True
    data_file = "datasets/micro_model_inputs.json"
    data_path = os.path.join(os.getcwd(), data_file)

    experiment_type = ExperimentType(
        input_type=ActType.CLEAN, patch_type=ActType.CORRUPT
    )

    _, test_loader = load_datasets_from_json(
        None,
        data_path,
        device=device,
        prepend_bos=True,
        batch_size=1,
        train_test_split=[1, 1],
        length_limit=2,
    )
    test_input = next(iter(test_loader))

    with t.inference_mode():
        clean_out = model(test_input.clean)
        corrupt_out = model(test_input.corrupt)

    assert t.allclose(clean_out, t.tensor([[25.0, 49.0]]))
    assert t.allclose(corrupt_out, t.tensor([[-25.0, -49.0]]))
    edges: OrderedSet[Edge] = graph_edges(model, factorized)
    edge_dict = dict([(edge.name, edge) for edge in edges])

    prune_scores = {
        edge_dict["Block Layer 0 Elem 1->Output"]: 3.0,
        edge_dict["Block Layer 0 Elem 0->Block Layer 1 Elem 1"]: 2.0,
        edge_dict["Input->Block Layer 0 Elem 0"]: 0,
    }

    pruned_outs = run_pruned(
        model,
        factorized,
        test_loader,
        experiment_type,
        [1, 2, 3],
        prune_scores,
        True,
        output_idx=slice(None),
        render_graph=True,
    )
    assert t.allclose(pruned_outs[0][0], clean_out, atol=1e-3)
    assert t.allclose(pruned_outs[1][0], t.tensor([[19.0, 41.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[2][0], t.tensor([[13.0, 25.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[3][0], t.tensor([[9.0, 13.0]]), atol=1e-3)


# data = setup_data()
# test_pruning(data)

# %%
