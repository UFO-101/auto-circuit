#%%
import os
from typing import Any, Dict

import pytest
import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune_functions.ACDC import acdc_prune_scores

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.fixture
def setup_data() -> Dict[str, Any]:
    device = "cpu"
    model = MicroModel(n_layers=2)
    return {"model": model, "device": device}


def test_acdc(setup_data: Dict[str, Any]):
    model, device = setup_data["model"], setup_data["device"]
    factorized = True
    data_file = "datasets/micro_model_inputs.json"
    data_path = os.path.join(os.getcwd(), data_file)

    train_loader, _ = load_datasets_from_json(
        None,
        data_path,
        device=device,
        prepend_bos=True,
        batch_size=1,
        train_test_split=[1, 1],
        length_limit=2,
    )
    test_input = next(iter(train_loader))

    with t.inference_mode():
        clean_out = model(test_input.clean)
        corrupt_out = model(test_input.corrupt)

    assert t.allclose(clean_out, t.tensor([[25.0, 49.0]]))
    assert t.allclose(corrupt_out, t.tensor([[-25.0, -49.0]]))
    prune_scores = acdc_prune_scores(
        model, factorized, train_loader, (1e-4, 1e-4), 0.1, slice(None)
    )
    # draw_graph(model, factorized, test_input.clean, edge_label_override=prune_scores)
    print("prune_scores", prune_scores)


# data = setup_data()
# test_acdc(data)


# %%
