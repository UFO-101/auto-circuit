#%%
from typing import Any

import pytest
import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.prune_functions.ACDC import acdc_prune_scores
from auto_circuit.utils.graph_utils import prepare_model


@pytest.mark.parametrize(
    "model, dataloader",
    [
        ("micro_model", "micro_dataloader"),
        ("mini_tl_transformer", "mini_tl_dataloader"),
    ],
)
def test_acdc(
    model: t.nn.Module,
    dataloader: DataLoader[PromptPairBatch],
    request: Any,
    show_graphs: bool = False,  # Useful for debugging
):
    fixture_model = request.getfixturevalue(model) if request else model
    prepare_model(fixture_model, factorized=True, slice_output=True, device="cpu")
    fixture_dataloader = request.getfixturevalue(dataloader) if request else dataloader
    acdc_prune_scores(
        model=fixture_model,
        train_data=fixture_dataloader,
        tao_exps=[-3],
        tao_bases=[1],
        test_mode=True,  # The actual test logic is embedded in the function
        show_graphs=show_graphs,
    )


# model = micro_model()
# model = mini_tl_transformer()
# dataloader = micro_dataloader()
# dataloader = mini_tl_dataloader()
# test_acdc(model, dataloader, request=None, show_graphs=True)
