#%%
from typing import Any

import pytest
import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.prune import run_pruned
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.types import Task
from auto_circuit.utils.graph_utils import prepare_model
from auto_circuit.visualize import draw_seq_graph


@pytest.mark.parametrize(
    "model, dataloader",
    [
        ("micro_model", "micro_dataloader"),
        ("mini_tl_transformer", "mini_tl_dataloader"),
    ],
)
def test_acdc(
    model: t.nn.Module,
    dataloader: PromptDataLoader,
    request: Any,
    show_graphs: bool = False,  # Useful for debugging
):
    fixture_model = request.getfixturevalue(model) if request else model
    prepare_model(fixture_model, factorized=True, slice_output=True, device="cpu")
    fixture_dataloader = request.getfixturevalue(dataloader) if request else dataloader
    task = Task(
        "test_acdc",
        fixture_model,
        fixture_dataloader,
        fixture_dataloader,
        lambda: set(),
    )
    acdc_prune_scores(
        task=task,
        tao_exps=[-3],
        tao_bases=[1],
        test_mode=True,  # The actual test logic is embedded in the function
        run_pruned_ref=run_pruned,
        show_graphs=show_graphs,
        draw_seq_graph_ref=draw_seq_graph,
    )


# model = micro_model()
# model = mini_tl_transformer()
# dataloader = micro_dataloader()
# dataloader = mini_tl_dataloader()
# test_acdc(model, dataloader, request=None, show_graphs=True)
