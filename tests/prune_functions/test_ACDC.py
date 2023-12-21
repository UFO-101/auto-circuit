#%%
from typing import Any

import pytest
import torch as t

from auto_circuit.prune import run_pruned
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.tasks import Task
from auto_circuit.utils.graph_utils import prepare_model
from auto_circuit.visualize import draw_seq_graph


@pytest.mark.parametrize(
    "model, dataloader_name",
    [
        ("micro_model", "micro_model_inputs"),
        ("mini_tl_transformer", "mini_prompts"),
    ],
)
def test_acdc(
    model: t.nn.Module,
    dataloader_name: str,
    request: Any,
    show_graphs: bool = False,  # Useful for debugging
):
    fixture_model = request.getfixturevalue(model) if request else model
    prepare_model(
        fixture_model, factorized=True, slice_output=True, device=t.device("cpu")
    )
    task = Task(
        key="test_acdc",
        name="test_acdc",
        batch_size=1,
        batch_count=1,
        token_circuit=False,
        _model_def=fixture_model,
        _dataset_name=dataloader_name,
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
# test_acdc(model, dataloader, request=None, show_graphs=True)
