#%%
import os
from typing import Optional

import pytest
import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import (
    PromptPairBatch,
)
from auto_circuit.metrics.answer_prob import measure_answer_prob
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.utils.graph_utils import prepare_model

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.mark.parametrize(
    "micro_dataloader, seq_len",
    [
        ({"multiple_answers": False, "batch_count": 1, "batch_size": 1}, None),
        ({"multiple_answers": False, "batch_count": 1, "batch_size": 2}, 3),
        ({"multiple_answers": False, "batch_count": 3, "batch_size": 1}, None),
        ({"multiple_answers": False, "batch_count": 4, "batch_size": 4}, 3),
        ({"multiple_answers": True, "batch_count": 1, "batch_size": 1}, None),
        ({"multiple_answers": True, "batch_count": 1, "batch_size": 3}, 3),
        ({"multiple_answers": True, "batch_count": 2, "batch_size": 1}, None),
        ({"multiple_answers": True, "batch_count": 5, "batch_size": 7}, 3),
    ],
    indirect=["micro_dataloader"],
)
def test_answer_prob(
    micro_model: MicroModel,
    micro_dataloader: DataLoader[PromptPairBatch],
    seq_len: Optional[int],
):
    """Check the measure_answer_prob metric works by passing simple pruned_outs."""
    prepare_model(micro_model, factorized=True, seq_len=seq_len, slice_output=True)
    pruned_out = [
        micro_model(batch.clean)[micro_model.out_slice] for batch in micro_dataloader
    ]

    answer_probs = measure_answer_prob(
        model=micro_model,
        test_loader=micro_dataloader,
        pruned_outs={0: pruned_out},
        prob_func=None,
    )
    pruned_out = t.stack(pruned_out)
    for batch_idx, batch in enumerate(micro_dataloader):
        avg_ans_prob = []
        for prompt_idx, prompt_answers in enumerate(batch.answers):
            probs = [
                pruned_out[batch_idx, prompt_idx, a].item() for a in prompt_answers
            ]
            avg_ans_prob.append(sum(probs) / len(probs))
        assert avg_ans_prob is not None
        assert answer_probs[0] == pytest.approx(sum(avg_ans_prob) / len(avg_ans_prob))


@pytest.mark.parametrize("seq_len", [None, 3])
def test_greater_than_answer_prob(
    mini_tl_transformer: t.nn.Module,
    greater_than_gpt2_dataloader: DataLoader[PromptPairBatch],
    seq_len: Optional[int],
):
    """Check the measure_answer_prob metric works by passing simple pruned_outs."""
    model, dataloader = mini_tl_transformer, greater_than_gpt2_dataloader
    prepare_model(model, factorized=True, seq_len=seq_len, slice_output=True)
    pruned_out = [model(batch.clean)[model.out_slice] for batch in dataloader]

    answer_probs = measure_answer_prob(
        model=model,
        test_loader=dataloader,
        pruned_outs={0: pruned_out},
        prob_func=None,
    )
    pruned_out = t.stack(pruned_out)
    for batch_idx, batch in enumerate(dataloader):
        avg_ans_probs = []
        for prompt_idx, prompt_answers in enumerate(batch.answers):
            probs = [
                pruned_out[batch_idx, prompt_idx, a].item() for a in prompt_answers
            ]
            avg_ans_probs.append(sum(probs) / len(probs))
        assert avg_ans_probs is not None
        avg_ans_prob = sum(avg_ans_probs) / len(avg_ans_probs)
        assert answer_probs[0] == pytest.approx(avg_ans_prob, abs=1e-4)


# micro_model = micro_model()
# dataloader = micro_dataloader(multiple_answers=True, batch_count=2, batch_size=1)
# test_answer_prob(micro_model, dataloader, seq_len=3)
# test_greater_than_answer_prob(seq_len=None)
