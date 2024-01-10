#%%
import os
from typing import Optional

import pytest
import torch as t

from auto_circuit.metrics.answer_value import measure_answer_val
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.tasks import Task

os.environ["TOKENIZERS_PARALLELISM"] = "False"


@pytest.mark.parametrize(
    "multiple_answers, batch_count, batch_size, seq_len",
    [
        (False, 1, 1, None),
        (False, 1, 2, 3),
        (False, 3, 1, None),
        (False, 4, 4, 3),
        (True, 1, 1, None),
        (True, 1, 3, 3),
        (True, 2, 1, None),
        (True, 5, 7, 3),
    ],
)
def test_answer_prob(
    micro_model: MicroModel,
    multiple_answers: bool,
    batch_count: int,
    batch_size: int,
    seq_len: Optional[int],
):
    """Check the measure_answer_prob metric works by passing simple pruned_outs."""
    dataset = f"micro_model_inputs{'_multiple_answers' if multiple_answers else ''}"
    task = Task(
        key="test_answer_prob",
        name="test_answer_prob",
        batch_size=batch_size,
        batch_count=batch_count,
        token_circuit=seq_len is not None,
        _model_def=micro_model,
        _dataset_name=dataset,
    )
    model = task.model

    pruned_out = [model(batch.clean)[model.out_slice] for batch in task.test_loader]

    answer_prob = measure_answer_val(
        task,
        prune_scores=None,
        pruned_outs={0: pruned_out},
        prob_func="logits",
    )
    pruned_out = t.stack(pruned_out)
    for batch_idx, batch in enumerate(task.test_loader):
        avg_ans_prob = []
        for prompt_idx, prompt_answers in enumerate(batch.answers):
            probs = [
                pruned_out[batch_idx, prompt_idx, a].item() for a in prompt_answers
            ]
            avg_ans_prob.append(sum(probs) / len(probs))
        assert avg_ans_prob is not None
        assert answer_prob[0][1] == pytest.approx(sum(avg_ans_prob) / len(avg_ans_prob))


@pytest.mark.parametrize("seq_len", [None, 3])
def test_greaterthan_answer_prob(
    mini_tl_transformer: t.nn.Module,
    seq_len: Optional[int],
):
    """Check the measure_answer_prob metric works by passing simple pruned_outs."""
    task = Task(
        key="test_greaterthan_answer_prob",
        name="test_greaterthan_answer_prob",
        batch_size=200,
        batch_count=1,
        token_circuit=seq_len is not None,
        _model_def=mini_tl_transformer,
        _dataset_name="greaterthan_gpt2-small_prompts",
    )
    model = task.model
    pruned_out = [model(batch.clean)[model.out_slice] for batch in task.test_loader]

    answer_prob = measure_answer_val(
        task,
        prune_scores=None,
        pruned_outs={0: pruned_out},
        prob_func="logits",
    )
    pruned_out = t.stack(pruned_out)
    for batch_idx, batch in enumerate(task.test_loader):
        avg_ans_probs = []
        for prompt_idx, prompt_answers in enumerate(batch.answers):
            probs = [
                pruned_out[batch_idx, prompt_idx, a].item() for a in prompt_answers
            ]
            avg_ans_probs.append(sum(probs) / len(probs))
        assert avg_ans_probs is not None
        avg_ans_prob = sum(avg_ans_probs) / len(avg_ans_probs)
        assert answer_prob[0][1] == pytest.approx(avg_ans_prob, abs=1e-4)


# micro_model = micro_model()
# test_answer_prob(micro_model, False, 1, 1, 4)
# test_greater_than_answer_prob(
#     mini_tl_transformer(), seq_len=None
# )
# test_greater_than_answer_prob(
#     mini_tl_transformer(),
#     seq_len=None,
# )
# model = mini_tl_transformer()
# test_greaterthan_answer_prob(model, seq_len=None)
# test_greaterthan_answer_prob(model, seq_len=3)
# print(model.tokenizer)
