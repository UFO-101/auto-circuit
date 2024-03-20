#%%

import torch as t

from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.tensor_ops import (
    batch_answer_diff_percents,
    batch_avg_answer_diff,
    batch_avg_answer_val,
    correct_answer_greater_than_incorrect_proportion,
    correct_answer_proportion,
)


def test_batch_avg_answer_val():
    """
    Tests batch_avg_answer_val, which calculates the average value of the correct
    answer's logits.
    """
    logits = t.tensor([[0.75, 0.2, 0.7], [0.3, 0.25, 0.5]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[2], [2]]),  # Not used in this test
    )
    assert batch_avg_answer_val(logits, batch).item() == 0.5
    list_batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([2]), t.tensor([2])],  # Not used in this test
    )
    assert batch_avg_answer_val(logits, list_batch).item() == 0.5


def test_batch_avg_answer_diff():
    """
    Tests batch_avg_answer_diff, which calculates the average difference between the
    correct and wrong answers' logits.
    """
    logits = t.tensor([[0.75, 0.2, 0.75], [0.3, 1.25, 0.25]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[2], [2]]),  # Used in this test!
    )
    # Average of (0.75 - 0.75) and (1.25 - 0.25) is 0.5
    assert batch_avg_answer_diff(logits, batch).item() == 0.5
    list_batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([2]), t.tensor([2])],  # Used in this test!
    )
    assert batch_avg_answer_diff(logits, list_batch).item() == 0.5


def test_correct_answer_proportion():
    correct_logits = t.tensor([[1.0, 0.2, 0.7], [0.3, 0.9, 0.5]])
    half_correct_logits = t.tensor([[0.5, 0.2, 0.7], [0.3, 0.6, 0.5]])
    incorrect_logits = t.tensor([[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[2], [2]]),  # Not used in this test
    )
    assert correct_answer_proportion(correct_logits, batch).item() == 1.0
    assert correct_answer_proportion(half_correct_logits, batch).item() == 0.5
    assert correct_answer_proportion(incorrect_logits, batch).item() == 0.0
    list_prompt = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([2]), t.tensor([2])],  # Not used in this test
    )
    assert correct_answer_proportion(correct_logits, list_prompt).item() == 1.0
    assert correct_answer_proportion(half_correct_logits, list_prompt).item() == 0.5
    assert correct_answer_proportion(incorrect_logits, list_prompt).item() == 0.0


def test_correct_greater_than_incorrect_proportion():
    """
    Tests correct_answer_greater_than_incorrect_proportion, which just checks if the
    correct answer has a higher value than all the wrong answers.
    """
    correct_logits = t.tensor([[9.0, 2.0, 7.0, 1.0], [3.0, 9.0, 5.0, 1.0]])
    correct_vs_wrong_ans_only_logits = t.tensor(
        [[6.0, 2.0, 7.0, 1.0], [3.0, 4.0, 5.0, 1.0]]
    )
    half_correct_logits = t.tensor([[5.0, 7.0, 7.0, 1.0], [3.0, 6.0, 5.0, 1.0]])
    half_correct_vs_wrong_ans_only_logits = t.tensor(
        [[5.0, 7.0, 7.0, 1.0], [3.0, 6.0, 9.0, 1.0]]
    )
    incorrect_logits = t.tensor([[1.0, 2.0, 7.0, 1.0], [3.0, 2.0, 5.0, 1.0]])
    batch = PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [1]]),
        wrong_answers=t.tensor([[1, 3], [0, 3]]),  # Used in this test!
    )

    # Create aliases for the functions with shorter names to improve readability
    corr_greater_prop = correct_answer_greater_than_incorrect_proportion
    corr_prop = correct_answer_proportion

    assert corr_greater_prop(correct_logits, batch).item() == 1.0
    assert corr_prop(correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(correct_vs_wrong_ans_only_logits, batch).item() == 1.0
    assert corr_greater_prop(half_correct_logits, batch).item() == 0.5
    assert corr_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.5
    assert corr_greater_prop(incorrect_logits, batch).item() == 0.0
    PromptPairBatch(
        key=12,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=[t.tensor([0]), t.tensor([1])],
        wrong_answers=[t.tensor([1, 3]), t.tensor([0, 3])],  # Used in this test!
    )
    assert corr_greater_prop(correct_logits, batch).item() == 1.0
    assert corr_prop(correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(correct_vs_wrong_ans_only_logits, batch).item() == 1.0
    assert corr_greater_prop(half_correct_logits, batch).item() == 0.5
    assert corr_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.0
    assert corr_greater_prop(half_correct_vs_wrong_ans_only_logits, batch).item() == 0.5
    assert corr_greater_prop(incorrect_logits, batch).item() == 0.0


def test_batch_answer_diff_percent():
    pred_vals: t.Tensor = t.tensor([[0.3, 0.7], [0.6, 0.4]])
    target_vals: t.Tensor = t.tensor([[0.2, 0.8], [0.5, 0.3]])
    batch: PromptPairBatch = PromptPairBatch(
        key=1,
        batch_diverge_idx=0,
        clean=t.tensor(0),
        corrupt=t.tensor(0),
        answers=t.tensor([[0], [0]]),
        wrong_answers=t.tensor([[1], [1]]),
    )
    logit_diff_percents = batch_answer_diff_percents(pred_vals, target_vals, batch)

    true_pred_logit_diffs = pred_vals[:, 0] - pred_vals[:, 1]
    true_target_logit_diffs = target_vals[:, 0] - target_vals[:, 1]
    true_logit_diff_percents = (true_pred_logit_diffs / true_target_logit_diffs) * 100
    assert t.allclose(logit_diff_percents, true_logit_diff_percents)


# test_batch_avg_answer_val()
# test_batch_avg_answer_diff()
# test_correct_answer_proportion()
# test_correct_vs_incorrect_answer_proportion()
# test_batch_answer_diff_percent()
