import math

import torch as t

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import PruneScores

# Copied from Subnetwork Probing paper: https://github.com/stevenxcao/subnetwork-probing
left, right, temp = -0.1, 1.1, 2 / 3


def sample_hard_concrete(mask: t.Tensor, batch_size: int) -> t.Tensor:
    mask = mask.repeat(batch_size, *([1] * mask.ndim))
    u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
    s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
    s_bar = s * (right - left) + left
    return s_bar.clamp(min=0.0, max=1.0)


def vocab_avg_val(vals: t.Tensor, indices: t.Tensor) -> t.Tensor:
    assert vals.ndim == indices.ndim
    return t.gather(vals, dim=-1, index=indices).mean()


def batch_avg_answer_val(
    vals: t.Tensor, batch: PromptPairBatch, wrong_answer: bool = False
) -> t.Tensor:
    answers = batch.answers if not wrong_answer else batch.wrong_answers
    if isinstance(answers, t.Tensor):
        return vocab_avg_val(vals, answers)
    else:
        # If each prompt has a different number of answers we have a list of tensor
        assert isinstance(answers, list)
        return t.stack([vocab_avg_val(v, a) for v, a in zip(vals, answers)]).mean()


def batch_avg_answer_diff(vals: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        # We don't use vocab_avg_val here because we need to calculate the average
        # difference between the correct and wrong answers not the difference between
        # the average correct and average incorrect answers
        ans_avgs = t.gather(vals, dim=-1, index=answers).mean(dim=-1)
        wrong_avgs = t.gather(vals, dim=-1, index=wrong_answers).mean(dim=-1)
        return (ans_avgs - wrong_avgs).mean()
    else:
        # If each prompt has a different number of answers we have a list of tensors
        assert isinstance(answers, list) and isinstance(wrong_answers, list)
        ans_avgs = [vocab_avg_val(v, a) for v, a in zip(vals, answers)]
        wrong_avgs = [vocab_avg_val(v, w) for v, w in zip(vals, wrong_answers)]
        return (t.stack(ans_avgs) - t.stack(wrong_avgs)).mean()


def correct_answer_proportion(logits: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    """
    What proportion of the logits have the correct answer as the maximum?
    """
    answers = batch.answers
    if isinstance(answers, t.Tensor):
        assert answers.shape[-1] == 1
        max_idxs = t.argmax(logits, dim=-1, keepdim=True)
        return (max_idxs == answers).float().mean()
    else:
        # If each prompt has a different number of answers we have a list of tensors
        assert isinstance(answers, list)
        corrects = []
        for prompt_idx, prompt_answer in enumerate(answers):
            assert prompt_answer.shape == (1,)
            corrects.append((t.argmax(logits[prompt_idx], dim=-1) == prompt_answer))
        return t.stack(corrects).float().mean()


def correct_answer_greater_than_incorrect_proportion(
    logits: t.Tensor, batch: PromptPairBatch
) -> t.Tensor:
    """
    What proportion of the logits have the correct answer with a greater value than all
    the wrong answers?
    """
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        assert answers.shape[-1] == 1
        answer_logits = t.gather(logits, dim=-1, index=answers)
        wrong_logits = t.gather(logits, dim=-1, index=wrong_answers)
        combined_logits = t.cat([answer_logits, wrong_logits], dim=-1)
        max_idxs = combined_logits.argmax(dim=-1)
        return (max_idxs == 0).float().mean()
    else:
        assert isinstance(answers, list) and isinstance(wrong_answers, list)
        corrects = []
        for i, (prompt_ans, prompt_wrong_ans) in enumerate(zip(answers, wrong_answers)):
            assert prompt_ans.shape == (1,)
            answer_logits = t.gather(logits[i], dim=-1, index=prompt_ans)
            wrong_logits = t.gather(logits[i], dim=-1, index=prompt_wrong_ans)
            combined_logits = t.cat([answer_logits, wrong_logits], dim=-1)
            max_idxs = combined_logits.argmax(dim=-1)
            corrects.append(max_idxs == 0)
        return t.stack(corrects).float().mean()


def multibatch_kl_div(input_logprobs: t.Tensor, target_logprobs: t.Tensor) -> t.Tensor:
    """
    Compute the average KL divergence between two sets of log probabilities.
    Assumes the last dimension is the log probability of each class.
    The other dimensions are batch dimensions.
    """
    assert input_logprobs.shape == target_logprobs.shape
    kl_div_sum = t.nn.functional.kl_div(
        input_logprobs,
        target_logprobs,
        reduction="sum",
        log_target=True,
    )
    n_batch = math.prod(input_logprobs.shape[:-1])
    return kl_div_sum / n_batch


def flat_prune_scores(prune_scores: PruneScores) -> t.Tensor:
    return t.cat([ps.flatten() for _, ps in prune_scores.items()])


def desc_prune_scores(prune_scores: PruneScores) -> t.Tensor:
    return flat_prune_scores(prune_scores).abs().sort(descending=True).values


def prune_scores_threshold(
    prune_scores: PruneScores | t.Tensor, edge_count: int
) -> t.Tensor:
    """
    Return the minimum absolute value of the top `edge_count` prune scores.
    Supports passing in a pre-sorted tensor of prune scores to avoid re-sorting.
    """
    if edge_count == 0:
        return t.tensor(float("inf"))  # return the maximum value so no edges are pruned

    if isinstance(prune_scores, t.Tensor):
        assert prune_scores.ndim == 1
        return prune_scores[edge_count - 1]
    else:
        return desc_prune_scores(prune_scores)[edge_count - 1]
