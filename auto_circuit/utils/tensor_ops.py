import math
from typing import Literal, Optional

import torch as t

from auto_circuit.data import PromptPairBatch

MaskFn = Optional[Literal["hard_concrete", "sigmoid"]]

# Copied from Subnetwork Probing paper: https://github.com/stevenxcao/subnetwork-probing
left, right, temp = -0.1, 1.1, 2 / 3


def sample_hard_concrete(mask: t.Tensor, batch_size: int) -> t.Tensor:
    mask = mask.repeat(batch_size, *([1] * mask.ndim))
    u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
    s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
    s_bar = s * (right - left) + left
    return s_bar.clamp(min=0.0, max=1.0)


def batch_avg_answer_val(
    vals: t.Tensor, batch: PromptPairBatch, wrong_answer: bool = False
) -> t.Tensor:
    answers = batch.answers if not wrong_answer else batch.wrong_answers
    if isinstance(answers, t.Tensor):
        return t.gather(vals, dim=-1, index=answers).mean()
    else:
        assert isinstance(answers, list)
        answer_probs = []
        for prompt_idx, prompt_answers in enumerate(answers):
            answer_probs.append(
                t.gather(vals[prompt_idx], dim=-1, index=prompt_answers).mean()
            )
        return t.stack(answer_probs).mean()


def batch_avg_answer_diff(vals: t.Tensor, batch: PromptPairBatch) -> t.Tensor:
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        ans_avg = t.gather(vals, dim=-1, index=answers).mean()
        wrong_ans_avg = t.gather(vals, dim=-1, index=wrong_answers).mean()
        return ans_avg - wrong_ans_avg
    else:
        assert isinstance(answers, list) and isinstance(wrong_answers, list)
        answer_probs = []
        wrong_answers_probs = []
        for prompt_idx, prompt_answers in enumerate(answers):
            answer_probs.append(
                t.gather(vals[prompt_idx], dim=-1, index=prompt_answers).mean()
            )
            wrong_answers_probs.append(
                t.gather(
                    vals[prompt_idx], dim=-1, index=wrong_answers[prompt_idx]
                ).mean()
            )
        return t.stack(answer_probs).mean() - t.stack(wrong_answers_probs).mean()


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
