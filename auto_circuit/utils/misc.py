from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import Iterator, Set

import torch as t
from torch.utils.hooks import RemovableHandle

from auto_circuit.data import PromptPairBatch


def repo_path_to_abs_path(path: str) -> Path:
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def module_by_name(model: t.nn.Module, module_name: str) -> t.nn.Module:
    return reduce(getattr, [model] + module_name.split("."))  # type: ignore


def set_module_by_name(
    model: t.nn.Module, module_name: str, new_module: t.nn.Module
) -> None:
    parent = model
    if "." in module_name:
        parent = reduce(getattr, [model] + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)


def percent_gpu_mem_used(total_gpu_mib: int = 49000) -> str:
    return (
        "Memory used {:.1f}".format(
            ((t.cuda.memory_allocated() / (2**20)) / total_gpu_mib) * 100
        )
        + "%"
    )


def batch_avg_answer_val(
    vals: t.Tensor, batch: PromptPairBatch, wrong_answer: bool = False
) -> t.Tensor:
    answers = batch.answers if not wrong_answer else batch.wrong_answers
    if isinstance(answers, t.Tensor):
        return t.gather(vals, dim=1, index=answers).mean()
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
        ans_avg = t.gather(vals, dim=1, index=answers).mean()
        wrong_ans_avg = t.gather(vals, dim=1, index=wrong_answers).mean()
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
